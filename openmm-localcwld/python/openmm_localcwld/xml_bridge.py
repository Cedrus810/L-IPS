"""把生产 CWLD 体系里的 ``CustomGBForce`` 换成 ``LocalCWLDForce``，走 XML。

**为什么是 XML 而不是绑定**：没有 SWIG（LCWLD-080 未做，环境里也没有 SWIG /
pybind11 / Cython），所以 Python 构造不出 ``LocalCWLDForce``。但它不需要构造 ——
LCWLD-060 的序列化代理注册之后，``XmlSerializer.deserialize`` 在 **C++ 侧**建对象，
Python 只拿到一个通用 ``Force`` 代理，而 ``System.addForce`` 本来就只要 ``Force*``。

**这里唯一手写的东西是那段 XML**，而手写格式正是 ``dec005_dump_1aay.py`` 开头警告过的
那类错：列序、单位、精度可以静默漂移。所以 :func:`swap_customgb_for_localcwld` 自带
**回环自检**（``verify=True``，默认开）：生成的 XML 交给 C++ 反序列化（每个 setter 都做
校验），再序列化回来，逐字段比对。写错属性名会在 C++ 侧直接抛（``getDoubleProperty``
没有默认值），写错数值会在比对时暴露。**自检不通过就不返回结果。**

⚠ **自检抓得到什么、抓不到什么**（`test_xml_bridge.py` 里有对应的用例）：

* 抓得到：属性名写错（C++ ``getDoubleProperty`` 没有默认值，直接抛）、数值越界
  （每个 setter 都校验）、结构错、以及**C++ 会改写的值**（比如 exclusion 被规范化成
  (min,max)，写 (3,1) 读回来是 (1,3)）。
* **抓不到：一个合法但本来就错的值。** C++ 会忠实地把它转一圈还给你。
  防这一类的是 :func:`_read_customgb` 的**按名取列**，以及
  ``test_parameters_match_the_customgb_source`` —— 它拿结果去对 CustomGBForce 源头，
  那才是数值正确性的守门人。回环自检守的是格式和映射，不是数值。

九个逐粒子参数**按名字取**，不按下标 —— 跟 C++ 验收测试同一条规矩：以后谁在 v26 里插一个
``addPerParticleParameter``，列错位必须响，不能静默改物理。

用法::

    from openmm_localcwld.xml_bridge import swap_customgb_for_localcwld
    new_xml = swap_customgb_for_localcwld(open("system.xml").read())
    system = openmm.XmlSerializer.deserialize(new_xml)

调用前必须先 ``Platform.loadPluginsFromDirectory(<插件目录>)``，否则代理没注册。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

#: 逐粒子参数：CustomGBForce 里的名字 -> LocalCWLDForce XML 属性名。
#: 两侧都用名字，所以这张表是唯一的对应关系，改名会立刻报 KeyError。
PARTICLE_PARAMETER_MAP = {
    "qbase": "qbase",
    "charge_mod": "chargeMod",
    "dpolar": "dpolar",
    "is_polar": "isPolar",
    "dens_source": "densSource",
    "dens_sink": "densSink",
    "source_class_weight": "sourceClassWeight",
    "static_phase": "staticPhase",
}
#: 单独列出：它是整数，且是 addParticle 的第九个参数。
RESIDUE_ID_PARAMETER = "mol_id"

#: 全局参数默认值。与 ``LocalCWLDForce`` 的 C++ 构造函数一致（plan 17.1），
#: 而 C++ 那套默认值已由 DEC-005 验收对着生产 CustomGBForce 验证过。
DEFAULT_GLOBALS = {
    "environmentCutoff": 0.35,
    "cutoffDistance": 1.2,
    "zmmOrder": 2,
    "rho0": 13.5,
    "kPolar": 0.8,
    "chargeDeltaClamp": 0.2,
    "useQPenalty": False,
    "qPenaltyStrength": 180.0,
    "one4PiEps0": 138.935458,
}

#: 代理写出的格式版本。跟 C++ 侧 SERIALIZATION_VERSION 对齐。
SERIALIZATION_VERSION = 1


def _children(parent: ET.Element, tag: str) -> list[ET.Element]:
    """``parent.find(tag)`` 的子节点；缺节点返回空表。

    不写 ``find(tag) or []``：Element 的真值等于"有没有子节点"，一个存在但为空的
    节点会被判成假，跟"节点不存在"混为一谈。
    """
    node = parent.find(tag)
    return [] if node is None else list(node)


class BridgeError(RuntimeError):
    """XML 桥接失败。永远带上失败的具体字段，不吞。"""


def _fmt(value: float) -> str:
    # 17 位有效数字足以让 double 无损回环；少一位就可能在自检里表现为
    # "只差最后一位"，那种差异最容易被当成噪声放过去。
    return repr(float(value))


def build_local_cwld_force_element(
    per_particle: list[dict],
    exclusions: list[tuple[int, int]],
    *,
    force_group: int = 0,
    name: str = "LocalCWLDForce",
    globals_: dict | None = None,
) -> ET.Element:
    """造一个 ``<Force type="LocalCWLDForce">`` 元素。"""
    g = dict(DEFAULT_GLOBALS)
    if globals_:
        unknown = set(globals_) - set(DEFAULT_GLOBALS)
        if unknown:
            raise BridgeError(f"未知的全局参数：{sorted(unknown)}")
        g.update(globals_)

    force = ET.Element("Force", {
        "type": "LocalCWLDForce",
        "version": str(SERIALIZATION_VERSION),
        "forceGroup": str(force_group),
        "name": name,
        "method": "0",                       # CutoffPeriodic，v0.1 唯一取值
        "environmentCutoff": _fmt(g["environmentCutoff"]),
        "cutoffDistance": _fmt(g["cutoffDistance"]),
        "zmmOrder": str(int(g["zmmOrder"])),
        "rho0": _fmt(g["rho0"]),
        "kPolar": _fmt(g["kPolar"]),
        "chargeDeltaClamp": _fmt(g["chargeDeltaClamp"]),
        "useQPenalty": "1" if g["useQPenalty"] else "0",
        "qPenaltyStrength": _fmt(g["qPenaltyStrength"]),
        "one4PiEps0": _fmt(g["one4PiEps0"]),
    })
    particles = ET.SubElement(force, "Particles")
    for p in per_particle:
        attrs = {xml_name: _fmt(p[xml_name]) for xml_name in PARTICLE_PARAMETER_MAP.values()}
        attrs["residueId"] = str(int(p["residueId"]))
        ET.SubElement(particles, "Particle", attrs)
    excl = ET.SubElement(force, "Exclusions")
    for p1, p2 in exclusions:
        ET.SubElement(excl, "Exclusion", {"p1": str(int(p1)), "p2": str(int(p2))})
    return force


def _read_customgb(gb: ET.Element) -> tuple[list[dict], list[tuple[int, int]]]:
    """从 CustomGBForce 元素里按名字读九个逐粒子参数和全部 exclusions。"""
    names = [e.get("name") for e in _children(gb, "PerParticleParameters")]
    if not names:
        raise BridgeError("CustomGBForce 里没有 PerParticleParameters")
    needed = set(PARTICLE_PARAMETER_MAP) | {RESIDUE_ID_PARAMETER}
    missing = needed - set(names)
    if missing:
        raise BridgeError(f"CustomGBForce 缺少逐粒子参数：{sorted(missing)}；"
                          f"实际有 {names}")
    column = {n: i for i, n in enumerate(names)}

    per_particle = []
    for particle in _children(gb, "Particles"):
        # OpenMM 把逐粒子参数写成 param1/param2/...，顺序与 PerParticleParameters 一致。
        vals = [float(particle.get(f"param{i + 1}")) for i in range(len(names))]
        row = {xml_name: vals[column[gb_name]]
               for gb_name, xml_name in PARTICLE_PARAMETER_MAP.items()}
        row["residueId"] = int(round(vals[column[RESIDUE_ID_PARAMETER]]))
        per_particle.append(row)
    if not per_particle:
        raise BridgeError("CustomGBForce 里没有粒子")

    exclusions = [(int(e.get("p1")), int(e.get("p2")))
                  for e in _children(gb, "Exclusions")]
    return per_particle, exclusions


def _compare_roundtrip(written: ET.Element, read_back: ET.Element) -> None:
    """逐字段比对我们写出的和 C++ 再序列化出来的，不一致就抛。"""
    for key, value in written.attrib.items():
        got = read_back.get(key)
        if got is None:
            raise BridgeError(f"回环丢了属性 {key!r}")
        try:
            if float(got) != float(value):
                raise BridgeError(f"回环后 {key!r} 变了：写 {value} 读 {got}")
        except ValueError:
            if got != value:
                raise BridgeError(f"回环后 {key!r} 变了：写 {value!r} 读 {got!r}")

    for section, child in (("Particles", "Particle"), ("Exclusions", "Exclusion")):
        a = _children(written, section)
        b = _children(read_back, section)
        if len(a) != len(b):
            raise BridgeError(f"回环后 {section} 数量变了：{len(a)} -> {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            for key, value in x.attrib.items():
                got = y.get(key)
                if got is None or float(got) != float(value):
                    raise BridgeError(
                        f"回环后 {section}[{i}].{key} 变了：写 {value} 读 {got}")


def swap_customgb_for_localcwld(system_xml: str, *, globals_: dict | None = None,
                                verify: bool = True) -> str:
    """把 System XML 里的 ``CustomGBForce`` 换成等价的 ``LocalCWLDForce``。

    :param verify: 默认开。关掉它就等于相信一段手写 XML，不要关。
    """
    root = ET.fromstring(system_xml)
    forces = root.find("Forces")
    if forces is None:
        raise BridgeError("System XML 里没有 <Forces>")
    gbs = [f for f in forces if f.get("type") == "CustomGBForce"]
    if len(gbs) != 1:
        raise BridgeError(f"期望恰好一个 CustomGBForce，实际 {len(gbs)} 个")
    gb = gbs[0]

    per_particle, exclusions = _read_customgb(gb)
    force = build_local_cwld_force_element(
        per_particle, exclusions,
        force_group=int(gb.get("forceGroup", 0)),
        globals_=globals_)

    index = list(forces).index(gb)
    forces.remove(gb)
    forces.insert(index, force)
    new_xml = ET.tostring(root, encoding="unicode")

    if verify:
        import openmm as mm
        try:
            system = mm.XmlSerializer.deserialize(new_xml)
        except Exception as exc:                       # noqa: BLE001
            raise BridgeError(
                f"C++ 侧拒绝了生成的 XML：{exc}. 插件加载了吗？"
                "（先 Platform.loadPluginsFromDirectory）") from exc
        back = ET.fromstring(mm.XmlSerializer.serialize(system))
        got = [f for f in back.find("Forces") if f.get("type") == "LocalCWLDForce"]
        if len(got) != 1:
            raise BridgeError(f"回环后 LocalCWLDForce 有 {len(got)} 个")
        _compare_roundtrip(force, got[0])

    return new_xml
