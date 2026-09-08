"""LCWLD-060 的实际兑现：不用 SWIG，从 Python 拿到含 LocalCWLDForce 的 System。

用真实的 1AAY dump（`docs/reports/dec005_1aay_dpolar_system.xml`，32794 粒子、
38998 条 exclusion）。不建 Context、不跑 MD —— 只做反序列化和逐字段比对。
"""
from __future__ import annotations

import ctypes
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

openmm = pytest.importorskip("openmm")

from openmm_localcwld.xml_bridge import (  # noqa: E402
    PARTICLE_PARAMETER_MAP,
    RESIDUE_ID_PARAMETER,
    BridgeError,
    swap_customgb_for_localcwld,
)

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "docs/reports/dec005_1aay_dpolar_system.xml"


@pytest.fixture(scope="module")
def plugin_loaded():
    """把 API 库拉进进程，触发序列化代理的 constructor 注册。

    直接 CDLL 而不是 loadPluginsFromDirectory：序列化不需要 CUDA 平台，
    走插件目录会把这个测试拴在 GPU 上。
    """
    for build in ("build-cuda", "build"):
        so = ROOT / build / "openmmapi" / "libOpenMMLocalCWLD.so"
        if so.exists():
            ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            return so
    pytest.skip("libOpenMMLocalCWLD.so 没找到，先 cmake --build")


@pytest.fixture(scope="module")
def dump_xml():
    if not DUMP.exists():
        pytest.skip(f"没有 dump：{DUMP}")
    return DUMP.read_text()


def test_roundtrip_on_real_1aay(plugin_loaded, dump_xml):
    """换完之后 C++ 能读回来，且逐字段与写出的一致（verify=True 自带比对）。"""
    new_xml = swap_customgb_for_localcwld(dump_xml)
    system = openmm.XmlSerializer.deserialize(new_xml)

    types = [type(system.getForce(i)).__name__ for i in range(system.getNumForces())]
    assert "CustomGBForce" not in types
    assert sum(t == "Force" or "LocalCWLD" in t for t in types) >= 1, types
    assert system.getNumParticles() == 32794


def test_parameters_match_the_customgb_source(plugin_loaded, dump_xml):
    """九个逐粒子参数确实按名字搬过去了，不是按下标。

    这是这套桥接唯一可能静默出错的地方：列错位不会抛，只会算出别的物理。
    """
    gb = [f for f in ET.fromstring(dump_xml).find("Forces")
          if f.get("type") == "CustomGBForce"][0]
    names = [e.get("name") for e in gb.find("PerParticleParameters")]
    column = {n: i for i, n in enumerate(names)}
    src = list(gb.find("Particles"))

    new = [f for f in ET.fromstring(swap_customgb_for_localcwld(dump_xml)).find("Forces")
           if f.get("type") == "LocalCWLDForce"][0]
    dst = list(new.find("Particles"))
    assert len(dst) == len(src)

    for i in (0, 1, 573, 1416, len(src) - 1):        # 含配位原子和金属
        for gb_name, xml_name in PARTICLE_PARAMETER_MAP.items():
            want = float(src[i].get(f"param{column[gb_name] + 1}"))
            got = float(dst[i].get(xml_name))
            assert got == want, f"粒子 {i} 的 {gb_name} -> {xml_name}"
        want_res = int(round(float(src[i].get(f"param{column[RESIDUE_ID_PARAMETER] + 1}"))))
        assert int(dst[i].get("residueId")) == want_res


def test_exclusions_are_carried_over(plugin_loaded, dump_xml):
    gb = [f for f in ET.fromstring(dump_xml).find("Forces")
          if f.get("type") == "CustomGBForce"][0]
    src = [(int(e.get("p1")), int(e.get("p2"))) for e in gb.find("Exclusions")]
    new = [f for f in ET.fromstring(swap_customgb_for_localcwld(dump_xml)).find("Forces")
           if f.get("type") == "LocalCWLDForce"][0]
    dst = [(int(e.get("p1")), int(e.get("p2"))) for e in new.find("Exclusions")]
    assert len(dst) == len(src) == 38998
    assert dst == src


def test_verification_catches_a_wrong_attribute_name(plugin_loaded, dump_xml,
                                                    monkeypatch):
    """属性名写错必须抛 —— C++ 的 getDoubleProperty 没有默认值。"""
    import openmm_localcwld.xml_bridge as bridge
    bad = dict(bridge.PARTICLE_PARAMETER_MAP)
    bad["qbase"] = "qbse"
    monkeypatch.setattr(bridge, "PARTICLE_PARAMETER_MAP", bad)
    with pytest.raises(BridgeError, match="C\\+\\+ 侧拒绝"):
        bridge.swap_customgb_for_localcwld(dump_xml)


def test_verification_catches_a_value_cpp_rewrites(plugin_loaded, dump_xml,
                                                   monkeypatch):
    """C++ 会改写的值必须被比对抓到：exclusion 规范化成 (min, max)。"""
    import openmm_localcwld.xml_bridge as bridge
    real = bridge.build_local_cwld_force_element

    def swapped(per_particle, exclusions, **kw):
        return real(per_particle, [(p2, p1) for p1, p2 in exclusions], **kw)

    monkeypatch.setattr(bridge, "build_local_cwld_force_element", swapped)
    with pytest.raises(BridgeError, match="Exclusion"):
        bridge.swap_customgb_for_localcwld(dump_xml)


def test_verification_cannot_catch_a_merely_wrong_value(plugin_loaded, dump_xml,
                                                        monkeypatch):
    """把自检的**边界**钉死：合法但错的数值，回环自检抓不到。

    截断精度不会让任何 setter 抛，C++ 会忠实地把错值转一圈还回来，
    比对两侧因此一致。守这一类的是按名取列 + 对着 CustomGBForce 源头比，
    不是回环。**这个用例存在的意义是：将来谁想把回环自检当数值守门人时，
    这里会告诉他不行。**
    """
    import openmm_localcwld.xml_bridge as bridge
    monkeypatch.setattr(bridge, "_fmt", lambda v: f"{float(v):.3g}")
    new_xml = bridge.swap_customgb_for_localcwld(dump_xml)      # 不抛
    force = [f for f in ET.fromstring(new_xml).find("Forces")
             if f.get("type") == "LocalCWLDForce"][0]
    qbase = float(list(force.find("Particles"))[1].get("qbase"))
    gb = [f for f in ET.fromstring(dump_xml).find("Forces")
          if f.get("type") == "CustomGBForce"][0]
    want = float(list(gb.find("Particles"))[1].get("param1"))
    assert qbase != want, "截断应该真的改变了数值，否则这个用例什么也没说"


def test_unknown_global_is_rejected(plugin_loaded, dump_xml):
    with pytest.raises(BridgeError, match="未知的全局参数"):
        swap_customgb_for_localcwld(dump_xml, globals_={"notAThing": 1.0})
