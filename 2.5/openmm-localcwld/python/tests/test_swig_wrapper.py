"""LCWLD-080：SWIG 包装器，与 LCWLD-060 的 XML 路径互证。

**先说清楚各道防线守的是什么** —— 初版这里写着"互证是 `.i` 漂移的守门人"，
变异测试证明那是夸大的：

* `.i` 少一个参数 / 类型对不上 ⇒ **编译期就挂**。SWIG 生成的是对真头文件的真调用，
  签名对不上没有匹配的重载。这条不需要运行时测试。
* `.i` 里把两个同类型参数的名字对调 ⇒ **没有任何影响**，测试也确实抓不到。
  生成的调用是纯位置的，`.i` 里的名字只是标签。所以这不是一个真的漂移风险。
* **C++ 头文件本身把参数顺序改了** ⇒ 两条路径会**一起**错（XML 代理也是位置调用），
  互证看不见。守这一条的是 `TestCudaLocalCWLDAcceptance` —— 它拿逐原子力去对
  生产 CustomGBForce，参数换位必然改变力。

**那互证守的是什么**：`xml_bridge` 那侧的**列映射**（它自己解析 `param{i}`、按名字
建表，那是纯 Python 逻辑，编译器管不着），以及两条路径产出的体系确实等价 ——
060 的产物能不能被 080 的类型化 API 接着用。

在真实 1AAY（32794 粒子、38998 条 exclusion）上跑。不建 Context、不跑 MD。
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

openmm = pytest.importorskip("openmm")

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "docs/reports/dec005_1aay_dpolar_system.xml"

#: CustomGBForce 参数名 -> addParticle 的位置。冻结顺序（plan 17.2）。
GB_ORDER = ("qbase", "charge_mod", "dpolar", "is_polar", "dens_source",
            "dens_sink", "source_class_weight", "static_phase", "mol_id")


@pytest.fixture(scope="module")
def localcwld():
    for build in ("build-cuda", "build"):
        d = ROOT / build / "python"
        if (d / "_localcwld.so").exists():
            sys.path.insert(0, str(d))
            import localcwld as mod
            return mod
    pytest.skip("_localcwld.so 没找到；用 -DLOCALCWLD_BUILD_PYTHON=ON 构建")


@pytest.fixture(scope="module")
def dump_xml():
    if not DUMP.exists():
        pytest.skip(f"没有 dump：{DUMP}")
    return DUMP.read_text()


def test_defaults_match_the_cpp_constructor(localcwld):
    """默认值是 DEC-005 验收赖以成立的前提（验收测试也不设全局参数）。"""
    f = localcwld.LocalCWLDForce()
    assert f.getEnvironmentCutoff() == 0.35
    assert f.getCutoffDistance() == 1.2
    assert f.getZMMOrder() == 2
    assert f.getRho0() == 13.5
    assert f.getKPolar() == 0.8
    assert f.getChargeDeltaClamp() == 0.2
    assert f.getUseQPenalty() is False
    assert f.getQPenaltyStrength() == 180.0
    assert f.getOne4PiEps0() == 138.935458
    assert f.getNonbondedMethod() == localcwld.LocalCWLDForce.CutoffPeriodic
    assert f.usesPeriodicBoundaryConditions() is True


def test_particle_parameter_order_survives_the_wrapper(localcwld):
    """九个参数进去什么顺序，出来必须是什么顺序。

    `.i` 里把两个 double 参数写反了不会有任何编译错误，只会把物理换掉。
    用九个互不相同的值，任何置换都会被抓到。
    """
    f = localcwld.LocalCWLDForce()
    sent = (-0.834, 0.021, -0.15, 1.0, 0.5, 0.25, 1.5, 0.75, 42)
    f.addParticle(*sent)
    got = tuple(f.getParticleParameters(0))
    assert got == sent, f"参数顺序被打乱：送入 {sent} 取回 {got}"


def test_exclusions_are_normalized(localcwld):
    f = localcwld.LocalCWLDForce()
    for _ in range(5):
        f.addParticle(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0)
    f.addExclusion(3, 1)
    assert tuple(f.getExclusionParticles(0)) == (1, 3)
    with pytest.raises(Exception):
        f.addExclusion(2, 2)          # 自排除
    with pytest.raises(Exception):
        f.addExclusion(1, 3)          # 重复


def test_setters_validate(localcwld):
    f = localcwld.LocalCWLDForce()
    with pytest.raises(Exception):
        f.setEnvironmentCutoff(-1.0)
    with pytest.raises(Exception):
        f.setZMMOrder(7)
    with pytest.raises(Exception):
        f.setRho0(0.0)


def _force_from_swig(localcwld, gb):
    """走类型化 API 从 CustomGBForce 造一个 LocalCWLDForce（路径 B）。"""
    names = [gb.getPerParticleParameterName(i)
             for i in range(gb.getNumPerParticleParameters())]
    column = {n: i for i, n in enumerate(names)}
    force = localcwld.LocalCWLDForce()
    for i in range(gb.getNumParticles()):
        p = gb.getParticleParameters(i)
        args = [p[column[n]] for n in GB_ORDER[:-1]]
        force.addParticle(*args, int(round(p[column["mol_id"]])))
    for i in range(gb.getNumExclusions()):
        a, b = gb.getExclusionParticles(i)
        force.addExclusion(a, b)
    return force


def test_swig_and_xml_paths_agree_on_real_1aay(localcwld, dump_xml):
    """两条独立路径在真实体系上逐字段一致。这是 `.i` 漂移的守门人。"""
    from openmm_localcwld.xml_bridge import swap_customgb_for_localcwld

    # 路径 A：XML 桥接（自己解析 param{i}，手写 XML，C++ 反序列化）
    a_force = [f for f in ET.fromstring(swap_customgb_for_localcwld(dump_xml))
               .find("Forces") if f.get("type") == "LocalCWLDForce"][0]
    a_particles = list(a_force.find("Particles"))
    a_excl = [(int(e.get("p1")), int(e.get("p2")))
              for e in a_force.find("Exclusions")]

    # 路径 B：SWIG 类型化 API
    system = openmm.XmlSerializer.deserialize(dump_xml)
    gb = next(system.getForce(i) for i in range(system.getNumForces())
              if isinstance(system.getForce(i), openmm.CustomGBForce))
    b = _force_from_swig(localcwld, gb)

    assert b.getNumParticles() == len(a_particles) == 32794
    assert b.getNumExclusions() == len(a_excl) == 38998

    xml_names = ("qbase", "chargeMod", "dpolar", "isPolar", "densSource",
                 "densSink", "sourceClassWeight", "staticPhase")
    for i in range(b.getNumParticles()):
        got = b.getParticleParameters(i)
        want = a_particles[i]
        for k, name in enumerate(xml_names):
            assert got[k] == float(want.get(name)), f"粒子 {i} 的 {name}"
        assert got[8] == int(want.get("residueId")), f"粒子 {i} 的 residueId"

    for i in range(b.getNumExclusions()):
        assert tuple(b.getExclusionParticles(i)) == a_excl[i], f"exclusion {i}"


def test_deserialized_force_can_be_cast_back(localcwld, dump_xml):
    """System.getForce() 给的是通用 Force —— cast/isinstance 必须能把它认回来。

    没有这两个，060 的 XML 路径产出的体系就没法用类型化 API 再操作
    （比如改 dpolar 后 updateParametersInContext）。
    """
    from openmm_localcwld.xml_bridge import swap_customgb_for_localcwld
    system = openmm.XmlSerializer.deserialize(swap_customgb_for_localcwld(dump_xml))
    found = [system.getForce(i) for i in range(system.getNumForces())
             if localcwld.LocalCWLDForce.isinstance(system.getForce(i))]
    assert len(found) == 1
    typed = localcwld.LocalCWLDForce.cast(found[0])
    assert typed.getNumParticles() == 32794
    assert typed.getEnvironmentCutoff() == 0.35
