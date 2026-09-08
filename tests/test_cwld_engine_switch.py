"""The engine switch added for LCWLD-161 must actually swap the force, and the
two engines must agree.

Guards the seam between `lips` and the plugin, which no other test covers: the
plugin's own suite checks the plugin, `lips`'s tests never loaded it. A silent
failure here means a run labelled `plugin` was produced by CustomGBForce, or
vice versa -- exactly the mix-up `cwld_engine_of` exists to make visible.

Single static energy/force evaluation, zero Integrator.step().
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP = os.path.join(ROOT, "openmm-localcwld", "docs", "reports",
                    "dec005_1aay_dpolar_system.xml")
ISOLATED_GROUP = 31           # the group dec005_dump_1aay.py isolates CWLD into


@pytest.fixture(scope="module")
def swapped():
    import openmm as mm
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from lips.engine import v26

    if not os.path.exists(DUMP):
        pytest.skip(f"dumped System not present: {DUMP}")
    try:
        v26.ensure_localcwld_plugin()
    except RuntimeError as exc:
        pytest.skip(f"plugin unavailable: {exc}")
    if "CUDA" not in {mm.Platform.getPlatform(i).getName()
                      for i in range(mm.Platform.getNumPlatforms())}:
        pytest.skip("no CUDA platform")

    with open(DUMP) as fh:
        gb_system = mm.XmlSerializer.deserialize(fh.read())
    # 2026-09-07：`_swap_to_localcwld` 的参数改成**必填关键字**了，因为原来它不传
    # `globals_`、桥就用自己硬编码的默认值（`zmmOrder: 2` 等），造成
    # 「L_IPS_ZMM_ELL=1 却跑 ell=2」的静默错误物理（回归见 test_plugin_globals.py）。
    #
    # ⚠ ell **必须从 DUMP 自己推断，不能读 L_IPS_ZMM_ELL**：DUMP 是一个冻结的
    # CustomGB System，它的 closure 已经烧在 XML 的 Lepton 串里，环境变量改不了它。
    # 我第一版写成 `zmm_ell=v26.ZMM_CLOSURE_ELL`，于是
    # `L_IPS_ZMM_ELL=1 pytest` 会拿 **CustomGB ell=2 对插件 ell=1**，
    # 报出 p99 |dF| = 105（看着像插件坏了，其实是对照设错了）。
    with open(DUMP) as fh:
        dump_xml = fh.read()
    ell_marks = {1: "1.5/", 2: "15.0/(8.0", 3: "35.0/(16.0"}
    dump_ell = [e for e, mark in ell_marks.items() if mark in dump_xml]
    assert len(dump_ell) == 1, f"无法从 DUMP 判断 closure 阶数：命中 {dump_ell}"
    dump_ell = dump_ell[0]

    return mm, v26, gb_system, v26._swap_to_localcwld(
        gb_system, r_env=0.35, rc=1.2, rho0=13.5, k_polar=0.8,
        q_clamp=v26.Q_DELTA_CLAMP, use_penalty=v26.ENABLE_Q_PENALTY,
        penalty_k=180.0, zmm_ell=dump_ell)


def test_swap_replaces_the_force(swapped):
    _, v26, gb_system, new_system = swapped
    assert v26.cwld_engine_of(gb_system) == "customgb"
    assert v26.cwld_engine_of(new_system) == "plugin"
    assert gb_system.getNumParticles() == new_system.getNumParticles()


def test_bad_engine_name_is_rejected(swapped):
    _, v26, gb_system, _ = swapped
    with pytest.raises(ValueError, match="engine"):
        v26.setup_cwld_lips_system(gb_system, None, engine="localcwld")


def test_engines_agree_on_forces(swapped):
    """Per-atom |dF| on the isolated CWLD group, p99 against DEC-005's atol.

    p99, not max: cutoff-boundary flips give ~120 atoms a legitimate floor.
    """
    import numpy as np
    mm, _, gb_system, new_system = swapped
    from openmm import unit

    with open(DUMP.replace("_system.xml", "_state_mixed.xml")) as fh:
        state = mm.XmlSerializer.deserialize(fh.read())

    def group_forces(system):
        integrator = mm.VerletIntegrator(0.001)
        ctx = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed"})
        ctx.setState(state)
        f = ctx.getState(getForces=True, groups={ISOLATED_GROUP}).getForces(asNumpy=True)
        out = np.array(f.value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
        del ctx, integrator
        return out

    d = np.linalg.norm(group_forces(gb_system) - group_forces(new_system), axis=1)
    assert np.percentile(d, 99) < 1.0, f"p99 |dF| = {np.percentile(d, 99):.3e}"
