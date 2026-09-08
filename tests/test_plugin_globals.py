"""插件路径必须真的用引擎给的参数 —— 这条回归就是 2026-09-07 那个 bug 的探针。

那个 bug：`_swap_to_localcwld()` 调 xml_bridge 时不传 `globals_`，桥用自己的
`DEFAULT_GLOBALS`（`zmmOrder: 2` 与全部引擎参数硬编码），且**从不读 pair 表达式**。
实测后果是 `L_IPS_ZMM_ELL=1` 下日志报 zmm1、插件实际跑 ZMMOrder=2。

它能潜伏是因为：
  * 所有生产跑恰好用的就是那套默认值；
  * DEC-005 的验收（plugin vs CustomGB 逐原子力）**也是在默认参数下做的**，
    所以那套验收结构上抓不到参数没传下去这件事。

⇒ 所以守卫不能只有「力对不对」，还必须有「参数有没有到」。本文件守后者。
"""
import os

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _load():
    """加载 1AAY 参考体系 + 引擎；插件不可用就 skip（CI/无 GPU 机器）。"""
    from lips import systems as ls
    from lips.engine import v26
    try:
        v26.ensure_localcwld_plugin()
        import localcwld                      # noqa: F401
    except Exception as exc:                  # noqa: BLE001
        pytest.skip(f"LocalCWLD 插件不可用：{exc}")
    top, system = ls.load_reference_system(name="1aay")
    return top, system, v26, localcwld


@pytest.fixture(scope="module")
def base():
    if os.environ.get("L_IPS_SKIP_PLUGIN_TESTS"):
        pytest.skip("L_IPS_SKIP_PLUGIN_TESTS 已设")
    return _load()


def _plugin_force(system, localcwld):
    hits = [f for f in system.getForces() if localcwld.LocalCWLDForce.isinstance(f)]
    assert len(hits) == 1, f"应当恰好有一个 LocalCWLDForce，实际 {len(hits)}"
    return localcwld.LocalCWLDForce.cast(hits[0])


@pytest.mark.parametrize("ell", [1, 2, 3])
def test_zmm_order_reaches_the_plugin(base, monkeypatch, ell):
    """引擎选的 ell 必须真的到插件里 —— 这是原 bug 的直接探针。"""
    top, system, v26, localcwld = base
    monkeypatch.setattr(v26, "ZMM_CLOSURE_ELL", ell)
    s = v26.setup_cwld_lips_system(system, top, engine="plugin")
    lf = _plugin_force(s, localcwld)
    assert lf.getZMMOrder() == ell, (
        f"引擎选 ell={ell}，插件里却是 {lf.getZMMOrder()} —— "
        f"参数没有传到插件（原 bug 复现）。日志会照样报 ell={ell}，"
        f"所以这种不一致在产物里看不出来。")
    # 标签与实际必须一致，否则 CSV 会说谎
    assert f"zmm{ell}" in v26.closure_label()


def test_engine_parameters_reach_the_plugin(base, monkeypatch):
    """非默认的 r_env/rc/rho0/k_polar/clamp 也必须到插件，不能被桥的默认值吃掉。"""
    top, system, v26, localcwld = base
    monkeypatch.setattr(v26, "Q_DELTA_CLAMP", 0.17)
    s = v26.setup_cwld_lips_system(
        system, top, engine="plugin",
        r_env=0.31, rc=1.15, rho0=12.25, k_polar=0.73)
    lf = _plugin_force(s, localcwld)
    got = dict(r_env=lf.getEnvironmentCutoff(), rc=lf.getCutoffDistance(),
               rho0=lf.getRho0(), k_polar=lf.getKPolar(),
               clamp=lf.getChargeDeltaClamp())
    want = dict(r_env=0.31, rc=1.15, rho0=12.25, k_polar=0.73, clamp=0.17)
    bad = {k: (got[k], want[k]) for k in want if abs(got[k] - want[k]) > 1e-12}
    assert not bad, (
        f"这些参数没传到插件（got, want）：{bad}\n"
        f"  ⇒ 插件路径上的参数扫描会静默跑成默认值。")


def test_non_zmm_closure_is_refused_not_silently_downgraded(base, monkeypatch):
    """插件表达不了的 closure 必须抛错 —— 静默回落成 ZMM ell=2 是最坏的失败。"""
    top, system, v26, localcwld = base
    monkeypatch.setattr(v26, "L_IPS_CLOSURE", "pcf1_c1")
    with pytest.raises(RuntimeError, match="表达不了"):
        v26.setup_cwld_lips_system(system, top, engine="plugin")


def test_customgb_path_still_uses_the_selected_closure(base, monkeypatch):
    """对照：customgb 路径是靠 Lepton 串走的，本来就跟着 ell 变。"""
    top, system, v26, _ = base
    import openmm as mm
    monkeypatch.setattr(v26, "ZMM_CLOSURE_ELL", 1)
    s = v26.setup_cwld_lips_system(system, top, engine="customgb")
    gb = [f for f in s.getForces() if isinstance(f, mm.CustomGBForce)]
    assert len(gb) == 1
    expr = "".join(gb[0].getEnergyTermParameters(i)[0]
                   for i in range(gb[0].getNumEnergyTerms()))
    assert "1.5" in expr, "ell=1 的 closure 常数项 1.5 应出现在 Lepton 串里"


# --------------------------------------------------------------------------
# ell=1 的**力**验收：参数到位 ≠ 力算对。
#
# DEC-005 那套 oracle（plugin vs 生产 CustomGBForce，逐原子力）此前**只在 ell=2
# 上跑过**，因为 ell=2 是唯一能真正到达插件的值（就是被修掉的那个 bug）。
# 所以在把生产默认换成 ell=1 之前，必须在 ell=1 上重跑一次同样的对照。
#
# ⚠ 与 `test_cwld_engine_switch.py` 的区别：那个用**冻结的 dump**（烧死 ell=2），
# 环境变量改不了它 —— 拿 `L_IPS_ZMM_ELL=1` 去跑它会变成「CustomGB ell=2 对
# 插件 ell=1」，报 p99 |dF| ≈ 105，看着像插件坏了，其实是对照设错了。
# 这里**两侧都现建在同一个 ell 上**，所以才是有效的验收。
CWLD_GROUP = 1                     # v26.CWLD_GROUP：CWLD 力被隔离到这一组
DEC005_ATOL = 1.0                  # DEC-005 的逐原子力阈值（ell=2 实测 p99 3.3e-3）


@pytest.mark.parametrize("ell", [1, 2])
def test_plugin_matches_customgb_forces_at_each_ell(base, monkeypatch, ell):
    """同一个 ell 下，插件与生产 CustomGBForce 的逐原子力必须一致（DEC-005 口径）。"""
    import numpy as np
    import openmm as mm
    from openmm import app, unit

    top, system, v26, localcwld = base
    if "CUDA" not in {mm.Platform.getPlatform(i).getName()
                      for i in range(mm.Platform.getNumPlatforms())}:
        pytest.skip("没有 CUDA 平台")

    monkeypatch.setattr(v26, "ZMM_CLOSURE_ELL", ell)
    from lips import paths
    pdb = app.PDBFile(paths.system_dir("1AAY", "amber", "solvated.pdb"))
    pos, box = pdb.positions, pdb.topology.getPeriodicBoxVectors()

    def cwld_forces(engine):
        s = v26.setup_cwld_lips_system(system, top, engine=engine)
        # ⚠ 必须**显式分组**：`assign_cwld_mts_force_groups` 只在 MTS 路径被调用，
        # 正常路径下所有力都留在 group 0。第一版直接取 `groups={1}`，于是两边都拿到
        # **全零**力、差值精确为 0、测试"通过" —— 一个零信号的假通过。
        for f in s.getForces():
            is_cwld = (isinstance(f, mm.CustomGBForce)
                       or localcwld.LocalCWLDForce.isinstance(f))
            f.setForceGroup(CWLD_GROUP if is_cwld else 0)
        integ = mm.VerletIntegrator(0.001)
        ctx = mm.Context(s, integ, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed"})
        ctx.setPositions(pos)
        if box is not None:
            ctx.setPeriodicBoxVectors(*box)
        f = ctx.getState(getForces=True, groups={CWLD_GROUP}).getForces(asNumpy=True)
        out = np.array(f.value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
        del ctx, integ
        return out

    ref = cwld_forces("customgb")
    new_ = cwld_forces("plugin")

    # ---- 信号必须存在，否则这个对照是空的 ----
    # 这条断言就是上面那个假通过的探针：零信号时它先红，而不是让 p99=0 假装成功。
    ref_rms = float(np.sqrt((np.linalg.norm(ref, axis=1) ** 2).mean()))
    assert ref_rms > 1.0, (
        f"参照的 CWLD 力 RMS = {ref_rms:.3e} kJ/mol/nm，太小 —— "
        f"这个对照没有信号（多半是 force group 取空了），p99 的通过毫无意义。")

    d = np.linalg.norm(ref - new_, axis=1)
    p99 = float(np.percentile(d, 99))
    # p99 而不是 max：cutoff 边界上的翻转会给约 120 个原子一个合理的地板
    assert p99 < DEC005_ATOL, (
        f"ell={ell}：插件与 CustomGB 的逐原子力 p99 |dF| = {p99:.3e} "
        f"（DEC-005 阈值 {DEC005_ATOL}，ell=2 实测 3.3e-3）")
    print(f"\n  ell={ell}: 参照力 RMS = {ref_rms:.3e} | p99 |dF| = {p99:.3e} "
          f"| max = {d.max():.3e} | 相对 p99/RMS = {p99/ref_rms:.2e} "
          f"(DEC-005 atol {DEC005_ATOL})")
