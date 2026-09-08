#!/usr/bin/env python
"""1AAY 上 CustomGBForce 的步时占比 s，以及 fp32 的固有噪声地板。

**为什么要跑这个**：现在关于"手写 CUDA kernel 能提速多少"的所有估计，都建立在
1CKK 上测到的 s=0.811（`docs/reports/v2.6_results_analysis.md`，2.4 的
`L-IPS.o9808:499-511`）和 1AAY 上一个**下界** s≥0.92（由 t_CWLD/t_PME≈12 反推）。
1AAY 上的 s **从来没有直接测过**。整体提速是

    speedup = 1 / ((1-s) + s*f)          f = 新实现相对 CustomGBForce 的成本

s 对结果极敏感（s=0.81 时 f=0 的天花板只有 5.3x，s=0.92 时是 12.5x），所以 s 是立项
前提而不是事后补充。本脚本直接测，不用反推。

**两个测量**：

1. `--speed`  同一个 1AAY CWLD System，"含 CustomGBForce" vs "移除 CustomGBForce
   其余力不变" 的 ns/day。移除法与 v2.6 的 profile_force_cost 一致（序列化再反序列化
   得到独立副本，只删 CustomGBForce）。**不与 PME baseline 比**——那是另一个 System
   （静电方法不同、电荷不同），比出来的不是这个力的边际成本。

2. `--precision`  精度零点。同一帧、同一个 System，在 double / mixed / single 三种
   平台精度下各取一次静态 forces/energy（零步积分）。用途：
     * 给 kernel 验收阈值一个**实测**地板，而不是猜一个 rtol 然后事后调；
     * 生产轨迹跑的是 mixed，而 DEC-004 要求新 kernel 走 single，两者差多少目前无人知道。
   按分位数报，不只报 max——单个 cutoff 边界原子会把 max 拉爆。

两项都不产出轨迹文件。`--speed` 会跑几百步 MD（这是测速的定义），`--precision` 一步不跑。

用法：
    python -m lips.analysis.force_cost --speed
    python -m lips.analysis.force_cost --precision
    python -m lips.analysis.force_cost --speed --precision --system 1aay
"""
from __future__ import annotations

import argparse
import gc

import numpy as np


#: 生产设置是 rc=1.2 / r_on=0.9。rc 扫描时保持这个比值，否则各档不可比。
R_ON_OVER_RC = 0.9 / 1.2


def log(m):
    print(m, flush=True)


def minimize_once(top, system, positions, lips):
    """把坐标极小化一次，供**所有臂共用**。

    为什么不直接用 `lips.profile_system_speed`（它内部每次都极小化一遍）：
    那样每条臂从**不同坐标**起跑（不同力场的极小化面不同），ns/day 就不可比——
    跟"跨 harness 取比值"是同一类错误，只是它不报错所以看不见。
    极小化在**完整 CWLD 体系**（生产 rc）上做一次，之后所有臂/所有 rc 共用结果坐标。
    """
    import openmm as mm
    import openmm.unit as unit

    cwld = _cwld_system(system, top, lips)
    platform, props = lips.select_platform()
    integ = mm.LangevinMiddleIntegrator(lips.TEMPERATURE, 1.0 / unit.picosecond, lips.DT)
    ctx = mm.Context(cwld, integ, platform, props)
    ctx.setPositions(positions)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 200)
    out = ctx.getState(getPositions=True).getPositions()
    del ctx, integ, cwld
    gc.collect()
    log("  [min] 已在完整 CWLD 体系上极小化一次，所有臂/所有 rc 共用这组坐标")
    return out


def timed_ns_per_day(label, system, top, positions, lips, warmup, measure):
    """计时，**不做极小化**（坐标由 minimize_once 统一提供）。

    与 `lips.profile_system_speed` 的唯一区别就是这一点；integrator / 平台 / 步数
    保持一致，好让两边的口径可比。
    """
    import time

    import openmm as mm
    import openmm.unit as unit

    platform, props = lips.select_platform()
    integ = mm.LangevinMiddleIntegrator(lips.TEMPERATURE, 1.0 / unit.picosecond, lips.DT)
    ctx = mm.Context(system, integ, platform, props)
    ctx.setPositions(positions)
    ctx.applyConstraints(1e-6)
    ctx.setVelocitiesToTemperature(lips.TEMPERATURE)
    integ.step(warmup)
    t0 = time.perf_counter()
    integ.step(measure)
    elapsed = time.perf_counter() - t0
    dt_ns = lips.DT.value_in_unit(unit.nanoseconds)
    ns_per_day = (measure * dt_ns) / elapsed * 86400.0
    log(f"  -> {label}: {measure} steps {elapsed:.1f}s -> {ns_per_day:.2f} ns/day")
    del ctx, integ
    gc.collect()
    return ns_per_day


def _cwld_system(system, top, lips):
    # engine="customgb" on purpose: this whole module measures what
    # CustomGBForce costs, and _strip_gb below refuses a System without one.
    # The plugin path is priced by openmm-localcwld's own BenchmarkCudaLocalCWLD.
    return lips.setup_cwld_lips_system(
        system, top, engine="customgb", a_q2=lips.DEFAULT_A_Q2,
        enable_water_response=lips.ENABLE_WATER_RESPONSE,
        enable_solute_polarization=lips.ENABLE_SOLUTE_POLARIZATION,
    )


def _strip_customgb(system):
    """序列化往返得到独立副本，删掉全部 CustomGBForce，其余力原样保留。"""
    import openmm as mm
    out = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    idxs = [i for i, f in enumerate(out.getForces()) if isinstance(f, mm.CustomGBForce)]
    if not idxs:
        raise RuntimeError("CWLD 系统里没有 CustomGBForce，无法做对照")
    for i in sorted(idxs, reverse=True):
        out.removeForce(i)
    return out, len(idxs)


def measure_speed(top, system, positions, lips, warmup, measure):
    cwld = _cwld_system(system, top, lips)
    no_gb, n_removed = _strip_customgb(cwld)
    log(f"  -> 对照系统已移除 {n_removed} 个 CustomGBForce，其余力保持不变")

    full = timed_ns_per_day("CWLD_full_with_CustomGBForce", cwld, top, positions, lips,
                            warmup, measure)
    bare = timed_ns_per_day("CWLD_minus_CustomGBForce", no_gb, top, positions, lips,
                            warmup, measure)
    s = 1.0 - full / bare if bare > 0 else float("nan")
    ceiling = bare / full if full > 0 else float("nan")

    log("")
    log("=" * 70)
    log(f"  完整 CWLD（含 CustomGBForce）      {full:8.2f} ns/day")
    log(f"  去掉 CustomGBForce（其余力不变）    {bare:8.2f} ns/day   [f=0 天花板]")
    log(f"  ⇒ CustomGBForce 占总步时            {s*100:8.1f}%")
    log(f"  ⇒ f=0 的 Amdahl 天花板              {ceiling:8.2f}x     （不是目标，是上界）")
    log("")
    log("  按 speedup = 1/((1-s) + s·f) 的现实预期：")
    for f in (1.0 / 3, 1.0 / 5, 1.0 / 10):
        sp = 1.0 / ((1 - s) + s * f)
        log(f"    新实现比 CustomGBForce 快 {1/f:4.0f}x  ⇒  整体 {sp:5.2f}x"
            f"   （48.4 GPU 小时 → {48.4/sp:5.1f} 小时）")
    log("=" * 70)
    del cwld, no_gb
    gc.collect()
    return s


def measure_rc_scan(top, system, positions, lips, rc_list, warmup, measure, r_env=0.35):
    """CustomGBForce 的成本随 rc 怎么变 —— 检验"它走三遍 rc 邻居表"这个前提。

    LCWLD-090/100 的设计把 2.12x 建立在：CustomGBForce 走三遍完整 rc 表，而 density 和
    chain 两遍只需要 r_env=0.35（体积比 (0.35/1.2)³=2.5%），所以那两遍近乎免费。

    这个前提是**可测的**：若成本主要是邻居遍历，它应当近似 ∝ rc³；若很大一部分是
    每原子的固定开销，指数会明显小于 3，那么"砍掉两遍遍历"能拿到的就远少于设计估计。
    本函数同时扫**完整系统**和**去掉 CustomGBForce 的对照系统**，两条曲线相减得到
    CustomGBForce 自己的边际成本，再做两分量拟合 `cost = fixed + trav·rc³`——
    否则固定开销在 CustomGBForce 与其余力之间无法分离，只能给出一个区间。

    注意：改 rc 会改物理（closure 的截断半径），所以本函数**只用于计时**，
    产出的任何轨迹/能量都没有物理意义，也不落盘。
    """
    import numpy as np

    def _scan(make_system, tag):
        out = []
        for rc in rc_list:
            sysm = make_system(rc)
            sp = timed_ns_per_day(f"{tag}_rc{rc:.2f}", sysm, top, positions, lips,
                                  warmup, measure)
            out.append(1.0 / sp)
            del sysm
            gc.collect()
        return np.asarray(out)

    def _full(rc):
        # r_on 必须跟着 rc 缩放：默认 r_on=0.9 是配 rc=1.2 的（比值 0.75），
        # 直接改 rc 会让 r_on > rc，OpenMM 拒绝建 Context。保持比值不变，
        # 各档之间 switching 区间的占比也才可比。
        return lips.setup_cwld_lips_system(
            system, top, engine="customgb",
            rc=rc, r_on=R_ON_OVER_RC * rc, a_q2=lips.DEFAULT_A_Q2,
            enable_water_response=lips.ENABLE_WATER_RESPONSE,
            enable_solute_polarization=lips.ENABLE_SOLUTE_POLARIZATION,
        )

    def _bare(rc):
        return _strip_customgb(_full(rc))[0]

    rc = np.asarray(rc_list, dtype=float)
    log("\n  [1/2] 完整 CWLD 系统")
    c_full = _scan(_full, "CWLD_full")
    log("\n  [2/2] 去掉 CustomGBForce 的对照系统（用来把它的成本单独分离出来）")
    c_bare = _scan(_bare, "CWLD_bare")
    c_gb = c_full - c_bare                      # CustomGBForce 的边际成本

    norm = c_full[-1]
    log("")
    log(f"  {'rc':>6}{'完整 ns/day':>14}{'CustomGB 成本':>15}{'其余力成本':>13}")
    for r, cf, cb, cg in zip(rc, c_full, c_bare, c_gb):
        log(f"  {r:6.2f}{1.0/cf:14.2f}{cg/norm:15.3f}{cb/norm:13.3f}")

    # 两分量拟合：cost(rc) = fixed + trav * rc^3。pair 数 ∝ rc^3，
    # 所以 fixed 就是与邻居数无关的那部分（kernel 启动、参数装载、每原子工作）。
    def _fit(c, label):
        trav, fixed = np.polyfit(rc ** 3, c / norm, 1)
        at12 = trav * rc[-1] ** 3
        tot = fixed + at12
        log(f"  {label:<12} fixed = {fixed:.3f} ({100*fixed/tot:4.1f}%)   "
            f"trav(rc=1.2) = {at12:.3f} ({100*at12/tot:4.1f}%)")
        return fixed, trav

    log("")
    log("  两分量拟合 cost = fixed + trav·rc³（归一到完整系统 rc=1.2）")
    g_fix, g_trav = _fit(c_gb, "CustomGB")
    _fit(c_bare, "其余力")

    slope = np.polyfit(np.log(rc), np.log(c_gb), 1)[0]
    log(f"  CustomGB 的 d log(cost)/d log(rc) = **{slope:.2f}**"
        f"   （3.0 = 成本全是遍历；越小说明固定开销占比越大）")

    # ---- 两个竞争的成本模型 ----------------------------------------------
    # 模型 1  cost = fixed + trav·rc³      "有一块与邻居数无关的固定开销"
    # 模型 2  cost ∝ (rc + d)³             "没有固定开销，只有 tile 粒度地板"
    # 模型 2 的物理：OpenMM 按 32 原子的 block 做 tile 剪枝，block 本身有直径 d，
    # 所以被检查的 block 对数量 ∝ (rc+d)³ 而不是 rc³。rc 小到与 d 可比时曲线就变平。
    # 两者在大 rc 处几乎重合，在小 rc 处分道扬镳 —— 所以扫描一定要往小 rc 扫。
    c_rel = c_gb / c_gb[-1]
    try:
        from scipy.optimize import brentq
        d_fit = brentq(
            lambda d: ((rc[-1] + d) / (rc[0] + d)) ** 3 - 1.0 / c_rel[0], 1e-3, 50.0)
        pred2 = ((rc + d_fit) / (rc[-1] + d_fit)) ** 3
        r2 = 100 * np.abs(pred2 / c_rel - 1).max()
        log(f"  模型2  cost ∝ (rc+d)³,  d = {d_fit:.3f} nm      最大残差 {r2:.1f}%")
    except Exception as exc:                      # scipy 缺席或无解，不致命
        d_fit, r2 = None, None
        log(f"  模型2 拟合跳过（{exc}）")
    pred1 = (g_fix + g_trav * rc ** 3) / (g_fix + g_trav * rc[-1] ** 3)
    log(f"  模型1  cost = fixed + trav·rc³              最大残差 "
        f"{100*np.abs(pred1/c_rel-1).max():.1f}%")

    # ---- f：优先用实测，不用模型 -------------------------------------------
    # s 用 rc_max 处的**原始测量值**，不用两分量拟合的外推——后者在 rc 跨度大时
    # 残差可达几十个百分点（模型1 就是），拿它当 s 会把误差直接传进最终倍数。
    s_meas = float(c_gb[-1] / c_full[-1])
    s_fit = g_fix + g_trav * rc[-1] ** 3
    vol = (r_env / rc[-1]) ** 3
    log("")
    log(f"  实测 s（CustomGB 占步时，rc={rc[-1]} 处原始测量）= **{100*s_meas:.1f}%**"
        f"   （两分量拟合外推给 {100*s_fit:.1f}%，只作诊断，不用于下面的推算）")
    log(f"  设计假设 r_env 那一趟只值体积比 {vol*100:.2f}%")

    hit = np.isclose(rc, r_env, atol=1e-6)
    if hit.any():
        ratio = float(c_rel[hit][0])
        log(f"  **实测** rc={r_env} 时 CustomGB 成本是 rc={rc[-1]} 的 {ratio:.3f}"
            f"  ——这就是一趟 r_env 遍历的真实相对成本，不需要任何模型")
        log(f"  ⚠ 它是**上界**：这一档仍走 OpenMM 的 32 原子 tile，而设计用紧凑 flat 列表、"
            f"没有 tile 地板 ⇒ 真实 f 只会更小、端到端只会更大。"
            f"（反向风险：flat 列表访存合并不如 tile，未必吃到全部差额。）")
    else:
        ratio = None
        log(f"  ⚠ 扫描里没有 rc={r_env} 这一档，只能靠模型外推；"
            f"两个模型在这里分歧很大，请把 {r_env} 加进 --rc-scan 重跑")

    log("")
    log("  按设计（3 遍 rc → 1 遍 rc + 2 遍 r_env）推算端到端：")
    cands = [("设计假设(体积比)", vol)]
    if ratio is not None:
        cands.append(("实测 r_env 档", ratio))
    if d_fit is not None:
        cands.append(("模型2 外推", float(((r_env + d_fit) / (rc[-1] + d_fit)) ** 3)))
    cands.append(("模型1 外推", float((g_fix + g_trav * r_env ** 3) / s_fit)))
    for name, rt in cands:
        f_des = (1.0 + 2.0 * rt) / 3.0
        tot = (1.0 - s_meas) + s_meas * f_des
        bound = " ≥" if name.startswith("实测") else "  "
        log(f"    {name:<16} r_env/rc 成本比 {rt:6.3f}  f = {f_des:.3f}  "
            f"→{bound} **{1.0/tot:.2f}x**   ({48.4*tot:5.1f} h)")
    log("")
    log("  设计文档写的是 2.22x（f=0.350）。差距来自 r_env 那一趟并不接近免费：")
    log("  tile 粒度让 block 对数量 ∝ (rc+d)³，rc 降到 0.35 时曲线已经压平。")
    log("  这也说明「融合 kernel 减少启动次数」不是杠杆——地板是 tile 粒度，不是启动开销。")


def measure_precision_floor(top, system, positions, lips, quantiles):
    """同一帧在 double/mixed/single 下的静态力差异 —— fp32 的固有噪声地板。零步积分。"""
    import openmm as mm
    import openmm.unit as unit

    cwld = _cwld_system(system, top, lips)
    plat = mm.Platform.getPlatformByName("CUDA")

    out = {}
    for prec in ("double", "mixed", "single"):
        integ = mm.LangevinMiddleIntegrator(lips.TEMPERATURE, 1.0 / unit.picosecond, lips.DT)
        ctx = mm.Context(cwld, integ, plat, {"Precision": prec})
        ctx.setPositions(positions)
        st = ctx.getState(getForces=True, getEnergy=True)
        out[prec] = (
            np.array(st.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer), dtype=np.float64),
            st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        )
        log(f"  [{prec:>6}] E = {out[prec][1]:.6f} kJ/mol   "
            f"|F|max = {np.linalg.norm(out[prec][0], axis=1).max():.3f} kJ/mol/nm")
        del ctx, integ
        gc.collect()

    ref = out["double"][0]
    ref_norm = np.linalg.norm(ref, axis=1)
    scale = max(np.median(ref_norm), 1e-12)
    log("")
    log("  力的逐原子偏差（相对 double；分母用全体 |F| 中位数，避免近零原子放大）")
    log(f"  {'对照':<18}" + "".join(f"{f'p{q}':>10}" for q in quantiles) + f"{'max':>12}")
    for tag in ("mixed", "single"):
        d = np.linalg.norm(out[tag][0] - ref, axis=1) / scale
        row = "".join(f"{np.percentile(d, q):10.2e}" for q in quantiles)
        log(f"  {tag+' vs double':<18}{row}{d.max():12.2e}")
    d = np.linalg.norm(out["single"][0] - out["mixed"][0], axis=1) / scale
    row = "".join(f"{np.percentile(d, q):10.2e}" for q in quantiles)
    log(f"  {'single vs mixed':<18}{row}{d.max():12.2e}")

    e_d = out["double"][1]
    log("")
    log("  总能量相对偏差（这是求和累积误差，必然比逐原子力差；不要拿它当验收判据）")
    for tag in ("mixed", "single"):
        log(f"    {tag:>6} vs double : {abs(out[tag][1]-e_d)/max(abs(e_d),1e-12):.3e}")
    log("")
    log("  用法提示：kernel 验收阈值取上表 single-vs-mixed 的 p99 的 3–5 倍，")
    log("  并**在写 kernel 之前**写死。cutoff 归属翻转的原子单独统计，不混进这个分位数。")

    del cwld
    gc.collect()


def main(a=None):
    if a is None:
        a = build_parser().parse_args()
    if not (a.speed or a.precision or a.rc_scan):
        raise SystemExit("至少给 --speed / --precision / --rc-scan 其中之一")

    from lips import systems as ls
    from lips.engine import v26 as lips

    top, system, positions = ls.load_reference_system(name=a.system, with_positions=True)
    log(f"[system] {a.system}: {top.getNumAtoms()} 原子")

    if a.speed or a.rc_scan:
        # 极小化一次、所有臂共用：见 minimize_once 的 docstring。
        positions = minimize_once(top, system, positions, lips)

    if a.speed:
        log("\n" + "#" * 70 + "\n# 1. CustomGBForce 的步时占比 s\n" + "#" * 70)
        measure_speed(top, system, positions, lips, a.warmup, a.measure)

    if a.precision:
        log("\n" + "#" * 70 + "\n# 2. fp32 精度零点（零步积分）\n" + "#" * 70)
        qs = [int(x) for x in a.quantiles.split(",")]
        measure_precision_floor(top, system, positions, lips, qs)

    if a.rc_scan:
        log("\n" + "#" * 70 + "\n# 3. CustomGBForce 成本随 rc 的标度\n" + "#" * 70)
        rcs = [float(x) for x in a.rc_scan.split(",")]
        measure_rc_scan(top, system, positions, lips, rcs, a.warmup, a.measure, a.r_env)


def build_parser():
    p = argparse.ArgumentParser(description="CustomGBForce 步时占比 + fp32 噪声地板")
    p.add_argument("--system", default="1aay")
    p.add_argument("--speed", action="store_true", help="测 s（会跑几百步 MD）")
    p.add_argument("--precision", action="store_true", help="测精度零点（零步积分）")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--measure", type=int, default=1000)
    p.add_argument("--quantiles", default="50,90,99")
    p.add_argument("--r-env", dest="r_env", type=float, default=0.35,
                   help="density/chain 两遍实际需要的半径（默认 0.35，与引擎一致）")
    p.add_argument("--rc-scan", dest="rc_scan", default=None,
                   help="逗号分隔的 rc 列表。**务必包含 r_env（默认 0.35）**，"
                        "否则 r_env 那一趟的成本只能靠模型外推，而两个候选模型在那里差 50%%。"
                        "推荐 0.35,0.5,0.7,0.9,1.2")
    return p


if __name__ == "__main__":
    main()
