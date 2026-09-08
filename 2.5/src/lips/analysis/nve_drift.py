#!/usr/bin/env python
"""Stage B · NVE 能量漂移 —— 力审计量不到的那个通道。

背景（STAGE1_CLOSURE_KILLSWITCH_REPORT.md §11）：静态力审计给出
zmm1 < zmm2 < pswfz2 < zmm3 的力精度排序，pswfz2 在每个通道都比 zmm2 差 3-15%。
但**力的 RMS 精度不度量 cutoff 处的光滑性**，而后者正是：
  (a) Sakuraba 偏好高 ell 的最可能理由（与力精度排序冲突需要一个解释）
  (b) pswfz2 唯一还可能赢的地方（带外能量低 13.7 倍 ⇒ 粒子穿越 cutoff 时力更平滑）
NVE 总能漂移直接量这件事。

设计要点（都是为了不出假信号）：
  * **固定电荷**，不挂 CustomGBForce —— closure 是 pair kernel，隔离它即可，且快得多
  * LJ 保留但电荷从 NonbondedForce 里剥掉，静电全部走被测 kernel；
    例外对（含 1-4）在**所有** run 里一律全排除，包括 PME 参照 —— 口径与 §11 力审计一致
  * 每个 kernel **先用它自己的 Hamiltonian 做 NVT 平衡**，再切 NVE；
    否则头几 ps 是换 Hamiltonian 的弛豫瞬变，会被误当成漂移
  * **移除 CMMotionRemover** —— 它会抽走质心动能，是非保守的，直接污染漂移测量
  * 漂移用总能对时间做线性拟合，**丢掉前 20% 的瞬变**
  * PME 作为参照给出漂移下限（积分器/约束本身的贡献）

用法：
    python closure_nve_drift.py --selftest                    # 合成体系，秒级
    L_IPS_DATA_DIR=/home/ruigengji/L-IPS/2.4 python -u closure_nve_drift.py
"""
from __future__ import annotations

import argparse, os, time
import numpy as np
import pandas as pd

from lips import paths
from lips import systems as ls
from lips.paths import DATA_DIR as _DATA_DIR
DATA_DIR = str(_DATA_DIR)

ONE_4PI_EPS0 = 138.93545764498226


def log(m): print(m, flush=True)


def strip_and_wire(system, kernel_expr, rc, keep_restraints=False):
    """把电荷从 NonbondedForce 剥掉，静电改由 kernel 承担；kernel_expr=None 表示保留 PME。

    返回 (system, n_excl)。原地修改传入的 system 副本。
    """
    import openmm as mm

    # 1) 移除 CMMotionRemover（非保守，会污染漂移）
    for i in reversed(range(system.getNumForces())):
        if isinstance(system.getForce(i), mm.CMMotionRemover):
            system.removeForce(i)
    # 2) 默认移除位置限制弹簧。build_1ckk_system 加的 CustomExternalForce 用的是
    #    **绝对坐标参考点**（建体系时的初始结构），而我们从轨迹第 N 帧出发；
    #    轨迹帧经周期性包裹后与参考点可能差一个盒矢量，弹簧会以巨力拉扯 → NaN。
    #    这个力在所有 kernel 里都一样，与 closure 对照无关，去掉最安全。
    n_removed = 0
    if not keep_restraints:
        for i in reversed(range(system.getNumForces())):
            if isinstance(system.getForce(i), mm.CustomExternalForce):
                system.removeForce(i); n_removed += 1

    # 3) 移除任何 barostat / thermostat
    for i in reversed(range(system.getNumForces())):
        f = system.getForce(i)
        if isinstance(f, (mm.MonteCarloBarostat, mm.AndersenThermostat)):
            system.removeForce(i)

    nb = [system.getForce(i) for i in range(system.getNumForces())
          if isinstance(system.getForce(i), mm.NonbondedForce)][0]
    from openmm.unit import elementary_charge
    charges = np.array([
        nb.getParticleParameters(i)[0].value_in_unit(elementary_charge)
        for i in range(nb.getNumParticles())])
    excl = [tuple(nb.getExceptionParameters(i)[:2])
            for i in range(nb.getNumExceptions())]

    # 例外对静电一律全排除（所有 run 口径一致，含 PME 参照）
    for i in range(nb.getNumExceptions()):
        p1, p2, qq, sig, eps = nb.getExceptionParameters(i)
        nb.setExceptionParameters(i, p1, p2, 0.0, sig, eps)

    # build_1ckk_system 的基础体系用 nonbondedCutoff=1.0 nm，而生产的 CWLD 路径
    # 会把它重设成 rc(=1.2)。这里必须照做：否则 (a) 与 CustomNonbondedForce 同组
    # cutoff 不一致会直接抛 "All Forces in a single force group must use the same
    # cutoff distance"，(b) LJ 口径在 PME 参照与各 kernel 之间也会不一致。
    nb.setCutoffDistance(rc)

    if kernel_expr is not None:
        nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)   # 电荷已清零，只剩 LJ
        for i in range(nb.getNumParticles()):
            q, sig, eps = nb.getParticleParameters(i)
            nb.setParticleParameters(i, 0.0, sig, eps)      # 只留 LJ
        cnb = mm.CustomNonbondedForce(f"{ONE_4PI_EPS0}*q1*q2*{kernel_expr}")
        cnb.addPerParticleParameter("q")
        cnb.addGlobalParameter("rc", rc)
        cnb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        cnb.setCutoffDistance(rc)
        cnb.setUseLongRangeCorrection(False)
        for q in charges:
            cnb.addParticle([float(q)])
        for (p1, p2) in excl:
            cnb.addExclusion(p1, p2)
        system.addForce(cnb)
    return system, len(excl), n_removed


def drift_of(system, positions, box, dt_ps, equil_ps, prod_ps, report_ps,
             seed, platform, precision, temp=300.0, minimize=False,
             trace_out=None, tag=""):
    import openmm as mm
    from openmm import unit

    props = {"Precision": precision} if platform.getName() == "CUDA" else {}
    # --- NVT 平衡：用被测 Hamiltonian 自己平衡，去掉换 Hamiltonian 的瞬变 ---
    integ = mm.LangevinMiddleIntegrator(temp * unit.kelvin, 1.0 / unit.picosecond,
                                        dt_ps * unit.picoseconds)
    integ.setRandomNumberSeed(seed)
    ctx = mm.Context(system, integ, platform, props) if props else mm.Context(system, integ, platform)
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(positions * unit.nanometer)
    ctx.applyConstraints(1e-10)      # 轨迹坐标是单精度存的，约束会有残余违背
    # ⚠ 只在需要时最小化。对**已经平衡过的 MD 帧**做最小化是错的：它把体系推离平衡，
    # 之后的弛豫会让 NVE 的积分误差随时间变化，线性拟合出来的"漂移"没有意义。
    # 自检的合成体系需要它（随机/格点摆位有尖峰），真实轨迹帧不需要。
    if minimize:
        mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 2000)
    ctx.setVelocitiesToTemperature(temp * unit.kelvin, seed)
    ctx.applyVelocityConstraints(1e-10)

    # fail-fast：起始态若已经病态，立刻报出来并指名最大受力原子，
    # 而不是跑满 equil_ps 再抛一个没有上下文的 "Particle coordinate is NaN"。
    st0 = ctx.getState(getEnergy=True, getForces=True)
    pe0 = st0.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f0 = st0.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)
    fmag = np.linalg.norm(f0, axis=1)
    worst = int(np.argmax(fmag))
    if (not np.isfinite(pe0)) or (not np.all(np.isfinite(f0))) or fmag[worst] > 1e6:
        raise RuntimeError(
            f"起始态病态：PE={pe0:.4g} kJ/mol，最大受力 {fmag[worst]:.4g} kJ/mol/nm "
            f"@ atom {worst}。常见原因：分子被盒边界劈开（需 image_molecules）、"
            f"位置限制弹簧参考点与轨迹帧差一个盒矢量（用 --keep-restraints 时）、"
            f"或起始帧本身有重叠。")

    integ.step(int(round(equil_ps / dt_ps)))
    st = ctx.getState(getPositions=True, getVelocities=True)
    pos_eq, vel_eq = st.getPositions(), st.getVelocities()
    del ctx, integ

    # --- NVE 生产：Verlet，无恒温器 ---
    integ = mm.VerletIntegrator(dt_ps * unit.picoseconds)
    ctx = mm.Context(system, integ, platform, props) if props else mm.Context(system, integ, platform)
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(pos_eq)
    ctx.setVelocities(vel_eq)

    nsteps = int(round(prod_ps / dt_ps))
    every = max(1, int(round(report_ps / dt_ps)))
    t, E, T = [], [], []
    kB = 0.008314462618
    ndof = (system.getNumParticles() * 3 - system.getNumConstraints())
    for s in range(0, nsteps, every):
        integ.step(every)
        st = ctx.getState(getEnergy=True)
        ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        pe = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        t.append((s + every) * dt_ps); E.append(ke + pe)
        T.append(2 * ke / (ndof * kB))
    t = np.array(t); E = np.array(E)
    m = t >= 0.2 * t[-1]                       # 丢掉前 20% 瞬变
    tm, Em = t[m], E[m]
    coef, cov = np.polyfit(tm, Em, 1, cov=True)
    slope = coef[0]                            # kJ/mol/ps
    slope_se = float(np.sqrt(cov[0, 0]))       # 斜率标准误

    # 线性度检验：E(t) 若是弯的，直线拟合照样给出很小的 slope_se（看着很"精确"），
    # 但那个斜率没有物理意义。
    half = len(tm) // 2
    s1 = float(np.polyfit(tm[:half], Em[:half], 1)[0])
    s2 = float(np.polyfit(tm[half:], Em[half:], 1)[0])
    # 归一化的前后半差异（**保留但不再作判据**，见下）
    nonlin_half = abs(s2 - s1) / max(abs(slope), 1e-30)

    # ⚠ 2026-09-07：前后半判据被实测证明**会被对称的非单调性骗过**。
    # mixed 精度那两批（dt=2fs / 1fs）实例：
    #   zmm1 dt2fs  前后半 0.02（"通过"）  四分段 +1.2e-1 +2.3e-1 +2.1e-1 +2.4e-2 → 离散 2.45
    #   zmm2 dt2fs  前后半 0.14（"通过"）  四分段 +1.2e-1 +2.0e-1 +4.1e-1 +1.3e-2 → 离散 3.85
    #   pme  dt1fs  前后半 0.01（"通过"）  四分段里 Q3 直接**变号** → 离散 2.65
    # 一个中间高、两头低的鼓包，前半均值≈后半均值 ⇒ |s2-s1| 天然接近 0。
    # 改用**分块斜率离散度**：把拟合窗切 nblk 段，看各段斜率的极差 / |整体斜率|。
    # 上面三例分别给 2.45 / 3.85 / 2.65，全部会被 >0.5 的阈值拦下。
    nblk = 4
    blk = [float(np.polyfit(tm[i * len(tm) // nblk:(i + 1) * len(tm) // nblk],
                            Em[i * len(tm) // nblk:(i + 1) * len(tm) // nblk], 1)[0])
           for i in range(nblk)]
    nonlin = (max(blk) - min(blk)) / max(abs(slope), 1e-30)

    if trace_out is not None:
        trace_out.append(pd.DataFrame({"tag": tag, "t_ps": t, "E_total": E}))
    return dict(drift_kJ_per_mol_per_ns=slope * 1000.0,
                slope_first_half=s1 * 1000.0, slope_second_half=s2 * 1000.0,
                nonlinearity=nonlin,                 # 分块斜率离散度（现判据）
                nonlinearity_half=nonlin_half,       # 旧的前后半判据，保留可比
                block_slopes_kJ_per_mol_per_ns=";".join(f"{b*1000.0:.4g}" for b in blk),
                drift_se_kJ_per_mol_per_ns=slope_se * 1000.0,
                drift_se_kT_per_ns_per_dof=slope_se * 1000.0 / (kB * temp * ndof),
                drift_per_atom_per_ns=slope * 1000.0 / system.getNumParticles(),
                drift_kT_per_ns_per_dof=slope * 1000.0 / (kB * temp * ndof),
                E0=float(E[0]), Emean=float(E.mean()), Estd=float(E.std()),
                T_start=float(T[0]), T_end=float(T[-1]), ndof=ndof)


def run(args):
    import openmm as mm
    import mdtraj as md
    from lips.engine import closure as cw
    from copy import deepcopy

    # 产物名先定、先查。**dt 必须进名字**：dt 对照是本脚本唯一的判据（真实积分
    # 误差 ∝ dt²），而旧版两次不同 dt 写同一个文件名 —— 第二次会把第一次吃掉，
    # 判据自己被自己毁掉。几小时的 MD，绝不能等跑完才发现要覆盖。
    _nk = len([k for k in args.kernels.split(",") if k.strip()])
    _sig = (f"dt{args.dt * 1000:g}fs_{args.precision}"
            f"_{_nk}k{paths.run_sig(args.kernels)}")
    out_csv = paths.result_new(f"closure_nve_drift_{_sig}.csv", force=args.force)
    traces_csv = paths.result_new(f"closure_nve_traces_{_sig}.csv", force=args.force)

    # 体系是参数，不是常量。原先写死 build_1ckk_system()，而 1CKK 已经不是当前
    # 体系；更要命的是这份加载没有 normalize_ion_resnames()，换到 1AAY 会让锌的
    # dens_source 静默变 0。统一走 lips.systems 这一个入口。
    topology, sys_pme = ls.load_reference_system_from_args(args)

    traj_path = args.traj if os.path.isabs(args.traj) else os.path.join(DATA_DIR, args.traj)
    md_top = md.Topology.from_openmm(topology)
    ls.assert_traj_matches_system(traj_path, md_top.n_atoms)   # 几小时的 MD，先校验
    tr = md.load(traj_path, top=md_top, stride=1)
    # 分子补全 + 重新包裹：约束（SETTLE/HBonds）用的是**原始坐标、不做最小镜像**，
    # 分子若被盒边界劈开，约束会直接爆。block_time_check_v26.py 同样调了这个。
    tr = tr.image_molecules(inplace=False)
    log(f"[traj]  {os.path.basename(traj_path)}: {tr.n_frames} frames，取第 {args.start_frame} 帧起")

    plat = mm.Platform.getPlatformByName(args.platform)
    log(f"[plat]  {args.platform} precision={args.precision}")
    log(f"[proto] dt={args.dt} ps, NVT 平衡 {args.equil_ps} ps -> NVE {args.prod_ps} ps, "
        f"{args.seeds} seeds")

    rows, traces = [], []
    for name in args.kernels.split(","):
        name = name.strip()
        expr = None if name == "pme" else cw.make_window(name).lepton("rc")
        for si in range(args.seeds):
            fi = min(args.start_frame + si * args.frame_stride, tr.n_frames - 1)
            box = [mm.Vec3(*v) for v in tr.unitcell_vectors[fi]]
            sysm, nex, nrm = strip_and_wire(deepcopy(sys_pme), expr, args.rc,
                                            keep_restraints=args.keep_restraints)
            if si == 0 and name == args.kernels.split(",")[0].strip():
                log(f"[sys]   移除位置限制 CustomExternalForce {nrm} 个"
                    f"（--keep-restraints 可保留）；排除对 {nex} 个")
            t0 = time.time()
            r = drift_of(sysm, tr.xyz[fi].astype(np.float64), box, args.dt,
                         args.equil_ps, args.prod_ps, args.report_ps,
                         1234 + si, plat, args.precision,
                         minimize=args.minimize, trace_out=traces,
                         tag=f"{name}_seed{si}")
            r.update(kernel=name, seed=si, frame=fi, sec=round(time.time() - t0, 1))
            rows.append(r)
            flag = "  ⚠非线性" if r["nonlinearity"] > 0.5 else ""
            log(f"  {name:8s} seed{si} 漂移 {r['drift_kT_per_ns_per_dof']:+.3e} "
                f"± {r['drift_se_kT_per_ns_per_dof']:.1e} kT/ns/dof  "
                f"({r['drift_kJ_per_mol_per_ns']:+.1f} kJ/mol/ns)  "
                f"T {r['T_start']:.1f}->{r['T_end']:.1f}K  "
                f"nonlin={r['nonlinearity']:.2f}{flag}  ({r['sec']}s)")

    out = pd.DataFrame(rows)
    fn = out_csv                      # 开头就定好并查过存在性，见 run() 顶部
    out.to_csv(fn, index=False)
    gg = out.groupby("kernel")["drift_kT_per_ns_per_dof"]
    g = gg.agg(["mean", "std", "count"])
    g["sem"] = g["std"] / np.sqrt(g["count"])
    log("\n" + "=" * 74)
    log("NVE 总能漂移（kT/ns/dof，绝对值越小越好）")
    log("=" * 74)
    log(g.to_string(float_format=lambda v: f"{v:+.4e}"))
    if "zmm2" in g.index:
        log("\n对 zmm2 归一（用 |漂移|，<1 更好）。括号内是 seed 间 sem 折算的比值不确定度：")
        base = abs(float(g.loc["zmm2", "mean"]))
        for k in g.index:
            v = abs(float(g.loc[k, "mean"])); e = float(g.loc[k, "sem"])
            log(f"    {k:8s} {v/base:6.3f}  (±{e/base:.3f})")
        log("\n⚠ 判读纪律：若某 kernel 与 zmm2 的差距 < 两者 sem 之和，就是噪声，不要当信号。")
        if "pme" in g.index:
            log(f"  PME 的 |漂移| = {abs(float(g.loc['pme','mean']))/base:.3f}（归一后）。")
            log("  ⚠ PME **不是**漂移地板 —— 它有倒空间网格插值误差；而一个在 rc 处光滑归零的"
                "截断 kernel\n     本身就是严格保守的 Hamiltonian，能量守恒可以优于 PME。"
                "所以 kernel 低于 PME 是正常的，\n     不能拿 PME 当'可疑'判据。"
                "真正的地板要用 dt 标度确定（见下）。")
    if "pme" in g.index:
        log(f"\n（PME 参照 = 积分器/约束本身的漂移下限：{float(g.loc['pme','mean']):+.3e}）")
    # 非线性汇总 + dt 标度提示
    bad = out[out["nonlinearity"] > 0.5]
    log("")
    if len(bad):
        log(f"⚠ {len(bad)}/{len(out)} 条 run 的 E(t) 明显非线性（前后半斜率差 > 整体斜率）。")
        log("  线性拟合的 slope_se 会小得很好看，但那个斜率没有物理意义。**先别用这批数字。**")
        log(f"  非线性最严重的：\n{bad.nlargest(min(3,len(bad)),'nonlinearity')[['kernel','seed','nonlinearity','nonlinearity_half','drift_kT_per_ns_per_dof']].to_string(index=False)}")
        log("  （nonlinearity = 分块斜率离散度；nonlinearity_half = 旧的前后半判据。"
            "后者会被对称鼓包骗过，只作对比用，不要拿它当通过条件。）")
    else:
        log("✓ 所有 run 的 E(t) 线性度可接受（前后半斜率差 < 整体斜率的 50%）。")
    log("\n判定漂移是否真实，必须再看 dt 标度：真实的积分器误差 ∝ dt²（dt 减半 ⇒ 漂移降到 1/4，"
        "\n符号不变）。若 dt 减半后漂移量级不变甚至变号，那测到的就不是漂移，是别的东西。"
        "\n跑 --dt 0.001 复算并对比本表。")

    if traces:
        tf = traces_csv
        pd.concat(traces, ignore_index=True).to_csv(tf, index=False)
        log(f"-> {os.path.basename(tf)}（E(t) 原始轨迹，事后可自己查曲率）")
    log(f"-> {os.path.basename(fn)}")


def selftest(args):
    """合成体系验管线：120 个带电 LJ 粒子，跑得动、能出漂移数就算过。"""
    import openmm as mm
    from lips.engine import closure as cw
    rng = np.random.default_rng(0)
    m, Lb = 5, 3.0                       # 5x5x5 格点，无重叠（随机摆位会重叠->LJ 爆->NaN）
    n = m ** 3
    g = (np.arange(m) + 0.5) * (Lb / m)
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    pos += rng.normal(0, 0.02, pos.shape)
    q = rng.normal(0, 0.4, n); q -= q.mean()
    box = [mm.Vec3(Lb, 0, 0), mm.Vec3(0, Lb, 0), mm.Vec3(0, 0, Lb)]
    # 必须能跑 CUDA：「同一 force group 内 cutoff 必须一致」这条约束
    # **只有 CUDA 平台强制**，Reference 上不报错 —— 自检若只跑 Reference
    # 就抓不到这类 bug（2026-09-01 实际漏掉过一次）。
    plat = mm.Platform.getPlatformByName(args.platform)
    log(f"[selftest] {n} 粒子（{m}³ 格点）/ {Lb} nm 盒子 / {plat.getName()} / dt={args.dt} ps")
    log(f"[selftest] NonbondedForce cutoff 故意设为 1.0 nm，kernel rc={args.rc} —— 复现崩溃条件")
    for name in args.kernels.split(","):
        name = name.strip()
        s = mm.System()
        for _ in range(n):
            s.addParticle(16.0)
        s.setDefaultPeriodicBoxVectors(*box)
        nb = mm.NonbondedForce()
        nb.setNonbondedMethod(mm.NonbondedForce.PME)
        # 故意设成 1.0 nm 而非 rc —— 复现 build_1ckk_system 的真实条件
        # (createSystem 用 nonbondedCutoff=1.0，生产 CWLD 路径才重设成 rc=1.2)。
        # 这样自检才会覆盖 "All Forces in a single force group must use the same
        # cutoff distance" 那条崩溃路径。
        nb.setCutoffDistance(1.0)
        for qq in q:
            nb.addParticle(float(qq), 0.3, 0.5)
        s.addForce(nb)
        expr = None if name == "pme" else cw.make_window(name).lepton("rc")
        s, _, _ = strip_and_wire(s, expr, args.rc)
        r = drift_of(s, pos, box, args.dt, args.equil_ps, args.prod_ps,
                     args.report_ps, 7, plat,
                     "double" if plat.getName() != "CUDA" else args.precision,
                     minimize=True)      # 合成体系必须最小化，否则尖峰直接炸出 NaN
        log(f"  {name:8s} 漂移 {r['drift_kT_per_ns_per_dof']:+.3e} kT/ns/dof   "
            f"T {r['T_start']:.0f}->{r['T_end']:.0f}K")
        assert np.isfinite(r["drift_kT_per_ns_per_dof"]), (
            f"{name} 漂移非有限——体系炸了，检查起始构型/最小化/dt")
    log("[selftest] PASS —— 剥电荷/挂 kernel/NVT->NVE/拟合 全链路通")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ls.add_system_arguments(ap)
    # 同 force_audit：默认不能是 32818 的 PME 轨迹，见 §7.5。
    ap.add_argument("--traj", default="ZN_1aay_cwld_ligm0p15metal_zmm_seed0_traj.dcd",
                    help="取初始构型的轨迹（相对 DATA_DIR 或绝对路径）")
    ap.add_argument("--start-frame", type=int, default=400)
    ap.add_argument("--frame-stride", type=int, default=30)
    ap.add_argument("--kernels", default="pme,zmm1,zmm2,zmm3,pswfz2")
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--equil-ps", type=float, default=200.0)  # 50ps 不够，残余弛豫会放大积分误差
    ap.add_argument("--prod-ps", type=float, default=1000.0)   # 100ps 灵敏度不够
    ap.add_argument("--report-ps", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rc", type=float, default=1.2)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--keep-restraints", action="store_true",
                    help="保留 build_1ckk_system 的位置限制弹簧。默认移除——"
                         "它用绝对坐标参考点，与包裹过的轨迹帧不匹配时会以巨力拉扯直接 NaN")
    ap.add_argument("--minimize", action="store_true",
                    help="NVT 前做能量最小化。**从已平衡的 MD 帧出发时不要开**——"
                         "它会把体系推离平衡，注入弛豫瞬变，让漂移随时间变化")
    ap.add_argument("--force", action="store_true",
                    help="覆盖已存在的产物 csv（默认拒跑）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest(a) if a.selftest else run(a)


if __name__ == "__main__":
    main()
