#!/usr/bin/env python
"""Zn 判据的单个作业 —— 一次调用只跑一个 (体系 × 方法)，方便拆成多个作业排队。

判据（见 zinc_finger.py / zn_water.py 的文档）：
    体系                  正确 CN        固定 +2      CWLD      AMOEBA
    Zn²⁺ 纯水            6（八面体）     ~6 ✓         ?         6 ✓
    Zn²⁺ Cys₂His₂(1AAY)  4（四面体）     ~6 ✗         ?         4 ✓
**同一个固定电荷不可能同时满足两边** —— CN 的**反差**才是判据，不是单个 CN。

方法：
  pme      标准 Amber 固定电荷，Zn=+2（**故意保留过配位失败**）
  pme_q    电荷转移诊断：Zn=--zn-charge，差额均分给 4 个配体（净电荷不变）。
           用来标定"修好配位需要多大的 Δq"，从而判断 CWLD 的 clamp 够不够。
  cwld     CWLD 隐式极化（复用 test_lips_vs_pmeV2.6.setup_cwld_lips_system）
  amoeba   AMOEBA 2018 极化基线

时间步：固定电荷/CWLD 用 2 fs（有 HBonds 约束）；**AMOEBA 无约束、必须 ≤1 fs**。

用法（一个作业一条命令，各自 ns 自己定）：
  python run_zn_job.py --system water --method pme    --ns 5 --seeds 3
  python run_zn_job.py --system 1aay  --method amoeba --ns 1 --seeds 1
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

from lips import systems as ls
from lips.paths import DATA_DIR as _DATA_DIR, RESULTS_DIR
# 产物写 RESULTS_DIR（默认 <repo>/data），不是仓库根。2026-09-05 之前它等于
# 仓库根，130 个 dcd/csv/json 就是这么堆出来的。改这里而不是改调用点：
# 落盘位置只能有一个来源。
SCRIPT_DIR = str(RESULTS_DIR)
ZN_CUT_A = 2.8


def log(m): print(m, flush=True)


def _load_main():
    from lips.engine import v26
    return v26


def make_system(system_kind, method, q_zn, box_nm, verbose=True, engine="plugin"):
    """返回 (topology, system, positions, dt_ps, tag)。"""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    if system_kind == "1aay":
        # 体系加载走 lips.systems 这一个入口：读预建体系、按 meta.json 还原
        # HID/HIE、以及（下面 cwld 分支要用的）离子残基名规范化，都在那里实现一次。
        # 这三步以前在 force_audit / nve_drift / 这里各写一份，各自演化，于是
        # normalize_ion_resnames() 只在其中一份里被调到——漏掉它会让 Zn 的
        # dens_source 静默变 0，CWLD 机制整个关掉且不报错，实测废掉过一整批轨迹。
        leg = "amoeba" if method == "amoeba" else "amber"
        top, system, pos = ls.load_reference_system(
            name=f"1aay-amoeba" if leg == "amoeba" else "1aay",
            normalize_ions=True,
            verbose=verbose, with_positions=True)
    elif system_kind == "water":
        from lips.build.zn_water import build_zn_water_system
        top, system, pos = build_zn_water_system(
            box_nm=box_nm, q_zn=2.0,
            ff_kind="amoeba" if method == "amoeba" else "amber", verbose=verbose)
        # ⚠ 水盒子是每次现建的（addSolvent 随机摆水），**必须把拓扑存下来**，
        #   否则事后拿 dcd 做逐帧分析时没有匹配的拓扑，轨迹等于废的。
        #   2026-09-02 首轮就是这么丢掉的：ZN_water_*.dcd 全部无法逐帧复分析。
        wdir = os.path.join(SCRIPT_DIR, "zn_water_top")
        os.makedirs(wdir, exist_ok=True)
        wp = os.path.join(wdir, f"water_{'amoeba' if method=='amoeba' else 'amber'}"
                                f"_box{box_nm:g}.pdb")
        if not os.path.exists(wp):
            app.PDBFile.writeFile(top, pos, open(wp, "w"), keepIds=True)
            if verbose:
                log(f"  水盒子拓扑已存 -> {os.path.relpath(wp, SCRIPT_DIR)}")
    else:
        raise ValueError(system_kind)

    # --- 方法特有的改造 ---
    if method == "pme_q":
        from lips.build.zinc_finger import apply_charge_transfer
        if system_kind != "1aay":
            raise SystemExit("pme_q（电荷转移）只对 1AAY 有定义：纯水里没有蛋白配体"
                             "接收电荷，非整数净电荷无法用整数离子中和。")
        apply_charge_transfer(system, top, pos, q_zn, verbose=verbose)
    elif method == "cwld":
        lips = _load_main()
        # ⚠⚠ 必须先规范化离子残基名。`build_phase_cwld_metadata` 判的是
        #     `residue_name in ION_RESNAME_ALIASES.values()`（values 是 Zn2+/Ca2+/...），
        #     而 PDB 里是 'ZN'（key，不是 value）。不规范化 -> Zn 的 dens_source=0，
        #     **金属离子直接从密度源里消失，而且完全静默**。
        #     build_1ckk_system 调了 3 次，我第一版 1AAY 一次没调，整批 CWLD 因此作废。
        #     （记忆 project_localcwld_plugin 记过同一个坑，那次是合成测试踩的。）
        # 规范化已在 lips.systems 里做过；这里只做那条硬断言，实现也在共享模块。
        ls.assert_metals_are_density_sources(top, system, verbose=verbose)
        if verbose:
            log(f"  注入 CWLD（closure={lips.closure_label()}）")
        system = lips.setup_cwld_lips_system(system, top, engine=engine)

    dt_ps = 0.001 if method == "amoeba" else 0.002
    # ⚠ tag 必须把**所有会改变物理的开关**都带上，否则两个不同配置会写同一批
    #   文件名。2026-09-02 实测后果：C1(dpolar 开) 与 C2(dpolar 关) 并行跑，
    #   两个进程各自持有同一路径的句柄、各自维护写偏移，帧被**交错写在一起**，
    #   dcd 头里帧数还是 2500 而实际 3379/4258 —— 读它的程序不会报错，只会
    #   静默给出垃圾结论。约 12 小时机时里只留下 1 条干净轨迹。
    tag = f"{system_kind}_{method}"
    if method == "pme_q":
        tag += f"_q{q_zn:g}"
    if method == "cwld":
        _ld = os.environ.get("L_IPS_LIGAND_DPOLAR", "").strip()
        if _ld:
            tag += (f"_lig{_ld.replace('-','m').replace('.','p')}"
                    f"{os.environ.get('L_IPS_LIGAND_SCOPE','metal').strip().lower()}")
        else:
            tag += "_ligoff"
        # closure 必须**带上阶数**进文件名（2026-09-07 加）。
        # 起因：原来这里只写族名 `zmm`，于是 ell=1 与 ell=2 的轨迹**同名** ——
        # 正是 2026-09-05 那次 984 MB 被覆盖成 61 MB 的形状，而且更隐蔽：
        # 两档 closure 的力差 7%（静态力审计），混进同一个统计不会报错。
        #
        # 口径（用户 2026-09-07 决定「只用于新跑」）：
        #   * 新跑一律写全 -> `_zmm1_` / `_zmm2_` / `_zmm3_`；
        #   * **既有产物保持 `_zmm_` 不重命名** —— 重命名会打断已发布报告里的
        #     几十处文件名引用。历史上的 `_zmm_` 一律指 **ell=2**（当时的冻结默认）。
        #   * 非 ZMM 的窗名自带参数（`pswfz2` / `pcf1_c0.25`），只需把 `.` 换成 `p`
        #     以免文件名里出现多余的点，转换约定与上面 dpolar 那行一致。
        _cl = os.environ.get("L_IPS_CLOSURE", "zmm").strip().lower()
        if _cl == "zmm":
            from lips.engine.v26 import ZMM_CLOSURE_ELL as _ell   # 单一来源，已校验 1/2/3
            tag += f"_zmm{_ell}"
        else:
            tag += f"_{_cl.replace('.', 'p')}"
        # 引擎必须进文件名。2026-09-05 实测后果：插件成为默认之后，一次 0.5 ns
        # 的验证跑和 2026-09-03 那批 CustomGBForce 的 5 ns 轨迹**同名**，
        # `--force` 一加就把 984 MB 覆盖成 61 MB，无备份、不可恢复。
        # 同名还有第二层危害：两个引擎不是逐比特相同（逐原子力 p99 3.3e-3），
        # 混进同一个统计量正是 2.4 那次 closure 不一致的重演。
        #
        # 只给 plugin 加后缀、customgb 保持原名：既有矩阵全是 CustomGBForce 跑的，
        # 补跑它们必须落回原来的文件名；而新引擎因此**不可能**覆盖到旧数据。
        if engine == "plugin":
            tag += "_plugin"
    return top, system, pos, dt_ps, tag


def zn_coordination(top, pos_nm):
    """返回每个 Zn 的 (CN, 按元素/残基分类, 角度列表)。"""
    import itertools
    atoms = list(top.atoms())
    xyz = np.asarray(pos_nm) * 10.0
    out = []
    for a in atoms:
        if a.residue.name not in ("ZN", "Zn", "Zn2+"):
            continue
        d = np.linalg.norm(xyz - xyz[a.index], axis=1)
        lig = [j for j in np.argsort(d)[1:16]
               if d[j] < ZN_CUT_A and atoms[j].element is not None
               and atoms[j].element.symbol in ("N", "O", "S")]
        kinds = {}
        for j in lig:
            rn = atoms[j].residue.name
            k = ("water" if rn in ("HOH", "WAT") else
                 "cys_S" if atoms[j].element.symbol == "S" else
                 "his_N" if rn.startswith("HI") else "other")
            kinds[k] = kinds.get(k, 0) + 1
        v = [(xyz[j] - xyz[a.index]) / d[j] for j in lig]
        ang = [float(np.degrees(np.arccos(np.clip(np.dot(p, q), -1, 1))))
               for p, q in itertools.combinations(v, 2)]
        out.append(dict(zn_index=a.index, CN=len(lig), kinds=kinds, angles=ang))
    return out


def run(args):
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    plat = mm.Platform.getPlatformByName(args.platform)
    props = {"Precision": args.precision} if args.platform == "CUDA" else {}
    log(f"[plat] {args.platform} {props}")

    for seed in range(args.seeds):
        top, system, pos, dt_ps, tag = make_system(
            args.system, args.method, args.zn_charge, args.box,
            verbose=(seed == 0), engine=args.engine)
        label = f"ZN_{tag}_seed{seed}"
        log(f"\n[{seed+1}/{args.seeds}] {label}  dt={dt_ps*1000:.0f} fs  "
            f"prod={args.ns} ns  ({top.getNumAtoms()} 原子)")
        # 产物落点必须打出来。2026-09-05 实测：L_IPS_RESULTS_DIR 没在那个 shell
        # 里导出，作业静默写回了默认目录，撞上同名旧数据被守卫拦下 —— 而日志里
        # 完全看不出它在往哪写，光看"已存在"三个字排查不出原因。
        log(f"      产物目录 {SCRIPT_DIR}")

        # 守卫必须在**平衡之前**。它原本在平衡之后，于是 2026-09-05 同一个撞名
        # 让两次运行各白烧了 9.5 分钟的 NVT+NPT 才报错。检查的是文件名，不需要
        # 任何模拟结果，没有理由等在后面。
        dcd = os.path.join(SCRIPT_DIR, f"{label}_traj.dcd")
        csv = os.path.join(SCRIPT_DIR, f"{label}_state.csv")
        # --force 会毁掉已有轨迹。2026-09-05 实测：一次 0.5 ns 的验证跑用 --force
        # 覆盖掉了一条 5 ns 生产轨迹（984 MB -> 61 MB），无备份、不可恢复。
        # 所以覆盖前先把要毁掉的东西的大小和日期打出来 —— 让人在看到 --force
        # 生效之前就有机会喊停。
        if os.path.exists(dcd) and args.force:
            _st = os.stat(dcd)
            log(f"      ⚠ --force：即将覆盖 {os.path.basename(dcd)}"
                f"（{_st.st_size/1e6:.0f} MB，{time.strftime('%Y-%m-%d %H:%M', time.localtime(_st.st_mtime))}）")
        if os.path.exists(dcd) and not args.force:
            raise SystemExit(
                f"{os.path.basename(dcd)} 已存在。拒绝覆盖 —— 两个进程写同一路径会把帧"
                f"交错在一起且不报错（2026-09-02 实测毁掉两条轨迹）。\n"
                f"要重跑请显式加 --force，或先把旧产出搬走。")
        nsteps = int(round(args.ns * 1000 / dt_ps))
        every = max(1, int(round(args.report_ps / dt_ps)))


        integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond,
                                            dt_ps * unit.picoseconds)
        integ.setRandomNumberSeed(1000 + seed)
        sim = app.Simulation(top, system, integ, plat, props) if props else \
            app.Simulation(top, system, integ, plat)
        sim.context.setPositions(pos)
        t0 = time.time()
        sim.minimizeEnergy(maxIterations=2000)
        sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 1000 + seed)
        sim.step(int(round(args.nvt_ps / dt_ps)))                    # NVT
        bar = mm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin, 25)
        bi = system.addForce(bar); sim.context.reinitialize(preserveState=True)
        sim.step(int(round(args.npt_ps / dt_ps)))                    # NPT
        system.removeForce(bi); sim.context.reinitialize(preserveState=True)
        log(f"      平衡完成 NVT {args.nvt_ps} ps + NPT {args.npt_ps} ps "
            f"（{time.time()-t0:.0f}s）")

        # 每条轨迹自带拓扑快照 —— 体系一旦被重建，事后就再也找不到匹配的拓扑了
        # （2026-09-03 实测：1AAY/amber/solvated.pdb 从 32818 被覆盖成 32794，
        #   8 条老轨迹当场无法分析，靠 amoeba 那份恰好留着的旧副本才救回来）。
        with open(os.path.join(SCRIPT_DIR, f"{label}_top.pdb"), "w") as _tf:
            app.PDBFile.writeFile(top, pos, _tf, keepIds=True)
        sim.reporters.append(app.DCDReporter(dcd, every))
        sim.reporters.append(app.StateDataReporter(
            csv, every, step=True, time=True, potentialEnergy=True,
            kineticEnergy=True, temperature=True, density=True, volume=True))
        sim.reporters.append(app.StateDataReporter(
            sys.stdout, every * 20, step=True, time=True, temperature=True,
            # currentStep 是全局的，平衡段已经走掉 (nvt+npt)/dt 步。传 nsteps
            # 会让进度条在跑到一半时"走完"，之后剩余时间变负，被格式化成
            # 23:59:31 那种假数字。
            speed=True, remainingTime=True,
            totalSteps=sim.currentStep + nsteps))
        t1 = time.time()
        sim.step(nsteps)
        log(f"      production 完成 {args.ns} ns，{time.time()-t1:.0f}s "
            f"({args.ns*86400/(time.time()-t1):.1f} ns/day)")

        st = sim.context.getState(getPositions=True)
        zc = zn_coordination(top, st.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        for k, z in enumerate(zc, 1):
            log(f"      末帧 Zn{k}: CN={z['CN']}  {z['kinds']}  "
                f"角度均值 {np.mean(z['angles']):.1f}°" if z['angles'] else "")
        with open(os.path.join(SCRIPT_DIR, f"{label}_zn_final.json"), "w") as f:
            # cwld_engine: which CWLD implementation produced this trajectory.
            # The plugin and CustomGBForce agree to p99 3.3e-3 per-atom force
            # but are not bitwise identical, so a statistic must not mix them --
            # and nothing else in a dcd says which one ran.
            json.dump(dict(label=label, method=args.method, system=args.system,
                           cwld_engine=_load_main().cwld_engine_of(system),
                           q_zn=args.zn_charge, ns=args.ns, seed=seed, zn=zc), f,
                      ensure_ascii=False, indent=2)
        log(f"      -> {os.path.basename(dcd)} / {os.path.basename(csv)} / {label}_zn_final.json")
        del sim, integ, system


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=["water", "1aay"])
    ap.add_argument("--method", required=True, choices=["pme", "pme_q", "cwld", "amoeba"])
    ap.add_argument("--ns", type=float, required=True, help="production 长度")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--zn-charge", type=float, default=1.0, help="仅 method=pme_q")
    ap.add_argument("--box", type=float, default=3.5, help="仅 system=water")
    ap.add_argument("--nvt-ps", type=float, default=100.0)
    ap.add_argument("--npt-ps", type=float, default=200.0)
    ap.add_argument("--report-ps", type=float, default=2.0)
    ap.add_argument("--engine", choices=("plugin", "customgb"), default="plugin",
                    help="CWLD 实现。plugin=LocalCWLDForce 原生插件（默认，快 3x）；"
                         "customgb=旧的 CustomGBForce。两者数值等价但不是逐比特相同，"
                         "**补跑 2026-09-05 之前的既有矩阵必须用 customgb**，否则"
                         "同一个统计量里混了两个引擎。仅 method=cwld 有效。")
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--force", action="store_true",
                    help="允许覆盖已存在的产出（默认拒绝，防止并行作业互相踩）")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
