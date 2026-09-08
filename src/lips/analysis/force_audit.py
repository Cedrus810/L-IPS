#!/usr/bin/env python
"""Stage A · 静态力审计 —— 不跑 MD，直接量各 closure 的真实力误差。

动机（见 docs/reports/STAGE1_CLOSURE_KILLSWITCH_REPORT.md §10 /
docs/plans/PLAN_PSWF_Closure_Family.md）：
pswfz2 与 zmm2 的差别在**带外**（q>2π ⟺ 波长 <1.2 nm），预期动的是金属–水第一/二壳
结构——而那批观测量的 run-to-run 噪声是 40-45%（project_lips_noise_vs_effect），
砍长度/砍 seed 的"减量 MD"注定只能得到假阴性。

所以这里干脆不跑 MD：固定坐标、固定电荷，**唯一变量是 pair kernel**。

    参照 : NonbondedForce + PME（精确 Ewald），LJ 关闭，例外对全排除
    被测 : CustomNonbondedForce + U(r)=(1-A(r/rc))/r，同 cutoff、同排除表

两者算的是同一批粒子对的同一个静电求和，差值 = 纯 closure 误差。
零采样噪声、零 Integrator.step()、几分钟出结果。

输出三层分解：
  1. 总体 RMS 力误差（绝对 kJ/mol/nm 与相对 |F_ref|）
  2. 按物种（二价金属 Ca2+/Zn2+/Mg2+ / 水O / 水H / Na+ / Cl- / 蛋白重原子 / 蛋白H）
  3. 按距最近**二价金属**的距离分箱，输出列 `rel_dM_*`（抓第一壳 0.24nm / 第二壳 0.45nm）

⚠ 第 2、3 项以前写死只认 Ca²⁺（1CKK 钙调蛋白时代）。换到 1AAY 锌指之后锌被静默
归进 protein_heavy，金属那一列和整个分壳 breakdown **消失且不报错**。现已改为按
元素识别所有二价金属，列名 `rel_dCa_*` 相应改为 `rel_dM_*`。

用法：
    lips-force-audit --selftest                       # 合成体系自检，秒级
    lips-force-audit --system 1aay --traj ZN_1aay_pme_seed0_traj.dcd
"""
from __future__ import annotations

import argparse, os, time
import numpy as np

from lips import paths
from lips import systems as ls
import pandas as pd

from lips.paths import DATA_DIR as _DATA_DIR
DATA_DIR = str(_DATA_DIR)

ONE_4PI_EPS0 = 138.93545764498226
RC_DEFAULT = 1.2


def log(m): print(m, flush=True)


# ---------------------------------------------------------------------------
def build_pair(system_template, charges, kernel_expr, rc, exclusions, box,
               platform, precision):
    """返回 (ref_context, test_context)：同一批粒子、同一排除表、LJ 全关。"""
    import openmm as mm
    from openmm import unit

    n = len(charges)

    def blank_system():
        s = mm.System()
        for _ in range(n):
            s.addParticle(1.0)
        s.setDefaultPeriodicBoxVectors(*box)
        return s

    # --- 参照：PME ---
    ref = blank_system()
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(rc * unit.nanometer)
    nb.setEwaldErrorTolerance(1e-6)          # 比要测的误差小若干量级
    nb.setUseDispersionCorrection(False)
    for q in charges:
        nb.addParticle(q, 1.0, 0.0)          # LJ 关闭
    for (i, j) in exclusions:
        nb.addException(i, j, 0.0, 1.0, 0.0)  # 例外对全排除（两侧一致即可）
    ref.addForce(nb)

    # --- 被测：截断 kernel ---
    test = blank_system()
    cnb = mm.CustomNonbondedForce(f"{ONE_4PI_EPS0}*q1*q2*{kernel_expr}")
    cnb.addPerParticleParameter("q")
    cnb.addGlobalParameter("rc", rc)
    cnb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    cnb.setCutoffDistance(rc * unit.nanometer)
    cnb.setUseLongRangeCorrection(False)
    for q in charges:
        cnb.addParticle([q])
    for (i, j) in exclusions:
        cnb.addExclusion(i, j)
    test.addForce(cnb)

    props = {"Precision": precision} if platform.getName() == "CUDA" else {}
    ctxs = []
    for s in (ref, test):
        ctxs.append(mm.Context(s, mm.VerletIntegrator(0.001), platform, props)
                    if props else mm.Context(s, mm.VerletIntegrator(0.001), platform))
    return ctxs


def forces(ctx, pos, box):
    from openmm import unit
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(pos * unit.nanometer)
    st = ctx.getState(getForces=True, getEnergy=True)     # 零 step()
    return (st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
            st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


# ---------------------------------------------------------------------------
DIVALENT_METAL_SPECIES = ls.DIVALENT_METAL_SPECIES   # 单一来源，见 lips.systems


#: 元素符号 -> 规范物种名。按**元素**判，不按残基名判：残基名会因为读盘、
#: 别名表、是否调过 normalize_ion_resnames 而变，元素不会。
DIVALENT_BY_ELEMENT = {"Ca": "Ca2+", "Zn": "Zn2+", "Mg": "Mg2+"}


def species_of(topology):
    """给每个原子打一个物种标签。

    ⚠ 历史坑：原实现只认 `rn in ("Ca2+","CA") and el == "Ca"`。换到 1AAY 锌指
    之后，**锌被静默归进 protein_heavy** —— 金属那一列 (rel_Ca2+) 和整个分壳
    breakdown 都消失了，而且不报错。现在按元素识别所有二价金属。
    """
    lab = []
    for a in topology.atoms():
        rn = a.residue.name
        el = a.element.symbol if a.element is not None else "?"
        if el in DIVALENT_BY_ELEMENT:
            lab.append(DIVALENT_BY_ELEMENT[el])
        elif rn in ("HOH", "WAT", "TIP3"):
            lab.append("water_O" if el == "O" else "water_H")
        elif el == "Na":
            lab.append("Na+")
        elif el == "Cl":
            lab.append("Cl-")
        elif el == "H":
            lab.append("protein_H")
        else:
            lab.append("protein_heavy")
    return np.array(lab)


def run_audit(args):
    import openmm as mm
    import mdtraj as md
    from lips.engine import closure as cw

    # 产物名先定、先查 —— 名字必须带上 precision 与 kernel 签名，否则同一条轨迹
    # 换一组 kernel 重跑会静默覆盖上一次（§7.3 那张表就是这么来的）。
    _tag = os.path.splitext(os.path.basename(
        args.traj if os.path.isabs(args.traj)
        else os.path.join(DATA_DIR, args.traj)))[0]
    for _suf in ("_1ckk", "_traj"):
        if _tag.endswith(_suf):
            _tag = _tag[:-len(_suf)]
    _nk = len([k for k in args.kernels.split(",") if k.strip()])
    out_csv = paths.result_new(
        f"closure_force_audit_{_tag}_{args.precision}"
        f"_{_nk}k{paths.run_sig(args.kernels)}.csv", force=args.force)

    # 体系是参数，不是常量（原先写死 build_1ckk_system()；1CKK 已非当前体系）。
    topology, sys_pme = ls.load_reference_system_from_args(args)

    nb0 = [sys_pme.getForce(i) for i in range(sys_pme.getNumForces())
           if isinstance(sys_pme.getForce(i), mm.NonbondedForce)][0]
    from openmm.unit import elementary_charge
    charges = np.array([nb0.getParticleParameters(i)[0].value_in_unit(elementary_charge)
                        for i in range(sys_pme.getNumParticles())])
    excl = []
    for i in range(nb0.getNumExceptions()):
        p1, p2, *_ = nb0.getExceptionParameters(i)
        excl.append((p1, p2))
    log(f"[build] {len(charges)} atoms, Σq={charges.sum():+.2e}, {len(excl)} 个排除对")

    sp = species_of(topology)
    # 二价金属，不再只认 Ca2+。1CKK（钙调蛋白）时代这里写死 "Ca2+"，换到 1AAY
    # 锌指之后恒为空——分壳那几列于是**静默消失**，不报错也不提示。
    # 现在按物种集合取，并在一个都没有时明说。
    metal_idx = np.where(np.isin(sp, DIVALENT_METAL_SPECIES))[0]
    metal_names = sorted(set(sp[metal_idx].tolist()))
    log(f"[build] 物种: " + ", ".join(f"{k}={int((sp==k).sum())}"
                                      for k in sorted(set(sp))))
    if metal_idx.size:
        log(f"[build] 二价金属 {metal_names} 共 {metal_idx.size} 个 -> 输出分壳误差")
    else:
        log(f"[build] ⚠ 体系里没有二价金属（找过 {list(DIVALENT_METAL_SPECIES)}），"
            f"分壳列 rel_dM_* 会全部缺省")

    traj_path = args.traj if os.path.isabs(args.traj) else os.path.join(DATA_DIR, args.traj)
    md_top = md.Topology.from_openmm(topology)
    ls.assert_traj_matches_system(traj_path, md_top.n_atoms)   # 不匹配就别烧机时
    tr = md.load(traj_path, top=md_top, stride=args.stride)
    if args.frames and tr.n_frames > args.frames:
        sel = np.linspace(0, tr.n_frames - 1, args.frames).astype(int)
        tr = tr.slice(sel)
    log(f"[traj]  {os.path.basename(traj_path)}: {tr.n_frames} frames")

    plat = mm.Platform.getPlatformByName(args.platform)
    log(f"[plat]  {args.platform} precision={args.precision}")

    box0 = [mm.Vec3(*v) for v in tr.unitcell_vectors[0]]
    rows = []
    for name in args.kernels.split(","):
        name = name.strip()
        w = cw.make_window(name)
        expr = w.lepton("rc")
        ref_c, test_c = build_pair(sys_pme, charges, expr, args.rc, excl, box0,
                                   plat, args.precision)
        t0 = time.time()
        acc = {}
        for f in range(tr.n_frames):
            box = [mm.Vec3(*v) for v in tr.unitcell_vectors[f]]
            pos = tr.xyz[f].astype(np.float64)
            Fr, Er = forces(ref_c, pos, box)
            Ft, Et = forces(test_c, pos, box)
            d = Ft - Fr
            d2 = np.sum(d * d, axis=1)
            r2 = np.sum(Fr * Fr, axis=1)
            acc.setdefault("d2", []).append(d2)
            acc.setdefault("r2", []).append(r2)
            acc.setdefault("dE", []).append(Et - Er)
            if metal_idx.size:
                # N x n_metal，n_metal 只有几个，实测 GPU 反而不划算
                # （N=32794 x 3：numpy 6.9 ms vs torch/cuda 4.4 ms，只 1.6x），
                # 所以这里保持 numpy。参照集大到几十以上时才值得搬 GPU。
                dist = np.min(np.linalg.norm(pos[None, metal_idx, :] - pos[:, None, :],
                                             axis=2), axis=1)
                acc.setdefault("dca", []).append(dist)
        d2 = np.concatenate(acc["d2"]); r2 = np.concatenate(acc["r2"])
        spN = np.tile(sp, tr.n_frames)
        row = {"kernel": name,
               "rms_dF": float(np.sqrt(d2.mean())),
               "rms_Fref": float(np.sqrt(r2.mean())),
               "rel_rms": float(np.sqrt(d2.mean() / r2.mean())),
               "dE_mean": float(np.mean(acc["dE"])),
               "dE_std": float(np.std(acc["dE"])),
               "sec": round(time.time() - t0, 1)}
        for k in sorted(set(sp)):
            m = spN == k
            row[f"rel_{k}"] = float(np.sqrt(d2[m].mean() / r2[m].mean()))
        if acc.get("dca"):
            dca = np.concatenate(acc["dca"])
            for lo, hi, tag in ((0.0, 0.32, "shell1"), (0.32, 0.55, "shell2"),
                                (0.55, 1.2, "mid"), (1.2, 1e9, "bulk")):
                m = (dca >= lo) & (dca < hi)
                row[f"rel_dM_{tag}"] = (float(np.sqrt(d2[m].mean() / r2[m].mean()))
                                        if m.sum() else np.nan)
        rows.append(row)
        metal_key = f"rel_{metal_names[0]}" if metal_names else None
        log(f"  {name:8s} rel_rms={row['rel_rms']:.4e}  "
            f"{metal_names[0] if metal_names else 'metal'}="
            f"{row.get(metal_key, float('nan')):.4e}  "
            f"shell1={row.get('rel_dM_shell1', float('nan')):.4e}  ({row['sec']}s)")

    out = pd.DataFrame(rows)
    fn = out_csv                      # 开头就定好并查过存在性，见 run_audit() 顶部
    out.to_csv(fn, index=False)
    log("\n" + "=" * 78)
    log("相对 RMS 力误差（对 PME 精确 Ewald），越小越好")
    log("=" * 78)
    cols = ["kernel", "rel_rms", "rel_Ca2+", "rel_dCa_shell1", "rel_dCa_shell2",
            "rel_water_O", "rel_water_H", "rel_protein_heavy", "dE_mean"]
    cols = [c for c in cols if c in out.columns]
    log(out[cols].to_string(index=False, float_format=lambda v: f"{v:.4e}"))
    base = out[out.kernel == "zmm2"]
    if len(base):
        log("\n对 zmm2 归一（<1 更好）：")
        n = out.copy()
        for c in cols[1:]:
            if c in n: n[c] = n[c] / float(base[c].iloc[0])
        log(n[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log(f"\n-> {os.path.basename(fn)}")


# ---------------------------------------------------------------------------
def selftest(args):
    """合成体系：验证 参照/被测 管线本身正确，不碰 1CKK。"""
    import openmm as mm
    from lips.engine import closure as cw
    rng = np.random.default_rng(0)
    n, Lb = 120, 3.0
    q = rng.normal(0, 0.5, n); q -= q.mean()             # 中性
    pos = rng.uniform(0, Lb, (n, 3))
    box = [mm.Vec3(Lb, 0, 0), mm.Vec3(0, Lb, 0), mm.Vec3(0, 0, Lb)]
    plat = mm.Platform.getPlatformByName("Reference")
    log(f"[selftest] {n} 个电荷, 盒子 {Lb} nm, rc={args.rc}, Reference platform")
    log(f"{'kernel':10s} {'rel_rms 力误差':>16s} {'|F_ref| rms':>14s}")
    prev = None
    for name in args.kernels.split(","):
        w = cw.make_window(name.strip())
        ref_c, test_c = build_pair(None, q, w.lepton("rc"), args.rc, [], box, plat, "double")
        Fr, Er = forces(ref_c, pos, box)
        Ft, Et = forces(test_c, pos, box)
        rel = float(np.sqrt(np.sum((Ft - Fr) ** 2) / np.sum(Fr ** 2)))
        log(f"{name.strip():10s} {rel:16.4e} {float(np.sqrt(np.mean(np.sum(Fr**2,axis=1)))):14.4e}")
        assert np.all(np.isfinite(Ft)) and rel < 1.0, f"{name} 结果异常"
        prev = rel
    log("[selftest] PASS —— 参照/被测两条管线都能跑通且量纲一致")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ls.add_system_arguments(ap)
    # 默认值不能用 PME 轨迹：它是 32818 原子，而 1AAY/amber 现在是 32794，
    # 32818 的 amber system.xml 已在那次覆盖里没了 —— 裸跑必崩（§7.5）。
    ap.add_argument("--traj", default="ZN_1aay_cwld_ligm0p15metal_zmm_seed0_traj.dcd",
                    help="要审计的轨迹文件名（相对 L_IPS_DATA_DIR 或绝对路径）")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--kernels", default="zmm1,zmm2,zmm3,pswfz2")
    ap.add_argument("--rc", type=float, default=RC_DEFAULT)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--precision", default="double")
    ap.add_argument("--force", action="store_true",
                    help="覆盖已存在的产物 csv（默认拒跑）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest(a) if a.selftest else run_audit(a)


if __name__ == "__main__":
    main()
