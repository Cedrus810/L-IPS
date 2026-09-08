#!/usr/bin/env python
"""Zn 配位的逐帧分析 —— 把"末帧一个分"换成连续统计量。

为什么要重做（见对话 2026-09-02）：首轮只在末帧打了一个分，得到
CWLD 7/9 vs PME 4/9 核心完整。方向对，但 Fisher p≈0.16 不显著，
而且同一 seed 的 3 个位点不独立。**每个位点"核心完整的时间占比"是连续量，
统计力强得多。**

三个指标（都按位点、按帧算）：
  core_intact : nS>=2 且 nN>=2 —— Cys₂His₂ 核心还在
  exact4      : CN==4 且恰为 2S+2N —— 严格正确
  frac180     : 角度 >160° 的占比 —— **唯一干净的几何判别量**
                （八面体 3/15=20%，四面体 0%；角度**均值**是简并的：
                  八面体 (12*90+3*180)/15=108° ≈ 四面体 109.5°，不能用）

同时做**截断对齐**：AMOEBA 只跑 1 ns，PME/CWLD 跑 5 ns。跑得久当然散得多，
所以除了全长，还要在 t<=--align-ns 的同一窗口里比一次，排掉长度混杂。

距离一律用最小镜像（DCDReporter 写的是包裹过的坐标，配体可能在盒子另一侧）。

用法：
    python zn_coordination.py                      # 全部 1AAY 轨迹
    python zn_coordination.py --align-ns 1.0       # 截断窗口（默认 1.0）
"""
from __future__ import annotations

import argparse, glob, itertools, os, re, sys
import numpy as np
import pandas as pd

from lips.paths import PROJECT_ROOT, DATA_DIR as _DATA_DIR
SCRIPT_DIR = str(PROJECT_ROOT)   # 兼容旧写法；新代码请直接用 lips.paths
ZN_CUT_A = 2.8


def log(m): print(m, flush=True)


def _min_image(d, box):
    """最小镜像。box 为 (3,3)，OpenMM 归约三斜取向。"""
    a, b, c = box[0], box[1], box[2]
    d = d - c * np.round(d[:, 2:3] / c[2])
    d = d - b * np.round(d[:, 1:2] / b[1])
    d = d - a * np.round(d[:, 0:1] / a[0])
    return d


def crystal_ligand_map(pdb=None, cutoff=2.8):
    """从晶体 1AAY.pdb 算出每个 Zn 的配体 {(resSeq, atomname), ...}，按 Zn 的 resSeq 索引。

    这是"配体是谁"的权威来源 —— 晶体结构给定的事实，不从模拟帧里猜。
    """
    pdb = pdb or os.path.join(SCRIPT_DIR, "1AAY", "1AAY.pdb")
    A = []
    for l in open(pdb):
        if l.startswith(("ATOM", "HETATM")):
            A.append((l[12:16].strip(), l[17:20].strip(), int(l[22:26]),
                      l[76:78].strip(),
                      np.array([float(l[30 + 8 * i:38 + 8 * i]) for i in range(3)])))
    out = {}
    for nm, rn, rs, el, xyz in A:
        if rn != "ZN":
            continue
        lig = set()
        for nm2, rn2, rs2, el2, x2 in A:
            if rn2 == "ZN" or el2 not in ("N", "O", "S"):
                continue
            if np.linalg.norm(x2 - xyz) < cutoff:
                lig.add((rs2, nm2))
        out[rs] = sorted(lig)
    return out


def classify(resname, elem):
    if resname in ("HOH", "WAT", "TIP3"):
        return "water"
    if elem == "S":
        return "cys_S"
    if resname.startswith("HI") and elem == "N":
        return "his_N"
    return "other"


def analyze_one(dcd, top_pdb, report_ps=2.0, chunk=200, cutoff=ZN_CUT_A):
    """report_ps: 帧间真实间隔。

    ⚠ **不要用 mdtraj 的 `tr.time`**。实测它对本项目所有 dcd 都报 1.00 ps/帧，
    而真实间隔是 2 ps（run_zn_job 的 --report-ps 默认 2.0），**一律差 2 倍**。
    首轮就是这么把"截断到 1 ns"实际截成了真实 2 ns，解体时刻也全部小一半。
    时间一律由帧序号 × report_ps 算，不依赖 dcd 头。
    """
    import mdtraj as md
    from lips.analysis.dcd_integrity import check_dcd
    t0 = md.load_frame(dcd, 0, top=top_pdb)
    # 硬校验：头声称帧数必须与文件大小一致。两次真实损坏都发生在头没变、内容变了的情况下
    check_dcd(dcd, strict=True)      # 只查自洽性；期望帧数由产出侧(run_zn_job)把关
    atoms = list(t0.topology.atoms)
    zn = [a.index for a in atoms if a.residue.name in ("ZN", "Zn", "Zn2+")]
    cand = np.array([a.index for a in atoms
                     if a.element is not None and a.element.symbol in ("N", "O", "S")])
    kinds = np.array([classify(atoms[i].residue.name,
                               atoms[i].element.symbol) for i in cand])

    x0 = t0.xyz[0] * 10.0                      # nm -> Å
    box0 = (t0.unitcell_vectors[0] * 10.0) if t0.unitcell_vectors is not None else None

    # --- 指定配体：按**晶体结构定的残基号**锁定，与任何一帧无关 -----------
    # 不能用"轨迹第 0 帧最近的 4 个"：production 第 0 帧已经是平衡之后（最小化+
    # 100ps NVT + 200ps NPT），若位点在平衡阶段就被拉开（实测 cwld_seed2 Zn2
    # 第 0 帧最远配体 3.89 Å），按最近邻去认就会认错，或者被自检拦下来。
    # 而"配体是谁"本来就是晶体结构给定的事实，不该从模拟帧里猜。
    lig_by_zn = crystal_ligand_map()
    zn_res = [atoms[i].residue for i in zn]
    desig = {}
    for zi_n, (zi, zres) in enumerate(zip(zn, zn_res), 1):
        want = lig_by_zn.get(int(zres.resSeq))
        if want is None:                      # ZN 残基号对不上晶体，退回按顺序取
            want = lig_by_zn[sorted(lig_by_zn)[zi_n - 1]]
        idx = []
        for (rs, an) in want:
            hit = [a.index for a in atoms
                   if a.residue.resSeq == rs and a.name == an]
            if len(hit) != 1:
                raise RuntimeError(f"{os.path.basename(dcd)} Zn{zi_n}: "
                                   f"找不到唯一的 resSeq={rs} name={an}（命中 {len(hit)} 个）")
            idx.append(hit[0])
        desig[zi_n] = dict(S=[i for i in idx if atoms[i].element.symbol == "S"],
                           N=[i for i in idx if atoms[i].element.symbol == "N"])
        # 第 0 帧的距离只报不拦 —— 大值本身就是"平衡阶段已破损"这个发现
        d0 = x0[idx] - x0[zi]
        if box0 is not None:
            d0 = _min_image(d0, box0)
        r0 = np.linalg.norm(d0, axis=1)
        if r0.max() > 3.0:
            log(f"        [注意] Zn{zi_n} 第 0 帧最远配体 {r0.max():.2f} Å "
                f"—— 该位点在**平衡阶段**就已被拉开，不是分析错误")

    rows = []
    frame_no = 0
    for tr in md.iterload(dcd, top=top_pdb, chunk=chunk):
        xyz = tr.xyz * 10.0                      # nm -> Å
        box = (tr.unitcell_vectors * 10.0) if tr.unitcell_vectors is not None else None
        for fi in range(tr.n_frames):
            t_ps = frame_no * report_ps          # 由帧序号算，不信 dcd 头
            frame_no += 1
            for zi_n, zi in enumerate(zn, 1):
                d = xyz[fi][cand] - xyz[fi][zi]
                if box is not None:
                    d = _min_image(d, box[fi])
                r = np.linalg.norm(d, axis=1)
                sel = r < cutoff
                k = kinds[sel]
                v = d[sel] / r[sel][:, None]
                ang = [float(np.degrees(np.arccos(np.clip(np.dot(p, q), -1, 1))))
                       for p, q in itertools.combinations(v, 2)]
                nS = int((k == "cys_S").sum()); nN = int((k == "his_N").sum())
                dg = desig[zi_n]
                def _r(idx):
                    if not idx: return []
                    dd = xyz[fi][idx] - xyz[fi][zi]
                    if box is not None:
                        dd = _min_image(dd, box[fi])
                    return list(np.linalg.norm(dd, axis=1))
                rS, rN = _r(dg["S"]), _r(dg["N"])
                rows.append(dict(t_ps=t_ps, zn=zi_n, CN=int(sel.sum()),
                                 rS_mean=float(np.mean(rS)) if rS else np.nan,
                                 rS_max=float(np.max(rS)) if rS else np.nan,
                                 rN_mean=float(np.mean(rN)) if rN else np.nan,
                                 rN_max=float(np.max(rN)) if rN else np.nan,
                                 rN_bound=float(np.mean([x < 2.5 for x in rN])) if rN else np.nan,
                                 nS=nS, nN=nN,
                                 nW=int((k == "water").sum()),
                                 nO=int((k == "other").sum()),
                                 core_intact=int(nS >= 2 and nN >= 2),
                                 exact4=int(sel.sum() == 4 and nS == 2 and nN == 2),
                                 frac180=float(np.mean([a > 160 for a in ang])) if ang else 0.0))
    return pd.DataFrame(rows)


def _pick_topology(dcd):
    """按 dcd 头里的原子数挑一个匹配的拓扑 PDB。"""
    from lips.analysis.dcd_integrity import dcd_header
    _, natoms, _ = dcd_header(dcd)
    cands = sorted(glob.glob(os.path.join(SCRIPT_DIR, "1AAY", "*", "solvated.pdb")))
    for c in cands:
        n = sum(1 for l in open(c) if l.startswith(("ATOM", "HETATM")))
        if n == natoms:
            return c
    return None


def main(args=None):
    if args is None:
        args = build_parser().parse_args()
    pat = os.path.join(SCRIPT_DIR, "ZN_1aay_*_traj.dcd")
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"没找到 {pat}")
    all_rows = []
    for f in files:
        base = os.path.basename(f).replace("_traj.dcd", "")
        m = re.match(r"ZN_1aay_(.+)_seed(\d+)$", base)
        method, seed = m.group(1), int(m.group(2))
        # 按**原子数**选拓扑，不按方法名。原因：体系被重建过，
        # 1AAY/amber/solvated.pdb 从 32818 变成了 32794，而 PME/pme_q/AMOEBA 那批
        # 轨迹是 32818 的。恰好 1AAY/amoeba/solvated.pdb 仍是那份旧溶剂化
        # （当初从 amber 复制、只把 CYM 改成 CYD），原子序完全一致，可以拿来当拓扑。
        # 教训：每条轨迹应当在产出时**自带**拓扑快照，不该靠事后找。
        top_pdb = _pick_topology(f)
        if top_pdb is None:
            log(f"[skip] {base}: 找不到原子数匹配的拓扑"); continue
        log(f"[分析] {base}  (拓扑 {os.path.relpath(top_pdb, SCRIPT_DIR)})")
        df = analyze_one(f, top_pdb, report_ps=args.report_ps)
        df["method"], df["seed"] = method, seed
        all_rows.append(df)
        log(f"        {df['t_ps'].max()/1000:.2f} ns, {len(df)//3} 帧/位点")
    D = pd.concat(all_rows, ignore_index=True)
    D.to_csv(os.path.join(SCRIPT_DIR, "zn_traj_timeseries.csv"), index=False)

    def table(sub, title):
        log("\n" + "=" * 78); log(title); log("=" * 78)
        g = sub.groupby("method").agg(
            n_run=("seed", "nunique"), ns=("t_ps", lambda x: x.max() / 1000),
            core_intact=("core_intact", "mean"), exact4=("exact4", "mean"),
            CN=("CN", "mean"), frac180=("frac180", "mean"))
        g2 = sub.groupby("method").agg(rN=("rN_mean", "mean"), rNmax=("rN_max", "mean"),
                                       rS=("rS_mean", "mean"), rNb=("rN_bound", "mean"))
        log(f"{'method':10s} {'run':>3s} {'ns':>5s} {'核心完整':>9s} {'严格CN=4':>9s} "
            f"{'平均CN':>6s} {'~180°':>7s} | {'Zn-N均':>7s} {'Zn-N最远':>8s} {'Zn-S均':>7s} {'N<2.5Å占比':>10s}")
        for k, r in g.iterrows():
            q = g2.loc[k]
            log(f"{k:10s} {int(r.n_run):3d} {r.ns:5.1f} {r.core_intact:9.1%} "
                f"{r.exact4:9.1%} {r.CN:6.2f} {r.frac180:7.1%} | "
                f"{q.rN:7.3f} {q.rNmax:8.3f} {q.rS:7.3f} {q.rNb:10.1%}")
        # 按 (seed, 位点) 给出散度 —— 9 个位点各自的时间占比才是可做统计的样本
        per = sub.groupby(["method", "seed", "zn"]).agg(
            core_intact=("core_intact", "mean"), rN_mean=("rN_mean", "mean"),
            rN_max=("rN_max", "mean")).reset_index()
        for col, lab in (("core_intact", "核心完整时间占比（二值判据）"),
                         ("rN_mean", "Zn–N(His) 平均距离 Å（连续量，越小越好）")):
            log(f"\n  每位点 {lab}：")
            for meth, blk in per.groupby("method"):
                v = blk[col].values
                log(f"    {meth:10s} " + " ".join(f"{x:.2f}" for x in v) +
                    f"   均值 {v.mean():.3f} ± {v.std(ddof=1)/np.sqrt(len(v)):.3f} (n={len(v)})")
            # CWLD vs PME 的两样本 t 检验（样本单位=位点，帧内相关不算独立样本）
            if {"cwld", "pme"} <= set(per["method"]):
                x = per[per.method == "cwld"][col].values
                y = per[per.method == "pme"][col].values
                d = x.mean() - y.mean()
                se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
                sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
                need = 16 * (sd / abs(d)) ** 2 if d else np.inf
                log(f"      CWLD-PME = {d:+.3f}, sem {se:.3f} -> {abs(d)/se:.2f}σ "
                    f"{'显著' if abs(d)/se > 2 else '**不显著**'}"
                    f"; 80%power 需每组 {need:.0f} 位点 = {need/3:.0f} seed")
        return per

    per_full = table(D, "全长（各方法自己的长度）")
    per_al = None
    for w in args.windows:
        per_al = table(D[D["t_ps"] <= w * 1000],
                       f"截断对齐：t ≤ {w} ns（真实时间，已修正 mdtraj 的 2 倍偏差）")
    per_full.assign(window="full").to_csv(
        os.path.join(SCRIPT_DIR, "zn_traj_per_site_full.csv"), index=False)
    per_al.assign(window=f"le{args.windows[-1]}ns").to_csv(
        os.path.join(SCRIPT_DIR, "zn_traj_per_site_aligned.csv"), index=False)

    # 解体时间尺度 —— 直接回答"1 ns 的 AMOEBA 够不够"
    log("\n" + "=" * 78)
    log("解体时间尺度：核心第一次破掉的时刻（判断 1 ns 窗口有没有信息量）")
    log("=" * 78)
    for (meth, seed, z), blk in D.groupby(["method", "seed", "zn"]):
        b = blk.sort_values("t_ps")
        bad = b[b["core_intact"] == 0]
        first = bad["t_ps"].iloc[0] / 1000 if len(bad) else None
        log(f"  {meth:10s} seed{seed} Zn{z}: "
            + (f"首次破损 @ {first:.2f} ns" if first is not None else "全程完好"))
    log(f"\n-> zn_traj_timeseries.csv / zn_traj_per_site_full.csv / zn_traj_per_site_aligned.csv")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-ps", type=float, default=2.0,
                    help="帧间真实间隔；不要信 dcd 头（mdtraj 对本项目 dcd 报 1.0，实为 2.0）")
    ap.add_argument("--windows", type=float, nargs="*", default=[1.0, 2.0, 3.0],
                    help="截断对齐窗口（ns，真实时间）")
    return ap


if __name__ == "__main__":
    raise SystemExit(main())
