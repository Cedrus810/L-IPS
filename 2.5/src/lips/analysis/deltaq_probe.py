#!/usr/bin/env python
"""CWLD 在 Zn 配位原子上到底给了多大的 Δq —— 直接看，不用做消融 MD。

动机：1CKK 当初就是这么被判死的 —— 配位羧基氧只动了 clamp 的 0.56%、
全部溶质极性原子 0.15%，说明极化机制**根本没启动**（[project-lips-cwld-vs-pme]）。
1AAY 的 Cys₂His₂ 如果 Δq 是那个量级，那 CWLD 相对 PME 的那点优势就不可能来自
"给配位原子极化"；如果是几十个百分点，机制就确实在这个位点上工作。

用 `compute_atom_delta_q_profile`（生产 CustomGBForce density_expr 的分析侧镜像，
MATH_CHANGE_MAP 表里的 #3），对晶体定死的 12 个配位原子逐帧重算 Δq。

同时报 Δq 与 Zn–N 距离的关系：His 被拉开时 Δq 若变大，那就是"恢复力"的直接证据。

用法：
    python zn_deltaq_probe.py                      # 全部 CWLD 轨迹
    python zn_deltaq_probe.py --stride 20          # 抽帧加速
"""
from __future__ import annotations

import argparse, glob, os
import numpy as np
import pandas as pd

from lips import paths
from lips import systems as ls


def log(m): print(m, flush=True)


def _resseq(res):
    """残基序号。OpenMM 拓扑是 res.id（str），mdtraj 是 res.resSeq（int）。"""
    v = getattr(res, "resSeq", None)
    if v is not None:
        return int(v)
    try:
        return int(res.id)
    except (TypeError, ValueError):
        return -1


def main(a=None):
    if a is None:
        a = build_parser().parse_args()
    import mdtraj as md
    from lips.analysis.zn_coordination import crystal_ligand_map

    from lips.engine import v26 as lips

    # 原先直接读 1AAY/amber/{solvated.pdb,system.xml}，既没还原 HID/HIE 也没
    # normalize_ion_resnames() —— 后者漏掉会让 Zn 的 dens_source 静默变 0。
    # 统一走 lips.systems，这类步骤只在那一个地方实现。
    top, system = ls.load_reference_system(name="1aay")
    log(f"[build] 1AAY/amber: {top.getNumAtoms()} 原子")
    meta = lips.build_phase_cwld_metadata(system, top)
    clamp = lips.Q_DELTA_CLAMP
    log(f"[meta ] Q_DELTA_CLAMP={clamp}；dens_source 原子 "
        f"{int((meta['dens_source']>0).sum())} 个，is_polar 原子 "
        f"{int((meta['is_polar']>0).sum())} 个")

    # --- 晶体定死的 12 个配位原子 + 3 个 Zn ---
    lig = crystal_ligand_map()
    atoms = list(top.atoms())
    zn_idx = [a.index for a in atoms if a.residue.name in ("ZN", "Zn", "Zn2+")]
    want, tag = [], {}
    for zres, pairs in sorted(lig.items()):
        for rs, an in pairs:
            # OpenMM 的 Residue 用 .id（字符串），mdtraj 才有 .resSeq —— 两种都兼容
            hit = [x.index for x in atoms if _resseq(x.residue) == rs and x.name == an]
            assert len(hit) == 1, f"resSeq={rs} name={an} 命中 {len(hit)}"
            want.append(hit[0])
            tag[hit[0]] = f"Zn{zres-200}:{atoms[hit[0]].residue.name}{rs}:{an}"
    # --- 对照组：非配位的蛋白 N / S -------------------------------------
    # 必须有这一组：dpolar 是**按元素**给的（S=0.015 给全部 7 个硫、N=0.012 给全部
    # 152 个氮），配位的 HIS:NE2 只占氮的 3.9%。所以"配位原子 Δq 很大"本身不能证明
    # 响应有靶向性 —— 得跟非配位的同元素原子比，才知道是不是均匀撒胡椒面。
    prot = [x for x in atoms if x.residue.name not in ("HOH", "WAT", "NA", "CL", "ZN")]
    ctrl_N = [x.index for x in prot
              if x.element and x.element.symbol == "N" and x.index not in set(want)]
    ctrl_S = [x.index for x in prot
              if x.element and x.element.symbol == "S" and x.index not in set(want)]
    if len(ctrl_N) > a.n_ctrl:
        ctrl_N = list(np.array(ctrl_N)[np.linspace(0, len(ctrl_N) - 1, a.n_ctrl, dtype=int)])
    for i in ctrl_N:
        tag[i] = f"ctrl_N:{atoms[i].residue.name}{_resseq(atoms[i].residue)}:{atoms[i].name}"
    for i in ctrl_S:
        tag[i] = f"ctrl_S:{atoms[i].residue.name}{_resseq(atoms[i].residue)}:{atoms[i].name}"
    coord_set = set(want)
    want = want + ctrl_N + ctrl_S
    log(f"[lig  ] 配位原子 {len(coord_set)} 个 + 对照 N {len(ctrl_N)} / S {len(ctrl_S)} 个")
    log(f"[check] 配位原子 is_polar={np.unique(meta['is_polar'][sorted(coord_set)])}, "
        f"dpolar={np.unique(meta['dpolar'][sorted(coord_set)])}, "
        f"src_w={np.unique(meta['source_class_weight'][sorted(coord_set)])}")

    out = []
    for f in sorted(glob.glob(paths.result("ZN_1aay_cwld_seed*_traj.dcd"))):
        base = os.path.basename(f).replace("_traj.dcd", "")
        log(f"\n[{base}]")
        from lips.analysis.dcd_integrity import check_dcd
        check_dcd(f, strict=True)
        tr = md.load(f, top=paths.system_dir("1AAY", "amber", "solvated.pdb"),
                     stride=a.stride)
        log(f"   {tr.n_frames} 帧")
        df = lips.compute_atom_delta_q_profile(tr, meta, want, frame_stride=1)
        df["label"] = df["atom_index"].map(tag)
        df["run"] = base
        out.append(df)
        g = df.groupby("label")["delta_Q"].agg(["mean", "std"])
        for k, r in g.iterrows():
            log(f"     {k:26s} Δq = {r['mean']:+.5f} ± {r['std']:.5f} e"
                f"   = clamp 的 {abs(r['mean'])/clamp:6.1%}")
    D = pd.concat(out, ignore_index=True)
    D.to_csv(paths.result("zn_deltaq_profile.csv"), index=False)

    log("\n" + "=" * 74)
    log("汇总：CWLD 在 Zn 配位原子上的电荷响应")
    log("=" * 74)
    def grp(lab):
        if lab.startswith("ctrl_N"): return "对照 非配位 N"
        if lab.startswith("ctrl_S"): return "对照 非配位 S(Met)"
        return "配位 Cys S" if ":SG" in lab else "配位 His N"
    D["grp"] = D["label"].map(grp)
    g = D.groupby("grp")["delta_Q"].agg(["mean", "std", "min", "max", "count"])
    for k in ["配位 Cys S", "配位 His N", "对照 非配位 S(Met)", "对照 非配位 N"]:
        if k not in g.index: continue
        r = g.loc[k]
        log(f"  {k:18s} Δq = {r['mean']:+.5f} ± {r['std']:.5f} e  "
            f"(范围 {r['min']:+.4f}..{r['max']:+.4f})  = clamp 的 {abs(r['mean'])/clamp:6.1%}"
            f"  n={int(r['count'])}")
    if {"配位 His N", "对照 非配位 N"} <= set(g.index):
        a1, a2 = g.loc["配位 His N", "mean"], g.loc["对照 非配位 N", "mean"]
        log(f"\n  **靶向性判据**：配位 His N / 非配位 N = {a1/a2:.2f}×"
            f"  —— 接近 1 就是均匀撒胡椒面，没有靶向性")
    log(f"\n  对照 1CKK（同一套 CWLD、Ca²⁺ 被 EF-hand 螯合死）：")
    log(f"    近 Ca 水 O          clamp 的 7.0%")
    log(f"    螯合 Ca 的羧基 O    clamp 的 0.56%")
    log(f"    全部溶质极性原子    clamp 的 0.15%   <- 机制没启动")
    log(f"\n-> zn_deltaq_profile.csv")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=10, help="抽帧步长")
    ap.add_argument("--n-ctrl", type=int, default=30, help="非配位 N 对照取样个数")
    return ap


if __name__ == "__main__":
    raise SystemExit(main())
