#!/usr/bin/env python
"""dens / Δq 的实现间一致性 + 真实体系上的耗时对比。

回答两个被混为一谈的问题：

1. **正确性**：`_local_dens_at`（`MATH_CHANGE_MAP` 的 #3，分析侧镜像，也是
   REPORT_1AAY_ZN §4.1 那三个 Δq 数字的唯一来源）从来没跟别的实现比过。
   本脚本用两条独立实现给它做对照：
     * `vec`  —— 本文件里按 CustomGBForce 的 density_expr 重写的**向量化** numpy 版，
                 走 `query_ball_tree` 而不是逐原子 `query_ball_point`，代码路径不同；

2. **速度**：这里测的不是 MD 的 ns/day，是**分析侧**对 K 个目标原子求 dens 的成本。
   `_local_dens_at` 逐原子 Python 循环，成本随 K 线性增长；向量化版一次算完，
   与 K 基本无关。所以存在一个交叉点，本脚本把它测出来而不是猜。
   预期：K=13（定点探针）逐原子更快；K≈6 万（水 q-profile）向量化更快。
   `openmm-localcwld` 的两份 CPU 实现（reference / fast）与本脚本无关，不引入。

**dens 与 dpolar 无关**（dpolar 只进 Δq），所以两种配置下 dens 应当逐位相同。
脚本会显式断言这一点——若不同，说明覆盖改到了不该改的东西。

用法（不要在有作业跑的时候占 CPU）：
    python -m lips.analysis.dens_parity                        # 用 solvated.pdb 单帧
    python -m lips.analysis.dens_parity --dcd <traj.dcd> --frame 0
    python -m lips.analysis.dens_parity --sizes 13,200,2000,all
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from lips import paths


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------
# 独立向量化实现：直接照 CustomGBForce 的 density_expr 写，不复用 _local_dens_at
# --------------------------------------------------------------------------
def dens_vectorized(targets, coords, box, has_box, meta, r_env):
    """dens_i = Σ_j src_w[j]·(1-(r/r_env)²)²·charge_mod[j]，
    j 取 dens_source>0、与 i 不同分子、0<r<r_env 的原子。"""
    mol_ids = meta["mol_ids"]
    src_w = meta["source_class_weight"]
    cmod = meta["charge_mod_array"]
    source_atoms = np.where(meta["dens_source"] > 0.0)[0]
    if len(source_atoms) == 0:
        return np.zeros(len(targets))

    src_xyz = coords[source_atoms]
    tgt_xyz = coords[targets]
    if has_box:
        src_tree = cKDTree(src_xyz, boxsize=box)
        tgt_tree = cKDTree(tgt_xyz, boxsize=box)
    else:
        src_tree = cKDTree(src_xyz)
        tgt_tree = cKDTree(tgt_xyz)

    # 与逐原子版不同的代码路径：一次 tree-vs-tree，而不是 K 次 query_ball_point
    nbr = tgt_tree.query_ball_tree(src_tree, r_env)
    counts = np.fromiter((len(x) for x in nbr), dtype=int, count=len(nbr))
    if counts.sum() == 0:
        return np.zeros(len(targets))

    ii = np.repeat(np.arange(len(targets)), counts)          # 目标序号
    jj = source_atoms[np.concatenate([np.asarray(x, dtype=int) for x in nbr if len(x)])]
    idx_i = np.asarray(targets)[ii]

    delta = coords[jj] - coords[idx_i]
    if has_box:
        delta -= box * np.round(delta / box)
    r = np.linalg.norm(delta, axis=1)

    keep = (r > 0.0) & (r < r_env) & (jj != idx_i)
    keep &= np.abs(mol_ids[idx_i] - mol_ids[jj]) > 0.5       # 同分子排除，逐位照抄
    if not keep.any():
        return np.zeros(len(targets))

    w = (1.0 - (r[keep] / r_env) ** 2) ** 2
    contrib = src_w[jj[keep]] * w * cmod[jj[keep]]
    out = np.zeros(len(targets))
    np.add.at(out, ii[keep], contrib)
    return out


# --------------------------------------------------------------------------
def build_meta(ligand_dpolar=None, scope="metal"):
    """按给定配置重建 meta。env 必须在 import lips.engine.v26 之前设好。"""
    if ligand_dpolar is None:
        os.environ.pop("L_IPS_LIGAND_DPOLAR", None)
    else:
        os.environ["L_IPS_LIGAND_DPOLAR"] = str(ligand_dpolar)
        os.environ["L_IPS_LIGAND_SCOPE"] = scope

    # 只有这几个必须**延迟** import：env 要在 import v26 之前设好（见上）。
    # `paths` 没有这个约束，已提到模块级——它在 main() 里用，放这里是个真 NameError。
    import importlib
    from lips import systems as ls
    from lips.engine import v26 as lips
    importlib.reload(lips)                      # 让模块级 env 读取重新生效

    top, system = ls.load_reference_system(name="1aay")   # 含离子名归一化 + HID/HIE 还原
    meta = lips.build_phase_cwld_metadata(system, top)
    return top, system, meta, lips


def coordinating_atoms(top, meta, lips):
    """被 dpolar 覆盖命中的原子；覆盖关时退回到同一批（按残基/原子名）。"""
    ov = meta.get("ligand_override_indices")
    # `if ov:` 对 numpy 数组是 ValueError（size!=1 时真值歧义），而覆盖关掉时
    # 这里正好是空数组 —— 所以第一次跑就炸在 override_off 那一档。
    if ov is not None and len(ov) > 0:
        return np.asarray(sorted(ov), dtype=int)
    names = {("HID", "NE2"), ("HIE", "ND1"), ("CYM", "SG"), ("CYD", "SG")}
    out = [a.index for a in top.atoms()
           if (a.residue.name, a.name) in names]
    return np.asarray(sorted(out), dtype=int)


def main(a=None):
    if a is None:
        a = build_parser().parse_args()

    # ---- 帧 ----
    if a.dcd:
        import mdtraj as md
        # 走 lips.paths，不要用相对路径：CWD 不在项目根时相对路径直接找不到文件，
        # 而这个脚本装成了 console script，从任何目录都可能被调起。
        top_path = a.top or paths.system_dir("1AAY", "amber", "solvated.pdb")
        tr = md.load_frame(a.dcd, a.frame, top=top_path)
        xyz, ucl = tr.xyz[0], tr.unitcell_lengths
        box = None if ucl is None else np.asarray(ucl[0], dtype=float)
    else:
        import mdtraj as md
        tr = md.load(paths.system_dir("1AAY", "amber", "solvated.pdb"))
        xyz = tr.xyz[0]
        box = None if tr.unitcell_lengths is None else np.asarray(tr.unitcell_lengths[0], dtype=float)
    has_box = box is not None and np.all(np.isfinite(box)) and np.all(box > 0.0)

    results = {}
    for tag, dp in (("override_off", None), ("override_m0p15", -0.15)):
        top, system, meta, lips = build_meta(dp, a.scope)
        coords = lips.wrap_into_box(xyz, box) if has_box else xyz
        lig = coordinating_atoms(top, meta, lips)
        if len(lig) == 0:
            raise SystemExit(
                f"[{tag}] 一个配位原子都没命中 —— 大概率是残基名没还原成 HID/HIE/CYM"
                f"（PDB 往返会塌回 HIS）。先查 meta.json 的 hid_resids/hie_resids。")

        log("")
        log("=" * 74)
        log(f"配置 {tag}：{top.getNumAtoms()} 原子，覆盖命中 {len(lig)} 个，"
            f"dens_source {int((meta['dens_source']>0).sum())} 个")

        # ---- 逐原子参数 dump（五项 + 元素 + is_charged_site 的代理）----
        atoms = list(top.atoms())
        log(f"{'原子':<22}{'dpolar':>9}{'is_pol':>8}{'phase':>7}{'src_w':>7}{'d_src':>7}{'elem':>6}")
        for i in lig:
            at = atoms[i]
            log(f"{at.residue.name}{at.residue.id}:{at.name:<12}"
                f"{meta['dpolar'][i]:>9.4f}{meta['is_polar'][i]:>8.1f}"
                f"{meta['static_phase'][i]:>7.1f}{meta['source_class_weight'][i]:>7.2f}"
                f"{meta['dens_source'][i]:>7.1f}"
                f"{(at.element.symbol if at.element else '?'):>6}")

        # ---- 三条实现算 dens ----
        source_atoms = np.where(meta["dens_source"] > 0.0)[0]
        tree = cKDTree(coords[source_atoms], boxsize=box) if has_box else cKDTree(coords[source_atoms])

        t0 = time.perf_counter()
        d_loop = np.array([lips._local_dens_at(int(i), coords, box, has_box, tree,
                                               source_atoms, meta, a.r_env) for i in lig])
        t_loop = time.perf_counter() - t0

        t0 = time.perf_counter()
        d_vec = dens_vectorized(lig, coords, box, has_box, meta, a.r_env)
        t_vec = time.perf_counter() - t0

        dev = np.abs(d_loop - d_vec)
        rel = dev / np.maximum(np.abs(d_loop), 1e-12)
        log("")
        log(f"[dens] loop vs vec  max|Δ| = {dev.max():.3e}   max rel = {rel.max():.3e}"
            f"   ({t_loop*1e3:.2f} ms vs {t_vec*1e3:.2f} ms, K={len(lig)})")

        # ---- Δq：分级报，近零处同时给绝对值 ----
        dq_loop = np.array([lips._delta_q_from_dens(int(i), d, meta, a.rho0, a.k_polar)
                            for i, d in zip(lig, d_loop)])
        dq_vec = np.array([lips._delta_q_from_dens(int(i), d, meta, a.rho0, a.k_polar)
                           for i, d in zip(lig, d_vec)])
        raw = (meta["static_phase"][lig] * meta["is_polar"][lig] * meta["dpolar"][lig]
               * np.tanh(a.k_polar * d_loop / a.rho0))
        clamp = lips.Q_DELTA_CLAMP
        log(f"[tanh] 内层自变量 k·dens/rho0 ∈ [{(a.k_polar*d_loop/a.rho0).min():.4f},"
            f" {(a.k_polar*d_loop/a.rho0).max():.4f}]")
        log(f"[clamp] |raw|/clamp ∈ [{np.abs(raw).min()/clamp:.4f}, {np.abs(raw).max()/clamp:.4f}]"
            f"   （<0.05 时 clamp 才近似恒等；大了要先扣掉 tanh 非线性）")
        log(f"[Δq  ] loop vs vec  max|Δ| = {np.abs(dq_loop-dq_vec).max():.3e}")
        log(f"[Δq  ] 均值 = {dq_loop.mean():+.5f} e = clamp 的 {100*dq_loop.mean()/clamp:+.2f}%"
            f"   （逐原子见下）")
        for i, d, q in zip(lig, d_loop, dq_loop):
            at = atoms[i]
            log(f"    {at.residue.name}{at.residue.id}:{at.name:<8}"
                f"dens={d:8.4f}  Δq={q:+.6f} e = clamp 的 {100*q/clamp:+6.2f}%")

        results[tag] = dict(lig=lig, dens=d_loop, dq=dq_loop, meta=meta)

        # ---- 耗时随 K 的走势 ----
        if a.sizes:
            all_sinks = np.where(meta["dens_sink"] > 0.0)[0]
            log("")
            log(f"[耗时] 目标原子数 K -> _local_dens_at 逐原子 / 向量化   "
                f"(可选目标池 = dens_sink {len(all_sinks)} 个)")
            for s in a.sizes.split(","):
                s = s.strip()
                k = len(all_sinks) if s == "all" else min(int(s), len(all_sinks))
                if k <= 0:
                    continue          # 空选择会让 cKDTree 抛，而这段在有用输出之后
                sel = all_sinks[np.linspace(0, len(all_sinks) - 1, k, dtype=int)]
                t0 = time.perf_counter()
                _ = [lips._local_dens_at(int(i), coords, box, has_box, tree,
                                         source_atoms, meta, a.r_env) for i in sel]
                tl = time.perf_counter() - t0
                t0 = time.perf_counter()
                _ = dens_vectorized(sel, coords, box, has_box, meta, a.r_env)
                tv = time.perf_counter() - t0
                log(f"    K={k:<7} loop {tl*1e3:9.2f} ms   vec {tv*1e3:9.2f} ms"
                    f"   加速 {tl/max(tv,1e-12):6.2f}x")

    # ---- 跨配置断言：dens 与 dpolar 无关 ----
    a_, b_ = results["override_off"], results["override_m0p15"]
    if np.array_equal(a_["lig"], b_["lig"]):
        dd = np.abs(a_["dens"] - b_["dens"]).max()
        log("")
        log(f"[跨配置] dens 差异 max|Δ| = {dd:.3e}  "
            f"{'✓ 符合预期（dens 不依赖 dpolar）' if dd < 1e-12 else '✗ 异常：覆盖改到了 dens 通路'}")
        log(f"[跨配置] Δq 均值：{a_['dq'].mean():+.5f} -> {b_['dq'].mean():+.5f} e"
            f"   （符号{'翻转' if a_['dq'].mean()*b_['dq'].mean() < 0 else '未变'}）")
    else:
        log("\n[跨配置] 两档命中的原子集合不同，无法逐原子比较")


def build_parser():
    p = argparse.ArgumentParser(description="dens/Δq 实现间一致性与耗时对比")
    p.add_argument("--dcd", default=None, help="轨迹（缺省用 1AAY/amber/solvated.pdb 单帧）")
    p.add_argument("--top", default=None, help="轨迹对应的拓扑 pdb")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--scope", default="metal", choices=["metal", "all"])
    p.add_argument("--r-env", dest="r_env", type=float, default=0.35)
    p.add_argument("--rho0", type=float, default=13.5)
    p.add_argument("--k-polar", dest="k_polar", type=float, default=0.8)
    p.add_argument("--sizes", default="13,200,2000",
                   help="耗时对比的目标原子数，逗号分隔，可含 all；空串关闭")
    return p


if __name__ == "__main__":
    main()
