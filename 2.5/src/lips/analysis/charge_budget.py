#!/usr/bin/env python
"""CWLD 的电荷收支：体系总净电荷 + 水分子净 Δq 的一阶/三阶分解。

回答两个问题（都只读现有轨迹，不跑 MD、不用 GPU）：

**Q1 总净电荷 $\\sum_i \\Delta q_i$**（= PLAN §13.4b 的阻塞项 H1）。
倒空间求和丢掉 k=0 等价于假设中性背景，所以路线 B 开工前必须知道 CWLD 把体系
带偏了多少。现有诊断 `compute_water_net_delta_q` 是**每帧 500 个三联体的抽样**，
不能用来推总量 —— 这里**不抽样**，逐原子全算。

**Q2 水的净 Δq 出自哪一项**（报告 §3.4.3 / §9.5-1）。实测每水 +8e-4 e 系统性同号，
两个候选机制：

    一阶：ΔQ_linear = d_O t_O + d_H t_H1 + d_H t_H2        (O/H 各自算 dens ⇒ 一阶不抵消)
    三阶：外层 tanh 的曲率，tanh a - 2 tanh(a/2) = -a³/4 + O(a⁵) > 0

设计意图是 d_H = -d_O/2 让一阶抵消，但生产引擎里 O 与两个 H 的 dens 各自算，
所以一阶本来就破了。**近 Ca/Zn 的水反号提示一阶主导**，但没人分解过。
本脚本直接给出 linear / nonlinear 两项，按「体相 / 近金属」分组。

用法：
    python -m lips.analysis.charge_budget --dcd data/XXX_traj.dcd --frames 5
    python -m lips.analysis.charge_budget            # 缺省用 solvated.pdb 单帧
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from lips import paths


def log(m):
    print(m, flush=True)


def dens_all(coords, box, has_box, meta, r_env, targets):
    """向量化 dens（与 dens_parity.dens_vectorized 同一公式，已验 2e-16）。"""
    from lips.analysis.dens_parity import dens_vectorized
    return dens_vectorized(targets, coords, box, has_box, meta, r_env)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dcd", default=None)
    ap.add_argument("--top", default=None)
    ap.add_argument("--frames", type=int, default=1, help="取多少帧（等间隔）")
    ap.add_argument("--ligand-dpolar", default="-0.15",
                    help="配位原子 dpolar 覆盖；空串=关闭")
    ap.add_argument("--scope", default="metal", choices=["metal", "all"])
    ap.add_argument("--r-env", dest="r_env", type=float, default=0.35)
    ap.add_argument("--rho0", type=float, default=13.5)
    ap.add_argument("--k-polar", dest="k_polar", type=float, default=0.8)
    ap.add_argument("--near-metal-nm", type=float, default=0.35)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    # env 必须在 import v26 之前设好（模块级读取）
    if a.ligand_dpolar.strip():
        os.environ["L_IPS_LIGAND_DPOLAR"] = a.ligand_dpolar.strip()
        os.environ["L_IPS_LIGAND_SCOPE"] = a.scope
    else:
        os.environ.pop("L_IPS_LIGAND_DPOLAR", None)

    # 产物名先定、先查 —— 与 force_audit/nve_drift 同一条规则（约束 #6）。
    # 第一版把检查放在末尾、名字也不带区分维度，于是第二次跑（换了 dcd 和帧数）
    # 在**算完之后**才被守卫拦下 —— 我自己犯了刚加的那条规则。
    _tag = "solvated" if not a.dcd else os.path.splitext(os.path.basename(a.dcd))[0]
    for _suf in ("_traj",):
        if _tag.endswith(_suf):
            _tag = _tag[:-len(_suf)]
    _dp = a.ligand_dpolar.strip() or "off"
    _sig = f"{_tag}_f{a.frames}_lig{_dp}"
    out_total = paths.result_new(f"charge_budget_total_{_sig}.csv", force=a.force)
    out_water = paths.result_new(f"charge_budget_water_split_{_sig}.csv", force=a.force)

    import mdtraj as md
    from lips import systems as ls
    from lips.engine import v26 as lips

    top, system = ls.load_reference_system(name="1aay")
    meta = lips.build_phase_cwld_metadata(system, top)
    clamp = lips.Q_DELTA_CLAMP

    top_path = a.top or paths.system_dir("1AAY", "amber", "solvated.pdb")
    if a.dcd:
        ls.assert_traj_matches_system(a.dcd, top.getNumAtoms())
        n_all = md.open(a.dcd).__len__() if hasattr(md.open(a.dcd), "__len__") else None
        idx = ([0] if a.frames <= 1 or not n_all
               else list(np.linspace(0, n_all - 1, a.frames).astype(int)))
        frames = [md.load_frame(a.dcd, int(i), top=top_path) for i in idx]
    else:
        frames = [md.load(top_path)]
        idx = [0]

    dpolar, sphase, ispolar = meta["dpolar"], meta["static_phase"], meta["is_polar"]
    A = sphase * ispolar * dpolar                       # 每原子的 A_i
    sinks = np.where(meta["dens_sink"] > 0.0)[0]
    triplets = np.asarray(meta["water_triplets"], dtype=int)
    metals = np.array([i for i in meta["ion_indices"]
                       if abs(meta["qbase"][i]) >= 1.5], dtype=int)

    rows, wrows = [], []
    for fi, tr in zip(idx, frames):
        ucl = tr.unitcell_lengths
        box = None if ucl is None else np.asarray(ucl[0], float)
        has_box = box is not None and np.all(np.isfinite(box)) and np.all(box > 0)
        coords = lips.wrap_into_box(tr.xyz[0], box) if has_box else tr.xyz[0]

        # ---- Q1：全体 sink 的 Δq，不抽样 ----
        dens = dens_all(coords, box, has_box, meta, a.r_env, sinks)
        t = np.tanh(a.k_polar * dens / a.rho0)
        dq = clamp * np.tanh(A[sinks] * t / clamp)
        total = dq.sum()
        rows.append(dict(frame=int(fi), n_sink=len(sinks),
                         sum_dq_e=total,
                         sum_abs_dq_e=np.abs(dq).sum(),
                         mean_dq_e=dq.mean(),
                         n_water=len(triplets)))

        # ---- Q2：逐水分子的 linear / nonlinear 分解 ----
        if len(triplets):
            pos = {int(i): j for j, i in enumerate(sinks)}
            ok = np.array([all(int(x) in pos for x in tri) for tri in triplets])
            tri = triplets[ok]
            ii = np.array([[pos[int(x)] for x in row] for row in tri])
            dq_tri = dq[ii].sum(axis=1)                        # 精确（含全部非线性）
            lin_tri = (A[tri] * t[ii]).sum(axis=1)             # 一阶（clamp 前）
            # 近金属分组
            if len(metals):
                d2m = np.min(np.linalg.norm(
                    coords[tri[:, 0]][:, None, :] - coords[metals][None, :, :], axis=2), axis=1)
            else:
                d2m = np.full(len(tri), np.inf)
            near = d2m <= a.near_metal_nm
            rows[-1]["min_water_metal_nm"] = float(np.min(d2m)) if len(metals) else np.nan
            rows[-1]["n_water_near_metal"] = int(near.sum())
            for label, sel in (("bulk", ~near), ("near_metal", near)):
                if not sel.any():
                    continue
                wrows.append(dict(
                    frame=int(fi), group=label, n=int(sel.sum()),
                    exact_mean_e=dq_tri[sel].mean(),
                    linear_mean_e=lin_tri[sel].mean(),
                    nonlinear_mean_e=(dq_tri[sel] - lin_tri[sel]).mean(),
                    linear_share=(lin_tri[sel].mean() / dq_tri[sel].mean()
                                  if dq_tri[sel].mean() else np.nan)))

    tot = pd.DataFrame(rows)
    wat = pd.DataFrame(wrows)

    log("=" * 78)
    log("Q1  体系总净电荷（不抽样，全体 dens_sink）—— PLAN §13.4b 的阻塞项 H1")
    log("=" * 78)
    log(tot.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    log("\n判读：|Σ Δq| 若到 e 量级，倒空间求和的中性背景假设就站不住，"
        "\n      路线 B 必须先处理（或在 Q 上加分子级中性约束，注意约束也进 λ_i）。")

    if len(wat):
        log("")
        log("=" * 78)
        log("Q2  水净 Δq 的一阶/三阶分解 —— 报告 §3.4.3 / §9.5-1 的归因")
        log("=" * 78)
        log(wat.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        log("\n判读：linear_share≈1 ⇒ 一阶 dens mismatch 主导（我推的三阶只是配角）；"
            "\n      ≈0 ⇒ 外层 tanh 曲率主导。近金属组反号则说明一阶确实随环境翻。")

    for fn, df in ((out_total, tot), (out_water, wat)):
        if len(df):
            df.to_csv(fn, index=False)
            log(f"-> {os.path.basename(fn)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
