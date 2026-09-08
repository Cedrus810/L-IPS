#!/usr/bin/env python
"""DEC-005 信号窗口：算出"差多少才算错"，而不只是"噪声有多大"。

**为什么需要它**：DEC-005 第一版把力的阈值定成"实测精度地板 × 4.5"。peer `2-5-9c`
指出那个构造只回答了"同算法不同精度差多少"，没回答"差多少才**算错**"。而新 CUDA
kernel 是**不同算法**（融合遍历、不同邻居次序），它的偏差本来就应该显著大于精度地板
——于是很可能出现 kernel 完全正确、偏差却超阈值、验收判红，然后只剩"推翻决策文件"
这一条路。跟总能量 rtol=1e-5 那颗地雷是同一个陷阱，只是换了一栏。

正确的判据是对**信号**定的：

    精度地板  ≤  验收阈值  ≪  我们要捕捉的物理效应本身

而这个"物理效应"手上就有：`dec005_precision_floor.py` 已经跑了 override 关 / 开 两档。
两档在覆盖原子上的力之差，就是 `dpolar=-0.15` 带来的力变化——正是把 PME 的 66.9%
顶到 100% 的那个东西。

**验的是谁**：生产 `CustomGBForce`，**不是**分析侧那份镜像 `_local_dens_at`。
两者共用同一套数学，但不是同一份实现。REPORT §4.1 的三个 Δq 数字出自
`_local_dens_at`，与本文件的 ΔF **串不起来**——这条结果不能顶替 A 工单
（`_local_dens_at` 对 CustomGBForce 的 parity）。

纯离线，读两个 npz，不上 GPU、不跑 MD。

    python docs/reports/dec005_signal_window.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

#: 距最近 Zn 13.13 Å，命中化学判据但**不配位**。它按设计就该没有信号，
#: 所以必须从"最弱信号"的统计里剔除，否则会把阈值压到零。
#: 同时它是最好的假阳性探针：见 report 中的断言。
BULK_PROBE = "HIE149:ND1"


def main():
    off = np.load(HERE / "dec005_precision_floor.npz", allow_pickle=True)
    on = np.load(HERE / "dec005_precision_floor_dpolar.npz", allow_pickle=True)
    floor = json.loads((HERE / "dec005_precision_floor.json").read_text())

    idx = np.asarray(on["override_indices"], dtype=int)
    labels = [str(x) for x in on["override_labels"]]
    # double 精度作差，把精度噪声从信号里排除干净。
    delta = np.linalg.norm(on["force_double"] - off["force_double"], axis=1)

    rows = sorted(zip(labels, idx), key=lambda t: -delta[t[1]])
    print(f"{'原子':<16s} {'|ΔF| kJ/mol/nm':>16s} {'|F| kJ/mol/nm':>15s}")
    for label, i in rows:
        print(f"{label:<16s} {delta[i]:16.3f} "
              f"{np.linalg.norm(off['force_double'][i]):15.1f}")

    coordinating = np.array([i for label, i in zip(labels, idx) if label != BULK_PROBE])
    probe = [i for label, i in zip(labels, idx) if label == BULK_PROBE]

    floor_abs = next(c for c in floor["comparisons"]
                     if c["label"] == "single_vs_double")["abs_p99"]
    weakest = float(delta[coordinating].min())

    print()
    print(f"  真配位原子 {coordinating.size} 个: "
          f"ΔF 最小 {weakest:.1f}  中位 {np.median(delta[coordinating]):.1f}  "
          f"最大 {delta[coordinating].max():.1f}")
    if probe:
        print(f"  {BULK_PROBE} (体相探针): ΔF = {delta[probe[0]]:.4f}  "
              f"—— 比最弱真信号小 {weakest / max(delta[probe[0]], 1e-12):.0f} 倍")
    print(f"  全体原子: p50 {np.percentile(delta, 50):.4f}  "
          f"p99 {np.percentile(delta, 99):.2f}")
    print()
    print(f"  精度地板 abs_p99 = {floor_abs:.4f} kJ/mol/nm")
    print(f"  最弱真信号        = {weakest:.1f} kJ/mol/nm")
    print(f"  可用窗口          = {weakest / floor_abs:.0f}x "
          f"({np.log10(weakest / floor_abs):.1f} 个数量级)")

    # 阈值取几何中位偏低处：远高于地板（给"不同算法"留余量），远低于最弱信号
    # （小到不足以改变结论）。
    suggested = 1.0
    print()
    print(f"  → 建议力阈值 atol = {suggested:g} kJ/mol/nm")
    print(f"      = 地板的 {suggested / floor_abs:.1f} 倍（不同算法的余量）")
    print(f"      = 最弱真信号的 1/{weakest / suggested:.0f}（小到不足以改变结论）")

    # --- 偏置基线（DEC-005 3.7）------------------------------------------
    # 幅度判据抓不到单边偏置。漏/重复 tile 的误差符号恒定，精度噪声符号随机，
    # 所以带符号均值的 t 值能把两者分开。基线必须实测：定点累加的 (long long)
    # 强制转换是向零截断，所以纯精度噪声本身就有非零均值。
    refF, tstF = off["force_double"], off["force_single"]
    dF = tstF - refF
    mag = np.linalg.norm(refF, axis=1)
    proj = np.einsum("ij,ij->i", dF, refF / np.maximum(mag, 1e-12)[:, None])
    nAtoms = len(proj)
    se = proj.std(ddof=1) / np.sqrt(nAtoms)
    tProj = proj.mean() / se
    tComp = [float(dF[:, k].mean() / (dF[:, k].std(ddof=1) / np.sqrt(nAtoms)))
             for k in range(3)]
    print()
    print(f"  偏置基线（纯精度噪声 single vs double, N={nAtoms}）")
    print(f"    mean(ΔF·F̂_ref) = {proj.mean():+.3e}   t = {tProj:+.2f}"
          f"   {'← 显著非零，基线必须扣掉' if abs(tProj) >= 3 else ''}")
    print(f"    逐分量 t = [{tComp[0]:+.2f}, {tComp[1]:+.2f}, {tComp[2]:+.2f}]  （与零相容）")
    print(f"    灵敏度 3·SE = {3*se:.3e} kJ/mol/nm"
          f"  = atol 的 1/{suggested/(3*se):.0f}")

    payload = {
        "purpose": "DEC-005 信号窗口：阈值对着信号定，不只对着噪声定",
        "bias_baseline": {
            "n_atoms": int(nAtoms),
            "mean_proj_on_ref_force": float(proj.mean()),
            "t_proj": float(tProj),
            "t_per_component": tComp,
            "three_sigma_sensitivity": float(3 * se),
            "note": ("t_proj 显著非零是 realToFixedPoint 的 (long long) 向零截断造成的，"
                     "不是 bug；验收判据必须用 |t - t_proj_baseline| < 3，"
                     "逐分量基线约为 0 可直接用 |t| < 3。"),
        },
        "source": ["dec005_precision_floor.npz", "dec005_precision_floor_dpolar.npz"],
        "signal_definition": "|F(dpolar=-0.15) - F(override off)|, double precision",
        "per_atom": {label: float(delta[i]) for label, i in zip(labels, idx)},
        "bulk_probe": BULK_PROBE,
        "bulk_probe_delta": float(delta[probe[0]]) if probe else None,
        "coordinating_count": int(coordinating.size),
        "weakest_real_signal": weakest,
        "median_real_signal": float(np.median(delta[coordinating])),
        "precision_floor_abs_p99": floor_abs,
        "window_ratio": weakest / floor_abs,
        "suggested_force_atol_kJ_per_mol_per_nm": suggested,
    }
    out = HERE / "dec005_signal_window.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
