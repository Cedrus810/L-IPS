#!/usr/bin/env python
"""Zn²⁺ 纯水盒子 —— 双侧判据的另一半。

判据设计（PMC3383645 + 1AAY 那半边，见 zinc_finger.py 的文档）：

    体系                  正确 CN        固定 +2      CWLD      AMOEBA
    Zn²⁺ 纯水            6（八面体）     ~6 ✓         ?         6 ✓
    Zn²⁺ Cys₂His₂        4（四面体）     ~6 ✗         ?         4 ✓

**同一个固定电荷不可能同时满足两边** —— 这就是极化的论点。
所以 CN=4 本身不是判据，**两个体系之间的 CN 反差**才是。
一个为了在蛋白位点拿到 4 而调小 Zn 电荷的固定电荷模型，会在纯水里给出偏低的 CN。

实验/文献参照（bulk Zn²⁺ 水合）：CN = 6，八面体，Zn–O ≈ 2.08–2.10 Å。

好处：盒子小、跑得飞快 ⇒ 可以多 seed 长轨迹，1CKK 那个"效应被噪声吞掉"的问题自动消失；
而且离子水合自由能就是 ABFE 里的 decharging，直接在目标路径上。

用法：
    python zn_water.py                    # Amber 固定电荷，Zn=+2
    python zn_water.py --zn-charge 1.0    # 电荷转移诊断（配衡离子随之调整）
    python zn_water.py --ff amoeba        # AMOEBA 极化基线
"""
from __future__ import annotations

import argparse
import numpy as np

ZN_O_CUTOFF_A = 2.8          # bulk Zn-O 第一壳（Zn-O ~2.09 Å）


def log(m): print(m, flush=True)


def build_zn_water_system(box_nm=3.5, q_zn=2.0, ff_kind="amber",
                          cutoff_nm=1.0, verbose=True):
    """返回 (topology, system, positions)。单个 Zn²⁺ + 水 + 配衡 Cl⁻。"""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from openmm.app import element

    # --- 只放一个 Zn 在盒子中心 ---
    top = app.Topology()
    ch = top.addChain()
    res = top.addResidue("ZN", ch)
    top.addAtom("ZN", element.zinc, res)
    c = box_nm / 2.0
    pos = [mm.Vec3(c, c, c)] * unit.nanometer
    top.setPeriodicBoxVectors([mm.Vec3(box_nm, 0, 0), mm.Vec3(0, box_nm, 0),
                               mm.Vec3(0, 0, box_nm)] * unit.nanometer)

    if ff_kind == "amber":
        ff = app.ForceField('amber19-all.xml', 'amber19/tip3p.xml')
    elif ff_kind == "amoeba":
        # AMOEBA 水是三点的，用 tip3p 几何摆水，amoeba2018 的 HOH 模板会匹配
        ff = app.ForceField('amoeba2018.xml')
    else:
        raise ValueError(f"未知 ff_kind={ff_kind!r}（amber / amoeba）")

    modeller = app.Modeller(top, pos)
    modeller.addSolvent(ff, model='tip3p',
                        boxSize=mm.Vec3(box_nm, box_nm, box_nm) * unit.nanometer,
                        neutralize=True)
    if verbose:
        n_by = {}
        for r in modeller.topology.residues():
            n_by[r.name] = n_by.get(r.name, 0) + 1
        log(f"  ff={ff_kind}  盒子 {box_nm} nm  原子 {modeller.topology.getNumAtoms()}")
        log(f"  残基组成: " + ", ".join(f"{k}×{v}" for k, v in sorted(n_by.items())))

    if ff_kind == "amoeba":
        # AMOEBA 水是柔性的，不加 rigidWater/HBonds；dt 必须 <=1 fs
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                 nonbondedCutoff=cutoff_nm * unit.nanometer,
                                 constraints=None, rigidWater=False,
                                 polarization='mutual',
                                 mutualInducedTargetEpsilon=1e-5)
    else:
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                 nonbondedCutoff=cutoff_nm * unit.nanometer,
                                 constraints=app.HBonds, rigidWater=True)

    if abs(q_zn - 2.0) > 1e-12:
        if ff_kind != "amber":
            raise ValueError("电荷转移诊断只对 amber 固定电荷有意义")
        _scale_zn_and_rebalance(system, modeller.topology, q_zn, verbose)

    if verbose:
        _report_charges(system, ff_kind)
        log(f"\n  参照（文献/实验，bulk Zn²⁺）: CN=6 八面体, Zn–O ≈ 2.08–2.10 Å")
        if ff_kind == "amoeba":
            log("  ⚠ AMOEBA: 柔性水、无约束 ⇒ dt 必须 ≤1 fs；比固定电荷慢 1–2 个量级。")
    return modeller.topology, system, modeller.positions


def _scale_zn_and_rebalance(system, topology, q_zn, verbose):
    """纯水里没有蛋白配体可以接收电荷，所以改 Zn 电荷后必须动配衡离子。

    只允许 q_zn 使净电荷仍为整数（例如 2.0/1.0）；1.5 这类会剩 ±0.5，
    在纯水里无法用整数离子中和 —— 直接报错，不静默留个非中性体系给 PME。
    """
    import openmm as mm
    from openmm import unit
    nb = [system.getForce(i) for i in range(system.getNumForces())
          if isinstance(system.getForce(i), mm.NonbondedForce)][0]
    atoms = list(topology.atoms())
    zn = [a.index for a in atoms if a.residue.name in ("ZN", "Zn", "Zn2+")]
    delta = 2.0 - q_zn
    resid = delta * len(zn)
    if abs(resid - round(resid)) > 1e-9:
        raise ValueError(
            f"q_zn={q_zn} 会留下 {resid:+.3f} e 的非整数净电荷，纯水盒子里没法用整数离子中和。"
            f"纯水这半边只能用使净电荷为整数的值（如 2.0 / 1.0）；"
            f"非整数的 q_zn 请只在 1AAY 上做（那里差额可以给蛋白配体，见 apply_charge_transfer）。")
    for zi in zn:
        q, s, e = nb.getParticleParameters(zi)
        nb.setParticleParameters(zi, q.value_in_unit(unit.elementary_charge) - delta, s, e)
    # 去掉 round(resid) 个 Cl- 的电荷（把它们变成"透明"的粒子会改 LJ，所以
    # 这里选择直接报出来由调用方决定；默认做法是减少配衡离子数量重建体系。
    cl = [a.index for a in atoms if a.residue.name in ("CL", "Cl", "Cl-")]
    need = int(round(resid))
    if need > 0:
        if len(cl) < need:
            raise RuntimeError(f"需要移除 {need} 个 Cl- 的电荷，但只有 {len(cl)} 个")
        for j in cl[:need]:
            q, s, e = nb.getParticleParameters(j)
            nb.setParticleParameters(j, 0.0, s, e)
        if verbose:
            log(f"  电荷转移(纯水): Zn +2.0 -> {q_zn:+.2f}；"
                f"把 {need} 个 Cl- 的电荷置零以维持中性")
            log(f"     ⚠ 这些 Cl- 仍保留 LJ ⇒ 相当于中性的类 Cl 粒子。"
                f"想更干净就用更少的配衡离子重建体系。")


def _report_charges(system, ff_kind):
    import openmm as mm
    from openmm import unit
    nbs = [system.getForce(i) for i in range(system.getNumForces())
           if isinstance(system.getForce(i), mm.NonbondedForce)]
    if nbs:
        q = np.array([nbs[0].getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                      for i in range(system.getNumParticles())])
        log(f"  Σq = {q.sum():+.6f} e（应 ≈0）, Σq² = {np.sum(q**2):.1f} e²")
    else:
        names = [type(system.getForce(i)).__name__ for i in range(system.getNumForces())]
        log(f"  无 NonbondedForce（{ff_kind} 走多极子）；力清单: {names}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=float, default=3.5)
    ap.add_argument("--zn-charge", type=float, default=2.0)
    ap.add_argument("--ff", default="amber", choices=["amber", "amoeba"])
    a = ap.parse_args()
    build_zn_water_system(box_nm=a.box, q_zn=a.zn_charge, ff_kind=a.ff)


if __name__ == "__main__":
    main()
