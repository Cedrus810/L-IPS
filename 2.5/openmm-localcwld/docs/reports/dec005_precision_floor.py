#!/usr/bin/env python
"""DEC-005 零点实验：测出 fp32 在真实 1AAY CWLD 体系上的固有噪声地板。

**为什么需要它**：LCWLD-100 的新 CUDA kernel 和它要对照的生产 `CustomGBForce`
**都是 fp32 GPU 代码**，没有一方是真值。光是求和顺序不同就能在 1e-4 量级产生差异，
与 kernel 写得对不对无关。不先量出这个地板，验收阈值就只能靠猜——猜紧了必然假失败
（DEC-005 初稿的总能量 rtol=1e-5 就是这么埋下的地雷），猜松了漏掉真 bug。

做法：同一帧、同一个 System、同一个 `CustomGBForce`，只改 OpenMM 的 `Precision`：

    double  ← 当近似真值
    mixed   ← 生产轨迹实际用的（run_zn_job.py 默认 --precision mixed）
    single  ← 新 kernel 按 DEC-004 要对齐的精度

顺带回答一个至今没人测过的问题：生产跑 mixed，DEC-004 要 kernel 走 single，差多少。

**本脚本不跑 MD。** 三次 `Context.getState()`，`Integrator.step()` 调用次数为零；
建 Context 只是为了取一次静态力和能量。

用法：

    python docs/reports/dec005_precision_floor.py
    python docs/reports/dec005_precision_floor.py --dpolar -0.15 --scope metal

输出落盘到 `docs/reports/dec005_precision_floor.json`（**必须落盘**——报告 §4.1 那两个
Δq 就是毁在只在屏幕上出现过）。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "dec005_precision_floor.json"

# 力的量纲 kJ/mol/nm；能量 kJ/mol。
PRECISIONS = ("double", "mixed", "single")


def log(msg):
    print(msg, flush=True)


def build_cwld_system(dpolar, scope):
    """真实 1AAY + CWLD 注入。体系通过 lips.systems，不绕过离子名归一化。

    dpolar/scope 走环境变量，因为 v26 是在 **import 时**读它们决定
    LIGAND_DPOLAR_OVERRIDE 的——必须在导入之前设好。
    """
    if dpolar is not None:
        os.environ["L_IPS_LIGAND_DPOLAR"] = str(dpolar)
        os.environ["L_IPS_LIGAND_SCOPE"] = scope
    else:
        os.environ.pop("L_IPS_LIGAND_DPOLAR", None)

    from lips import systems as ls
    v26 = ls.load_v26_module()

    topology, system, positions = ls.load_reference_system(
        name="1aay", with_positions=True, verbose=True)
    # 硬断言：金属必须是密度源。漏掉离子名归一化会让 CWLD 静默关闭，
    # 那样测出来的"噪声地板"是另一个物理体系的。
    ls.assert_metals_are_density_sources(topology, system, verbose=True)

    meta = v26.build_phase_cwld_metadata(system, topology)
    cwld_system = v26.setup_cwld_lips_system(system, topology)
    return topology, cwld_system, positions, v26, meta


def static_state(system, positions, box, precision):
    """建 Context、取一次力和能量、销毁。零 Integrator.step()。"""
    import openmm as mm
    import openmm.unit as unit

    platform_cuda = mm.Platform.getPlatformByName("CUDA")
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = mm.Context(system, integrator, platform_cuda, {"Precision": precision})
    context.setPeriodicBoxVectors(*box)
    context.setPositions(positions)
    state = context.getState(getForces=True, getEnergy=True)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del context, integrator
    return np.asarray(forces, dtype=np.float64), float(energy)


def compare(reference, other, label):
    """逐原子力偏差，按分位数报。

    只报 max 是没用的：单个落在 cutoff 边界上的原子就能把 max 拉爆，而那类离散跳变
    由 DEC-005 第 4 节单独立规则处理，不该混进这里的统计。
    """
    diff = np.linalg.norm(other - reference, axis=1)
    mag = np.linalg.norm(reference, axis=1)
    # 相对偏差只在力本身不接近零时才有意义；用中位数力作为尺度下限。
    scale = np.maximum(mag, np.median(mag))
    rel = diff / scale
    q = lambda a, p: float(np.percentile(a, p))
    return {
        "label": label,
        "abs_p50": q(diff, 50), "abs_p99": q(diff, 99), "abs_max": float(diff.max()),
        "rel_p50": q(rel, 50), "rel_p99": q(rel, 99), "rel_max": float(rel.max()),
        "ref_force_median": float(np.median(mag)),
        "worst_atom": int(np.argmax(diff)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dpolar", type=float, default=None,
                    help="L_IPS_LIGAND_DPOLAR；不给则用默认元素规则（覆盖关）")
    ap.add_argument("--scope", default="metal", choices=["metal", "all"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    import openmm as mm

    names = [mm.Platform.getPlatform(i).getName()
             for i in range(mm.Platform.getNumPlatforms())]
    if "CUDA" not in names:
        raise SystemExit(f"需要 CUDA platform，当前只有 {names}")

    log("=" * 72)
    log("DEC-005 零点实验：fp32 噪声地板（纯静态，零 Integrator.step()）")
    log("=" * 72)
    topology, system, positions, v26, meta = build_cwld_system(args.dpolar, args.scope)
    box = system.getDefaultPeriodicBoxVectors()
    n_atoms = system.getNumParticles()
    log(f"[sys] {n_atoms} 原子，覆盖 dpolar={args.dpolar} scope={args.scope}")

    results, forces = {}, {}
    for precision in PRECISIONS:
        f, e = static_state(system, positions, box, precision)
        forces[precision] = f
        results[precision] = {"energy_kJ_per_mol": e,
                              "force_rms": float(np.sqrt((f ** 2).sum(axis=1).mean()))}
        log(f"[{precision:6s}] E = {e:+.6f} kJ/mol   |F|_rms = "
            f"{results[precision]['force_rms']:.4f} kJ/mol/nm")

    comparisons = [
        compare(forces["double"], forces["mixed"], "mixed_vs_double"),
        compare(forces["double"], forces["single"], "single_vs_double"),
        compare(forces["mixed"], forces["single"], "single_vs_mixed"),
    ]
    log("")
    log(f"{'对照':>18s} {'rel_p50':>10s} {'rel_p99':>10s} {'rel_max':>10s} {'abs_p99':>10s}")
    for c in comparisons:
        log(f"{c['label']:>18s} {c['rel_p50']:10.3e} {c['rel_p99']:10.3e} "
            f"{c['rel_max']:10.3e} {c['abs_p99']:10.3e}")

    floor = next(c for c in comparisons if c["label"] == "single_vs_double")
    log("")
    log("→ fp32 固有噪声地板（single vs double, rel_p99）: "
        f"{floor['rel_p99']:.3e}")
    log(f"→ DEC-005 建议阈值 = 地板 × 3~5 = "
        f"{floor['rel_p99']*3:.3e} ~ {floor['rel_p99']*5:.3e}")
    log("  ⚠ 这只是力的阈值依据。dens / Q 是中间量，CustomGBForce 取不出来，")
    log("    它们的阈值要等 LCWLD-100 的 kernel 能单独 dump 时再按同样方法定。")

    # 逐原子力落盘。初版只存了聚合量，结果第二天想算"信号有多大"就得重跑 GPU
    # —— 正是 §4.1 那两个 Δq 栽的同一个坑。npz 因为 32794x3 的 float64 塞进 json
    # 既大又慢。
    npz_path = str(Path(args.out).with_suffix(".npz"))
    override = np.asarray(meta["ligand_override_indices"], dtype=int)
    atoms = list(topology.atoms())
    np.savez_compressed(
        npz_path,
        force_double=forces["double"].astype(np.float64),
        force_single=forces["single"].astype(np.float64),
        force_mixed=forces["mixed"].astype(np.float64),
        override_indices=override,
        override_labels=np.array(
            [f"{atoms[i].residue.name}{atoms[i].residue.id}:{atoms[i].name}" for i in override]),
    )
    log(f"-> {npz_path}  （逐原子力 + {len(override)} 个覆盖原子的下标/标签）")

    payload = {
        "purpose": "DEC-005 零点实验：fp32 噪声地板",
        "per_atom_forces_npz": os.path.basename(npz_path),
        "n_override_atoms": int(override.size),
        "override_labels": [f"{atoms[i].residue.name}{atoms[i].residue.id}:{atoms[i].name}"
                            for i in override],
        "system": "1AAY + CWLD (production CustomGBForce)",
        "n_atoms": n_atoms,
        "ligand_dpolar": args.dpolar,
        "ligand_scope": args.scope,
        "integrator_steps": 0,
        "per_precision": results,
        "comparisons": comparisons,
        "suggested_force_rtol_range": [floor["rel_p99"] * 3, floor["rel_p99"] * 5],
        "env": {
            "python": platform.python_version(),
            "openmm": mm.version.version,
            "numpy": np.__version__,
            "gpu": mm.Platform.getPlatformByName("CUDA").getPropertyDefaultValue("DeviceName")
                   if "DeviceName" in mm.Platform.getPlatformByName("CUDA").getPropertyNames()
                   else "unknown",
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
