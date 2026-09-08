#!/usr/bin/env python
"""导出 DEC-005 验收所需的一切，用 **OpenMM 自己的序列化**，供 C++ 侧读取。

**为什么要 dump**：`LocalCWLDForce` 还没有 Python 绑定（LCWLD-080/SWIG 未做，env 里
也没装 SWIG），所以对照只能在 C++ 侧做；而体系、metadata 和参考力只有 Python 拿得到。
dump 是这条缝。它也让验收不必等 LCWLD-080。

**为什么不自己定文件格式**：初版写了个 ASCII 表。那和"手搓一份参数映射"是同一类错，
而且失效模式更阴——列序、单位、精度可以静默漂移，看起来全对。改用 `XmlSerializer`：
传输链上没有一行我们自己写的解析代码，**OpenMM 是格式的唯一权威**。

产物（每档配置两个精度）：

    <tag>_system.xml     整个 CWLD System。里面那个 CustomGBForce 自带九个逐粒子参数、
                         参数名、以及全部 exclusions —— C++ 侧按**名字**取，不按下标，
                         这样以后谁在 v26 里插一个 addPerParticleParameter，列也不会静默错位。
    <tag>_state_<p>.xml  positions / box / **隔离力组的**参考力 / 能量
    <tag>_provenance.json  力组号、精度、代码版本、L_IPS_* 环境变量实际取值、对照分组

⚠ **力组隔离是必须的，不是优化。** `state` 里的 forces 默认是**全体系合力**，
而我们要比的是 CustomGBForce 一个力。更麻烦的是 `assign_cwld_mts_force_groups()`
（`v26.py:495`）只在 MTS 打开时才被调用，而 `USE_MTS=False` 是冻结结论 ——
所以这里必须**自己显式** setForceGroup 再取 state。不做这一步，比的就是
"LocalCWLDForce 单个力 vs 全体系合力"，差几个数量级，还会被误读成 kernel 全错。

⚠ **double 那一份只能现在取。** 偏置 t 检验的基线需要 fp64 参考，事后补不了。

**不跑 MD。** 建一次体系 + 每个精度一次静态 `getState()`，`Integrator.step()` 为零。

用法：

    python docs/reports/dec005_dump_1aay.py                  # 覆盖关
    python docs/reports/dec005_dump_1aay.py --dpolar -0.15   # 覆盖开（13 个原子）
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

#: 距最近 Zn 13.13 Å，命中化学判据但不配位。DEC-005 B 组：单列，不并进统计。
BULK_PROBE = "HIE149:ND1"

#: 给 CustomGBForce 用的隔离力组。31 是 OpenMM 允许的最大组号，不会跟别的力撞。
ISOLATED_GROUP = 31

PRECISIONS = ("mixed", "double")


def log(m):
    print(m, flush=True)


def code_version():
    try:
        import lips
        version = getattr(lips, "__version__", "unknown")
    except Exception:
        version = "unknown"
    try:
        rev = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        rev = ""
    return {"lips": version, "git": rev or "not-under-version-control"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dpolar", type=float, default=None)
    ap.add_argument("--scope", default="metal", choices=["metal", "all"])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    # v26 在 import 时读这些环境变量决定 LIGAND_DPOLAR_OVERRIDE，必须先设。
    if args.dpolar is not None:
        os.environ["L_IPS_LIGAND_DPOLAR"] = str(args.dpolar)
        os.environ["L_IPS_LIGAND_SCOPE"] = args.scope
    else:
        os.environ.pop("L_IPS_LIGAND_DPOLAR", None)

    import openmm as mm
    import openmm.unit as unit
    from lips import systems as ls

    tag = args.tag or f"dec005_1aay_{'dpolar' if args.dpolar is not None else 'off'}"

    v26 = ls.load_v26_module()
    topology, system, positions = ls.load_reference_system(
        name="1aay", with_positions=True, verbose=True)
    # 漏掉离子名规范化会让 Zn 静默不是密度源 —— 那样 dump 的是另一个物理体系。
    ls.assert_metals_are_density_sources(topology, system, verbose=True)
    meta = v26.build_phase_cwld_metadata(system, topology)
    cwld = v26.setup_cwld_lips_system(system, topology)

    gb = None
    for i in range(cwld.getNumForces()):
        f = cwld.getForce(i)
        if isinstance(f, mm.CustomGBForce):
            f.setForceGroup(ISOLATED_GROUP)
            gb = f
        else:
            f.setForceGroup(0)
    if gb is None:
        raise SystemExit("CWLD 体系里没有 CustomGBForce，取不到参考力")
    names = [gb.getPerParticleParameterName(i)
             for i in range(gb.getNumPerParticleParameters())]
    log(f"[gb  ] CustomGBForce 力组 {ISOLATED_GROUP}，逐粒子参数 {names}")

    (HERE / f"{tag}_system.xml").write_text(mm.XmlSerializer.serialize(cwld))

    energies = {}
    for precision in PRECISIONS:
        integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
        context = mm.Context(cwld, integrator, mm.Platform.getPlatformByName("CUDA"),
                             {"Precision": precision})
        context.setPositions(positions)
        state = context.getState(getPositions=True, getForces=True, getEnergy=True,
                                 groups={ISOLATED_GROUP})
        (HERE / f"{tag}_state_{precision}.xml").write_text(
            mm.XmlSerializer.serialize(state))
        e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        f = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer)
        energies[precision] = e
        log(f"[{precision:6s}] E = {e:+.6f} kJ/mol   "
            f"|F|_rms = {np.sqrt((np.asarray(f)**2).sum(axis=1).mean()):.4f}")
        del context, integrator

    atoms = list(topology.atoms())
    label = lambda i: f"{atoms[i].residue.name}{atoms[i].residue.id}:{atoms[i].name}"
    override = [int(i) for i in meta["ligand_override_indices"]]
    water_o = [int(i) for i in meta["water_o_indices"]]
    water_h = [a.index for a in atoms
               if a.residue.name in ("HOH", "WAT", "TP3", "SOL")
               and a.element is not None and a.element.symbol == "H"]
    rng = np.random.default_rng(0)
    pick = lambda a, n: sorted(int(x) for x in
                               (a if len(a) <= n else rng.choice(a, n, replace=False)))

    provenance = {
        "purpose": "DEC-005 acceptance reference for LocalCWLDForce",
        "system": "1AAY + CWLD (production CustomGBForce)",
        "n_atoms": cwld.getNumParticles(),
        "isolated_force_group": ISOLATED_GROUP,
        "force_group_note": ("state forces are the ISOLATED CustomGBForce only. "
                             "v26's assign_cwld_mts_force_groups() runs only with "
                             "USE_MTS=True (frozen False), so this script sets the "
                             "group itself."),
        "precisions": list(PRECISIONS),
        "energies_kJ_per_mol": energies,
        "per_particle_parameter_names": names,
        "integrator_steps": 0,
        "env": {k: os.environ.get(k) for k in
                ("L_IPS_LIGAND_DPOLAR", "L_IPS_LIGAND_SCOPE", "L_IPS_CLOSURE")},
        "code_version": code_version(),
        "python": platform.python_version(),
        "openmm": mm.version.version,
        "groups": {
            "A_coordinating": sorted(i for i in override if label(i) != BULK_PROBE),
            "B_bulk_probe": sorted(i for i in override if label(i) == BULK_PROBE),
            "C_metals": [int(i) for i in ls.divalent_metal_indices(topology)],
            "D_water_O": pick(water_o, 200),
            "D_water_H": pick(water_h, 200),
        },
        "labels": {},
    }
    for g in provenance["groups"].values():
        for i in g:
            provenance["labels"][str(i)] = label(i)
    (HERE / f"{tag}_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False))

    # 分组单独出一份最简形式：`name count i1 i2 ...`，一行一组。
    # 这不是"自定义数据格式"——里面只有整数下标，没有单位、没有浮点、没有列序，
    # 也就没有那三类静默漂移。真正的数据（参数/坐标/力）一律走 XmlSerializer。
    with open(HERE / f"{tag}_groups.txt", "w") as fh:
        fh.write("# DEC-005 comparison groups: name count indices...\n")
        for name, idx in provenance["groups"].items():
            fh.write(f"{name} {len(idx)} " + " ".join(str(i) for i in idx) + "\n")

    log(f"[dump] A={len(provenance['groups']['A_coordinating'])} "
        f"B={len(provenance['groups']['B_bulk_probe'])} "
        f"C={len(provenance['groups']['C_metals'])}")
    for suffix in ("system.xml", "state_mixed.xml", "state_double.xml",
                   "provenance.json", "groups.txt"):
        p = HERE / f"{tag}_{suffix}"
        log(f"  -> {p.name}  ({p.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
