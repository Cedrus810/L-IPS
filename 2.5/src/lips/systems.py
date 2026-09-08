#!/usr/bin/env python
"""体系加载的单一入口 —— 分析脚本共用，不跑任何 MD。

**为什么存在**：`measure_szz.py`、`closure_force_audit.py`、`zn_coordination.py`、
`zn_deltaq_probe.py` 原先各自硬编码 `build_1ckk_system()`、各自抄一遍
"读预建体系 → 还原 HID/HIE → 规范化离子残基名"。抄漏一步的代价是真金白银的：
`run_zn_job.py` 里的注释记着，漏调 `normalize_ion_resnames()` 会让 Zn 的
`dens_source` 静默变 0，**CWLD 机制等于整个关掉**，一整批轨迹作废。这种步骤
只能有一份实现。

**体系是参数，不是常量**。1CKK（钙调蛋白）已经不是当前研究体系，当前是 1AAY
锌指；任何写死某个 PDB 的分析代码迟早会在新体系上静默空转——`closure_force_audit`
的 `ca_idx = where(sp == "Ca2+")` 就已经这样过一次（1AAY 没有钙，分壳列直接消失
且不报错）。

用法：

    from lips import systems as ls
    top, system = ls.load_reference_system(name="1aay")
    top, system = ls.load_reference_system(topology="x.pdb", system_xml="x.xml")
    ls.add_system_arguments(parser)                 # --system / --topology / --system-xml
    top, system = ls.load_reference_system_from_args(args)
"""
from __future__ import annotations

import json
import os

import numpy as np

from lips import paths


#: 二价金属物种名（**规范化之后**的 resname）。加新金属只改这里。
DIVALENT_METAL_SPECIES = ("Ca2+", "Zn2+", "Mg2+")

#: 当前正在研究的体系。1CKK 已经不是了（见模块文档），保留它只为复算旧数据。
DEFAULT_SYSTEM = "1aay"


def load_v26_module():
    """拿到 CWLD 引擎模块。

    打包之前这里得用 `spec_from_file_location` 按路径加载
    `test_lips_vs_pmeV2.6.py`——文件名不是合法模块名，五个脚本各写一遍这段。
    现在它就是 `lips.engine.v26`，普通 import 即可。函数保留是为了兼容调用方。
    """
    from lips.engine import v26
    return v26


def restore_his_protonation(topology, meta_json_path, verbose=True):
    """把 app.PDBFile 读盘时塌成 HIS 的残基名还原成 HID/HIE，返回还原个数。

    `app.PDBFile` 读盘时用残基名替换表把 HID/HIE/HIP **全部塌成 HIS**（写盘是
    保留的，丢失发生在读这一侧）。名字一塌就无法判断哪个氮是去质子的那个
    （HID 是 NE2、HIE 是 ND1），所有按残基名做的逻辑都会看走眼。System 本身
    来自 system.xml，力场参数是对的，坏掉的只有名字。
    """
    if not os.path.exists(meta_json_path):
        if verbose:
            print(f"  ⚠ 没有 {meta_json_path}，HIS 质子化态无法还原")
        return 0
    with open(meta_json_path) as fh:
        meta = json.load(fh)
    hid = set(meta.get("hid_resids", []))
    hie = set(meta.get("hie_resids", []))
    n = 0
    for residue in topology.residues():
        if residue.name == "HIS" and residue.id:
            rid = int(residue.id)
            if rid in hid:
                residue.name = "HID"
                n += 1
            elif rid in hie:
                residue.name = "HIE"
                n += 1
    if verbose and n == 0 and (hid or hie):
        print(f"  ⚠ 一个都没还原 —— 检查 {meta_json_path} 的 hid_resids/hie_resids")
    return n


def _prebuilt_loader(subdir):
    """从 <PROJECT_ROOT>/<subdir>/ 读 solvated.pdb + system.xml + meta.json。"""
    def load(verbose=True, with_positions=False):
        import openmm as mm
        import openmm.app as app
        directory = paths.system_dir(subdir)
        pdb_path = os.path.join(directory, "solvated.pdb")
        xml_path = os.path.join(directory, "system.xml")
        if not (os.path.exists(pdb_path) and os.path.exists(xml_path)):
            raise SystemExit(
                f"缺预建体系 {directory}/（需要 solvated.pdb + system.xml）。"
                f"先跑 zinc_finger.py 之类的构建脚本。")
        pdb = app.PDBFile(pdb_path)
        topology = pdb.topology
        with open(xml_path) as fh:
            system = mm.XmlSerializer.deserialize(fh.read())
        restore_his_protonation(topology, os.path.join(directory, "meta.json"), verbose)
        return (topology, system, pdb.positions) if with_positions else (topology, system)
    return load


def _v26_builder(func_name):
    """调 lips.engine.v26 里的 build_*_system()，现建（慢）。"""
    def load(verbose=True, with_positions=False):
        lips = load_v26_module()
        topology, system, positions = getattr(lips, func_name)()
        return (topology, system, positions) if with_positions else (topology, system)
    return load


#: name -> (说明, 取 (topology, System) 的 callable)。加新体系只加一行。
SYSTEM_BUILDERS = {
    "1aay": ("预建 1AAY/amber/（Cys2His2 锌指，当前研究体系）",
             _prebuilt_loader(os.path.join("1AAY", "amber"))),
    "1aay-amoeba": ("预建 1AAY/amoeba/",
                    _prebuilt_loader(os.path.join("1AAY", "amoeba"))),
    "1ckk": ("build_1ckk_system()（钙调蛋白，**已非当前体系**，仅复算旧数据用）",
             _v26_builder("build_1ckk_system")),
}


def add_system_arguments(parser, default=DEFAULT_SYSTEM):
    """给 argparse 挂上统一的体系选项。"""
    parser.add_argument("--system", default=default,
                        help=f"参考体系名（决定原子顺序与 qbase）。可选："
                             f"{', '.join(sorted(SYSTEM_BUILDERS))}。默认 {default}")
    parser.add_argument("--topology", default=None,
                        help="预建体系的 pdb（与 --system-xml 成对；优先于 --system）")
    parser.add_argument("--system-xml", default=None,
                        help="预建体系的 system.xml（与 --topology 成对）")
    return parser


def load_reference_system(name=None, topology=None, system_xml=None,
                          normalize_ions=True, verbose=True, with_positions=False):
    """返回 (topology, System)，或 with_positions=True 时 (topology, System, positions)。

    只取体系，不跑任何一步积分（`with_positions=True` 也只是把预建体系里的坐标
    一并交出来，给需要起始构型的驱动脚本用）。

    normalize_ions=True 时会调 v2.6 的 `normalize_ion_resnames()`。**默认开着是
    有原因的**：`build_phase_cwld_metadata()` 判的是规范化之后的名字（'Zn2+'），
    而 PDB 里是 'ZN'；不规范化则金属的 dens_source=0，CWLD 静默失效。
    """
    import openmm as mm
    import openmm.app as app

    if topology or system_xml:
        if not (topology and system_xml):
            raise SystemExit("--topology 和 --system-xml 必须成对给出")
        if verbose:
            print(f"[system] 读预建体系 {topology} + {system_xml}（不跑 MD）")
        pdb = app.PDBFile(topology)
        top, positions = pdb.topology, pdb.positions
        with open(system_xml) as fh:
            system = mm.XmlSerializer.deserialize(fh.read())
        restore_his_protonation(
            top, os.path.join(os.path.dirname(os.path.abspath(topology)), "meta.json"),
            verbose)
    else:
        name = name or DEFAULT_SYSTEM
        if name not in SYSTEM_BUILDERS:
            raise SystemExit(
                f"未知体系 {name!r}；可选 {sorted(SYSTEM_BUILDERS)}，"
                f"或用 --topology/--system-xml 指定任意预建体系")
        desc, loader = SYSTEM_BUILDERS[name]
        if verbose:
            print(f"[system] {name}: {desc} —— 只取拓扑与参数，不跑 MD")
        loaded = loader(verbose=verbose, with_positions=with_positions)
        top, system = loaded[0], loaded[1]
        positions = loaded[2] if with_positions else None

    if top.getNumAtoms() != system.getNumParticles():
        raise SystemExit(
            f"拓扑 {top.getNumAtoms()} 原子与 System {system.getNumParticles()} "
            f"粒子数不一致，逐原子参数会错位")

    if normalize_ions:
        load_v26_module().normalize_ion_resnames(top)

    if verbose:
        metals = [a for a in top.atoms() if a.residue.name in DIVALENT_METAL_SPECIES]
        print(f"[system] {top.getNumAtoms()} 原子，二价金属 {len(metals)} 个"
              + (f"（{sorted({a.residue.name for a in metals})}）" if metals else
                 f"（找过 {list(DIVALENT_METAL_SPECIES)}，一个都没有）"))
    return (top, system, positions) if with_positions else (top, system)


def assert_traj_matches_system(dcd_path, n_system_atoms, label=None):
    """轨迹原子数必须与参考体系一致，不一致直接抛（带上可操作的诊断）。

    为什么要拦：用别的体系的轨迹配现在的 System，qbase 会整体错位，
    **算出来是一条看着完全正常的假曲线**。`analysis/szz.py` 有同族的闸门
    （那边在自动发现的循环里，语义是 skip；这里只有一条轨迹，语义是 raise）。

    返回轨迹的原子数。
    """
    import mdtraj as md
    probe = md.open(dcd_path)
    try:
        n_traj = probe.read(1)[0].shape[1]
    finally:
        probe.close()
    if n_traj == n_system_atoms:
        return n_traj

    name = label or os.path.basename(dcd_path)
    hint = ("多半是别的体系的轨迹（如 ZN_water_*）碰巧匹配了文件名模板")
    if n_traj == 32818 and n_system_atoms == 32794:
        hint = (
            "这正是 1AAY/amber/solvated.pdb 被从 32818 覆盖成 32794 的后果："
            "**32818 的 amber system.xml 已经不存在了**，老的 PME/pme_q/AMOEBA "
            "轨迹没有匹配的 System 可用。\n"
            "  不能用现在的 qbase 硬算——会整体错位。改用 32794 的 CWLD 轨迹，"
            "例如 ZN_1aay_cwld_ligm0p15metal_zmm_seed0_traj.dcd；\n"
            "  静态力审计只需要坐标，且已验证 kernel 排序与系综无关"
            "（见 REPORT/STAGE1 §11.1 两个独立系综同序）。")
    raise SystemExit(
        f"原子数不符，拒绝运行：轨迹 {name} 有 {n_traj} 原子，"
        f"参考体系有 {n_system_atoms}。\n  {hint}")


def assert_metals_are_density_sources(topology, system, verbose=True, **meta_kwargs):
    """硬断言：所有二价金属都必须是 CWLD 的 density source。

    这是本项目被咬过两次的不变量。`build_phase_cwld_metadata()` 判的是**规范化
    之后**的残基名（'Zn2+'），而 PDB 里写的是 'ZN'；没规范化就会让金属的
    `dens_source=0`——CWLD 机制对金属完全失效，**而且全程静默**，轨迹照跑、
    结果照出，只是物理是错的。所以宁可在这里炸掉，也不要拿到一批看不出问题的
    废数据。
    """
    lips = load_v26_module()
    meta = lips.build_phase_cwld_metadata(system, topology, **meta_kwargs)
    bad = [f"{a.residue.name}{a.residue.id}:{a.name}"
           for a in topology.atoms()
           if a.element is not None and a.element.symbol in ("Zn", "Ca", "Mg")
           and meta["dens_source"][a.index] <= 0.0]
    if bad:
        raise RuntimeError(
            f"这些二价金属的 dens_source=0（不会作为密度源，CWLD 机制等于关掉）：{bad}。"
            f"多半是残基名没规范化：需要 'Zn2+' 这类 value，而不是 'ZN' 这类 key。")
    if verbose:
        n = sum(1 for a in topology.atoms()
                if a.element is not None and a.element.symbol in ("Zn", "Ca", "Mg"))
        print(f"[system] ✓ {n} 个二价金属全部 dens_source>0")
    return meta


def load_reference_system_from_args(args, normalize_ions=True, verbose=True):
    """add_system_arguments() 装好的那三个参数 -> (topology, System)。"""
    return load_reference_system(
        name=getattr(args, "system", None),
        topology=getattr(args, "topology", None),
        system_xml=getattr(args, "system_xml", None),
        normalize_ions=normalize_ions, verbose=verbose)


def base_charges(system):
    """从 System 的第一个 NonbondedForce 取 qbase（单位 e）。"""
    from openmm import NonbondedForce
    from openmm.unit import elementary_charge
    forces = [system.getForce(i) for i in range(system.getNumForces())
              if isinstance(system.getForce(i), NonbondedForce)]
    if not forces:
        raise SystemExit("System 里没有 NonbondedForce，拿不到 qbase")
    nbf = forces[0]
    return np.array([nbf.getParticleParameters(i)[0].value_in_unit(elementary_charge)
                     for i in range(system.getNumParticles())])


def divalent_metal_indices(topology):
    """返回二价金属原子的下标（要求已规范化残基名）。"""
    return np.array([a.index for a in topology.atoms()
                     if a.residue.name in DIVALENT_METAL_SPECIES], dtype=int)
