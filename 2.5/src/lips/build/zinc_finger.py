#!/usr/bin/env python
"""1AAY (Zif268 锌指) 体系构建 —— CWLD 的第二个测试体系。

为什么换到这里（1CKK 的诊断见 STAGE1_CLOSURE_KILLSWITCH_REPORT.md）：
1CKK 的 Ca²⁺ 被 EF-hand 螯合死（7.5 个配位氧里 6.6 个是蛋白、只有 0.9 个水），
局部密度几乎恒定 ⇒ Δq 恒定且只用掉 clamp 的 0.15–7% ⇒ CWLD 退化成固定电荷模型。
更要命的是 **PME 在 1CKK 上本来就对**（配位 7.66，与实验相符）——headroom 为零，
没有可赢的空间。

1AAY 换的是判据本身：
    固定电荷 Amber (PME)  —— **已知失败**，错成六配位八面体
    AMOEBA / Drude        —— 已知成功，四配位四面体
    CWLD                  —— 待测
（依据 PMC3383645, "Modeling Zinc Proteins with Polarizable Potentials"）

判据是**定性几何量**，不是会被噪声吞掉的 RDF 峰高：
  * CN(Zn) 按配体类型分解（Cys S / His N / 水 O）
  * 配体-Zn-配体角分布：四面体单峰 109.5° vs 八面体 90°+180° 双峰

晶体参照（本文件 1AAY.pdb 实测，纯几何，见 report_crystal_reference()）：
    三个位点全是 Cys₂His₂，平均角 109.1–109.4°（理想 109.47°），
    Zn–S 2.15–2.38 Å，Zn–N 1.94–2.08 Å，无任何接近 90/180° 的角对。

力场就绪度（已核对，不需要换力场）：
  * `amber19/tip3p.xml` 有 `<Residue name="ZN">`，`charge="2.0"` 的**纯非键 12-6 离子**
    —— 正是我们要的"已知会失败"的基线。**不要**换成 12-6-4：那等于先用一个 r⁻⁴ 项
    手工把过配位补掉，CWLD 就没东西可证明了。
  * `amber19/protein.ff19SB.xml` 有 CYM（净电荷 −1.0，SG −0.8844，无 HG）与 HID/HIE/HIP。
  * `hydrogens.xml` 的 CYS 条目**只声明 CYS/CYX 变体，没有 CYM**，HG 带 variant="CYS"。
    所以 His 可以走 `variants=['HID',...]`，Cys 必须"先加氢、再删 HG、再改名 CYM"。

用法：
    python zinc_finger.py                 # 建体系并打印全部诊断
    python zinc_finger.py --report-only   # 只报晶体参照，不建体系
"""
from __future__ import annotations

import argparse, itertools, os
import numpy as np

from lips import paths
PDB_PATH = paths.system_dir("1AAY", "1AAY.pdb")
ZN_LIGAND_CUTOFF_A = 2.8          # 第一壳判据（晶体里 Zn-S 最远 2.38 Å）


def log(m): print(m, flush=True)


# ---------------------------------------------------------------------------
def _pdb_atoms(path):
    out = []
    for l in open(path):
        if l.startswith(("ATOM", "HETATM")):
            out.append(dict(name=l[12:16].strip(), res=l[17:20].strip(), ch=l[21],
                            seq=int(l[22:26]), el=l[76:78].strip(),
                            xyz=np.array([float(l[30 + 8 * i:38 + 8 * i]) for i in range(3)])))
    return out


def detect_zn_ligands(path=PDB_PATH, cutoff=ZN_LIGAND_CUTOFF_A):
    """从坐标几何检测每个 Zn 的第一壳。**不写死残基号**，换结构也能用。"""
    A = _pdb_atoms(path)
    zns = [a for a in A if a["res"] in ("ZN", "Zn", "Zn2+")]
    sites = []
    for z in zns:
        shell = [(float(np.linalg.norm(a["xyz"] - z["xyz"])), a) for a in A
                 if a is not z and a["el"] in ("N", "O", "S")]
        shell = sorted((t for t in shell if t[0] < cutoff), key=lambda t: t[0])
        sites.append(dict(zn=z, shell=shell))
    return sites


def report_crystal_reference(path=PDB_PATH):
    """打印晶体参照行 —— 这是整张对照表的 ground truth。"""
    sites = detect_zn_ligands(path)
    log(f"晶体参照（{os.path.basename(path)}，纯几何，cutoff {ZN_LIGAND_CUTOFF_A} Å）")
    log(f"  Zn 数量: {len(sites)}")
    cys, hid, other_his = set(), set(), set()
    for i, s in enumerate(sites, 1):
        z = s["zn"]
        log(f"  ── Zn{i} (resSeq {z['seq']})  CN={len(s['shell'])}")
        for r, a in s["shell"]:
            log(f"       {a['res']}{a['seq']:<4d} {a['name']:<4s} {r:.3f} Å")
            if a["res"] == "CYS" and a["name"] == "SG":
                cys.add(a["seq"])
            elif a["res"] == "HIS" and a["el"] == "N":
                hid.add((a["seq"], a["name"]))
        if len(s["shell"]) >= 3:
            v = [(a["xyz"] - z["xyz"]) / r for r, a in s["shell"]]
            ang = [np.degrees(np.arccos(np.clip(np.dot(p, q), -1, 1)))
                   for p, q in itertools.combinations(v, 2)]
            log(f"       角度 {' '.join(f'{x:.1f}' for x in sorted(ang))}"
                f"   均值 {np.mean(ang):.1f}°  (四面体 109.47°)")
    all_his = sorted({a["seq"] for a in _pdb_atoms(path) if a["res"] == "HIS"})
    hid_seq = sorted({s for s, _ in hid})
    log(f"\n  质子化态指派：")
    log(f"    CYS -> CYM (硫醇盐, −1): {sorted(cys)}")
    log(f"    HIS -> HID (配位 NE2, H 在 ND1): {hid_seq}")
    log(f"    HIS -> HIE (不配位, 默认): {sorted(set(all_his) - set(hid_seq))}")
    log(f"  净电荷记账：6×CYM(−1) + 3×Zn(+2) = {len(cys)*-1 + len(sites)*2:+d}（Zn 位点自洽）")
    return sorted(cys), hid_seq, sorted(set(all_his) - set(hid_seq))


# ---------------------------------------------------------------------------
def apply_charge_transfer(system, topology, positions, q_zn, verbose=True):
    """把 Zn 的电荷从 +2 挪到 q_zn，差额均分给它的 4 个第一壳配体。

    **为什么不是"直接缩 Zn 电荷"**：那会让净电荷变成 3*(q_zn-2)，q_zn=1.5 时是 -1.5，
    非整数，没法用整数配衡离子中和 PME。而物理上正确的图像本来就是**电荷转移**：
    硫醇盐把电子给 Zn，Zn 从 +2 降下来、给电子的配体相应变正，**位点总电荷不变**。
    这样体系严格中性，而且这正是 CWLD 在做的事（局部重分布），所以这个扫描
    直接标定了"要修好配位需要多大的 Δq"——即 CWLD 的 clamp 够不够。

    注意：只改电荷、不改 LJ。这是**诊断模型**，用来量 headroom，不是生产力场。
    """
    import openmm as mm
    from openmm import unit
    nb = [system.getForce(i) for i in range(system.getNumForces())
          if isinstance(system.getForce(i), mm.NonbondedForce)][0]
    atoms = list(topology.atoms())
    xyz = np.array(positions.value_in_unit(unit.nanometer)) * 10.0
    zn_idx = [a.index for a in atoms if a.residue.name in ("ZN", "Zn", "Zn2+")]
    delta = 2.0 - q_zn
    moved = 0
    for zi in zn_idx:
        d = np.linalg.norm(xyz - xyz[zi], axis=1)
        lig = [j for j in np.argsort(d)[1:12]
               if d[j] < ZN_LIGAND_CUTOFF_A and atoms[j].element is not None
               and atoms[j].element.symbol in ("N", "O", "S")]
        if len(lig) != 4:
            raise RuntimeError(f"Zn(idx {zi}) 第一壳有 {len(lig)} 个配体，期望 4；"
                               f"电荷转移的分配方式没定义")
        q, sig, eps = nb.getParticleParameters(zi)
        nb.setParticleParameters(zi, q.value_in_unit(unit.elementary_charge) - delta, sig, eps)
        for j in lig:
            qj, sj, ej = nb.getParticleParameters(j)
            nb.setParticleParameters(
                j, qj.value_in_unit(unit.elementary_charge) + delta / len(lig), sj, ej)
        moved += 1
    if verbose:
        q = np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                      for i in range(system.getNumParticles())])
        log(f"  电荷转移: {moved} 个 Zn，+2.0 -> {q_zn:+.2f}，"
            f"每个配体 {delta/4:+.4f} e；Σq = {q.sum():+.6f} e（应仍为 0）")
    return system


def build_1aay_system(padding_nm=1.5, ionic_M=0.15, cutoff_nm=1.0, q_zn=2.0,
                      ff_kind="amber", save=True, verbose=True):
    """返回 (topology, system_pme, positions)，签名对齐 build_1ckk_system()。

    q_zn: Zn 的有效电荷。2.0 = 标准 Amber 12-6（**故意保留固定电荷的过配位失败**）。
          <2.0 走 apply_charge_transfer()，用来标定修好配位需要多少 Δq。
    ff_kind: "amber"（固定电荷基线）或 "amoeba"（极化基线）。
          去质子 Cys 的残基名两边不同：Amber 叫 **CYM**，AMOEBA/Tinker 叫 **CYD**
          （已核对 amoeba2018.xml：CYD 无 HG、SG 多极 c0=-0.85155 vs CYS 的 -0.03264）。
    """
    if ff_kind not in ("amber", "amoeba"):
        raise ValueError(f"ff_kind 只支持 amber / amoeba，收到 {ff_kind!r}")
    thiolate = "CYM" if ff_kind == "amber" else "CYD"
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from pdbfixer import PDBFixer

    cys_seq, hid_seq, hie_seq = report_crystal_reference() if verbose else \
        (lambda t: t)(report_crystal_reference())

    if verbose:
        log("\n" + "=" * 66)
        log("构建体系")
        log("=" * 66)

    fixer = PDBFixer(filename=PDB_PATH)
    fixer.findMissingResidues(); fixer.findMissingAtoms()
    if verbose:
        log(f"  pdbfixer: 缺失残基段 {len(fixer.missingResidues)}，"
            f"缺重原子残基 {len(fixer.missingAtoms)}")
    fixer.addMissingAtoms()                       # 只补重原子，氢交给 Modeller
    top, pos = fixer.topology, fixer.positions

    ff = (app.ForceField('amber19-all.xml', 'amber19/tip3p.xml') if ff_kind == "amber"
          else app.ForceField('amoeba2018.xml'))
    # ⚠ 加氢**必须**用 Amber 力场，不能用 AMOEBA：
    #   Modeller.addHydrogens() 内部要 createSystem(nonbondedMethod=CutoffNonPeriodic)
    #   来优化氢位置，而 AmoebaMultipoleForce 只支持 NoCutoff / PME，会抛
    #   "Invalid nonbonded method for AmoebaMultipoleForce"。
    #   氢的位置不依赖最终用哪套力场，所以先用 Amber 加氢、再用目标力场建体系。
    ff_for_h = app.ForceField('amber19-all.xml', 'amber19/tip3p.xml')
    modeller = app.Modeller(top, pos)

    # --- His 变体：Modeller 认 HID/HIE，直接用 variants 指定 ---
    variants = []
    for res in modeller.topology.residues():
        if res.name == "HIS":
            variants.append("HID" if res.id and int(res.id) in hid_seq else "HIE")
        else:
            variants.append(None)
    modeller.addHydrogens(ff_for_h, pH=7.0, variants=variants)
    # ⚠ `Modeller.addHydrogens(variants=...)` **只按变体加氢，不改残基名**（实测）。
    #   结果是"名字叫 HIS、结构是 HID"：力场靠原子组成仍能匹配到 HID 模板（物理没错），
    #   但名字与化学不一致，任何按残基名做的逻辑都会看走眼。显式改名。
    n_ren = 0
    for res, v in zip(modeller.topology.residues(), variants):
        if v in ("HID", "HIE", "HIP") and res.name == "HIS":
            res.name = v; n_ren += 1
    if verbose:
        log(f"  HIS -> HID/HIE 显式改名 {n_ren} 个（addHydrogens 不改名）")
    if verbose:
        log(f"  加氢完成，原子数 {modeller.topology.getNumAtoms()}"
            f"（HID {sum(v=='HID' for v in variants)} 个 / "
            f"HIE {sum(v=='HIE' for v in variants)} 个）")

    # --- Cys -> CYM：hydrogens.xml 不认 CYM 变体，只能先加氢再删 HG 并改名 ---
    to_del, renamed = [], 0
    for res in modeller.topology.residues():
        if res.name == "CYS" and res.id and int(res.id) in cys_seq:
            hg = [a for a in res.atoms() if a.name == "HG"]
            if not hg:
                raise RuntimeError(f"CYS{res.id} 没有 HG，加氢步骤可能已改变；请检查")
            to_del += hg
            res.name = thiolate
            renamed += 1
    if renamed != len(cys_seq):
        raise RuntimeError(f"应改名 {len(cys_seq)} 个 CYS -> {thiolate}，实际 {renamed}")
    modeller.delete(to_del)
    if verbose:
        log(f"  CYS -> {thiolate}: {renamed} 个，删除 HG {len(to_del)} 个"
            f" -> 原子数 {modeller.topology.getNumAtoms()}")

    prot_top, prot_pos = modeller.topology, modeller.positions

    # --- 溶剂化 ---
    modeller.addSolvent(ff, model='tip3p', padding=padding_nm * unit.nanometer,
                        ionicStrength=ionic_M * unit.molar, neutralize=True)
    if verbose:
        bv = modeller.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        log(f"  溶剂化: padding {padding_nm} nm, {ionic_M} M NaCl -> "
            f"{modeller.topology.getNumAtoms()} 原子, 盒子 "
            f"{np.diag(np.array(bv)).round(3)} nm")

    if ff_kind == "amoeba":
        # AMOEBA 水柔性、不加约束；dt 必须 ≤1 fs
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
        apply_charge_transfer(system, modeller.topology, modeller.positions,
                              q_zn, verbose=verbose)
    if save:
        _save_prepared(modeller.topology, modeller.positions, system,
                       ff_kind, q_zn, padding_nm, ionic_M, prot_top, prot_pos,
                       cys_seq, hid_seq, hie_seq, verbose)
    if verbose:
        nbs = [system.getForce(i) for i in range(system.getNumForces())
               if isinstance(system.getForce(i), mm.NonbondedForce)]
        if nbs:
            q = np.array([nbs[0].getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                          for i in range(system.getNumParticles())])
            log(f"  createSystem OK: Σq = {q.sum():+.4f} e（应 ≈0）, Σq² = {np.sum(q**2):.1f} e²")
        else:
            log(f"  createSystem OK（{ff_kind} 走多极子，无 NonbondedForce）；"
                f"力数 {system.getNumForces()}")
        _report_built_coordination(modeller.topology, modeller.positions, verbose=True)
        log("\n  ⚠ 不加位置限制弹簧 —— 1CKK 的一个教训是锁住蛋白后极化响应根本没在采样里。")
        log("  ⚠ Zn 用的是纯非键 12-6 +2，**故意**保留固定电荷的过配位失败模式。")
    return modeller.topology, system, modeller.positions


def _save_prepared(top, pos, system, ff_kind, q_zn, padding_nm, ionic_M,
                   prot_top, prot_pos, cys_seq, hid_seq, hie_seq, verbose=True):
    """把处理好的拓扑/体系落盘到 1AAY/<ff_kind>/，免得每次重做转换。

    q_zn != 2.0 时目录名带后缀，避免覆盖基线。
    """
    import json
    import openmm as mm
    import openmm.app as app
    tag = ff_kind if abs(q_zn - 2.0) < 1e-12 else f"{ff_kind}_qzn{q_zn:g}"
    out = paths.system_dir("1AAY", tag)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "prepared_protein.pdb"), "w") as f:
        app.PDBFile.writeFile(prot_top, prot_pos, f, keepIds=True)
    with open(os.path.join(out, "solvated.pdb"), "w") as f:
        app.PDBFile.writeFile(top, pos, f, keepIds=True)
    with open(os.path.join(out, "system.xml"), "w") as f:
        f.write(mm.XmlSerializer.serialize(system))
    import openmm.unit as unit
    bv = np.array(top.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    meta = dict(ff_kind=ff_kind, q_zn=q_zn, padding_nm=padding_nm, ionic_M=ionic_M,
                n_atoms=top.getNumAtoms(), n_atoms_protein=prot_top.getNumAtoms(),
                box_nm=[float(x) for x in np.diag(bv)],
                thiolate_resname="CYM" if ff_kind == "amber" else "CYD",
                cym_resids=cys_seq, hid_resids=hid_seq, hie_resids=hie_seq,
                note="Zn 用纯非键 12-6；q_zn<2 是电荷转移诊断（差额均分给 4 配体，净电荷不变）")
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if verbose:
        log(f"  已落盘 -> 1AAY/{tag}/ "
            f"(prepared_protein.pdb, solvated.pdb, system.xml, meta.json)")


def _report_built_coordination(topology, positions, verbose=True):
    """建完体系后立刻复查配位 —— MD 之前就该还是 4 配位。"""
    import openmm.unit as unit
    xyz = np.array(positions.value_in_unit(unit.nanometer)) * 10.0   # -> Å
    atoms = list(topology.atoms())
    zn_idx = [a.index for a in atoms if a.residue.name in ("ZN", "Zn", "Zn2+")]
    log(f"\n  建成后配位复查（cutoff {ZN_LIGAND_CUTOFF_A} Å）:")
    for k, zi in enumerate(zn_idx, 1):
        d = np.linalg.norm(xyz - xyz[zi], axis=1)
        sel = [j for j in np.argsort(d)[1:12]
               if d[j] < ZN_LIGAND_CUTOFF_A and atoms[j].element is not None
               and atoms[j].element.symbol in ("N", "O", "S")]
        tags = [f"{atoms[j].residue.name}{atoms[j].residue.id}:{atoms[j].name}" for j in sel]
        log(f"    Zn{k}: CN={len(sel)}  {' '.join(tags)}")


def build_amoeba_from_amber_solvation(cutoff_nm=1.0, verbose=True):
    """从 1AAY/amber/solvated.pdb 直接建 AMOEBA 体系 —— **两条腿共用同一套坐标**。

    为什么必须这样：两次独立的 `addSolvent` 会随机摆水，实测导致 amber 起始
    CN=4/4/4 而 amoeba CN=4/5/5（多出来的是一个恰好落在 2.8 Å 内的 HOH:O）。
    初始条件在对照的两条腿之间随机不同是不能接受的。
    CYM 与 CYD 只差残基名、原子完全一样，所以复用坐标是严格合法的。
    """
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    src = paths.system_dir("1AAY", "amber", "solvated.pdb")
    if not os.path.exists(src):
        raise SystemExit(f"先建 amber 那条腿：python zinc_finger.py --ff amber\n缺 {src}")
    pdb = app.PDBFile(src)
    top = pdb.topology
    n = 0
    for r in top.residues():
        if r.name == "CYM":
            r.name = "CYD"; n += 1
    if verbose:
        log(f"  复用 amber 的溶剂化坐标：{top.getNumAtoms()} 原子，CYM->CYD {n} 个")
    ff = app.ForceField('amoeba2018.xml')
    system = ff.createSystem(top, nonbondedMethod=app.PME,
                             nonbondedCutoff=cutoff_nm * unit.nanometer,
                             constraints=None, rigidWater=False,
                             polarization='mutual', mutualInducedTargetEpsilon=1e-5)
    out = paths.system_dir("1AAY", "amoeba")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "solvated.pdb"), "w") as f:
        app.PDBFile.writeFile(top, pdb.positions, f, keepIds=True)
    with open(os.path.join(out, "system.xml"), "w") as f:
        f.write(mm.XmlSerializer.serialize(system))
    import json
    meta = json.load(open(paths.system_dir("1AAY", "amber", "meta.json")))
    meta.update(ff_kind="amoeba", thiolate_resname="CYD",
                shared_solvation_from="1AAY/amber/solvated.pdb",
                note=meta.get("note", "") + "；坐标与 amber 腿完全相同（共用溶剂化）")
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if verbose:
        log(f"  已落盘 -> 1AAY/amoeba/（与 amber 坐标严格一致）")
        _report_built_coordination(top, pdb.positions, verbose=True)
    return top, system, pdb.positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true", help="只报晶体参照，不建体系")
    ap.add_argument("--padding", type=float, default=1.5)
    ap.add_argument("--ionic", type=float, default=0.15)
    ap.add_argument("--zn-charge", type=float, default=2.0,
                    help="Zn 有效电荷。2.0=标准 Amber（故意保留过配位失败）；"
                         "<2.0 走电荷转移，差额均分给 4 个配体，净电荷不变")
    ap.add_argument("--ff", default="amber", choices=["amber", "amoeba"])
    ap.add_argument("--amoeba-from-amber", action="store_true",
                    help="AMOEBA 复用 amber 那条腿的溶剂化坐标，保证初始条件严格一致")
    a = ap.parse_args()
    if a.report_only:
        report_crystal_reference()
    elif a.amoeba_from_amber:
        build_amoeba_from_amber_solvation()
    else:
        build_1aay_system(padding_nm=a.padding, ionic_M=a.ionic, q_zn=a.zn_charge,
                          ff_kind=a.ff)


if __name__ == "__main__":
    main()
