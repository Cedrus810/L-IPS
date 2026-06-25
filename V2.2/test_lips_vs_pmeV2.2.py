import openmm as mm
from openmm import app, unit
import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib.pyplot as plt
import os, glob, gc, time, warnings
from pdbfixer import PDBFixer
warnings.filterwarnings('ignore')

# ==========================================
# 🌟 全局配置 (针对 1CKK 蛋白体系优化)
# ==========================================
TEMPERATURE = 300 * unit.kelvin
PRESSURE = 1.0 * unit.bar
DT = 0.002 * unit.picoseconds
# ⚠️ 测试阶段建议用短步数，跑通后再改大！
EQ_STEPS = 50000        # 100 ps (NVT+NPT 各 50ps)
PROD_STEPS = 250000     # 500 ps (正式发 paper 请改为 50000000 即 100ns)
REPORT_INTERVAL = 5000  # 10 ps 存一帧
DEBYE_PER_E_NM = 48.032
ENABLE_SOLUTE_POLARIZATION = True
SOLUTE_DPOLAR_BY_ELEMENT = {
    "O": 0.012,
    "N": 0.012,
    "C": 0.005,
    "S": 0.015,
    "H": 0.0,
}
SOLUTE_IS_POLAR_BY_ELEMENT = {
    "O": 1.0,
    "N": 1.0,
    "C": 0.8,
    "S": 1.0,
    "H": 0.5,
}

WATER_RESNAMES = {"HOH", "WAT", "SOL", "TP3"}
AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
ION_RESNAME_ALIASES = {
    "NA": "Na+", "Na": "Na+", "Na+": "Na+",
    "CL": "Cl-", "Cl": "Cl-", "Cl-": "Cl-",
    "K": "K+", "K+": "K+",
    "CA": "Ca2+", "Ca": "Ca2+", "CAL": "Ca2+", "Ca2+": "Ca2+",
    "MG": "Mg2+", "Mg": "Mg2+", "Mg2+": "Mg2+",
    "ZN": "Zn2+", "Zn": "Zn2+", "Zn2+": "Zn2+",
}

def normalize_ion_resnames(topology):
    for res in topology.residues():
        res.name = ION_RESNAME_ALIASES.get(res.name, res.name)

def extract_nonbonded_charges(system):
    nb_force = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    charges = np.zeros(system.getNumParticles())
    for i in range(system.getNumParticles()):
        q, _, _ = nb_force.getParticleParameters(i)
        charges[i] = q.value_in_unit(unit.elementary_charge)
    return charges

def binary_acf(occupancy):
    """Return normalized intermittent ACF for a boolean frame x site occupancy matrix."""
    occ = occupancy.astype(float)
    n_frames = occ.shape[0]
    acf = np.zeros(n_frames)
    for lag in range(n_frames):
        denom = np.sum(occ[:n_frames-lag])
        if denom > 0:
            acf[lag] = np.sum(occ[:n_frames-lag] * occ[lag:]) / denom
        else:
            acf[lag] = np.nan
    return acf

def first_crossing_time(times, values, threshold):
    valid = np.where(np.isfinite(values))[0]
    if len(valid) == 0:
        return np.nan
    for idx in valid:
        if values[idx] <= threshold:
            return times[idx]
    return np.nan

def build_1ckk_system():
    print("="*60)
    print("🧬 [1/4] 正在读取并修复 1CKK (Calcium-bound Calmodulin)...")
    print("="*60)
    # 1. 读取并修复 PDB (如果你没有 1CKK.cif，请改为 PDBFixer(pdbid='1ckk'))
    fixer = PDBFixer(filename='1CKK.pdb')
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)
    
    # 2. 加载力场 (Amber19SB + TIP3P，保证与 CWLD 参数兼容)
    print("  -> 加载 Amber19SB + TIP3P 力场...")
    forcefield = app.ForceField('amber19-all.xml', 'amber19/tip3p.xml')
    modeller = app.Modeller(fixer.topology, fixer.positions)
    
    # 3. 修复离子命名 Bug
    normalize_ion_resnames(modeller.topology)
        
    # 4. 加溶剂 (⚠️ 注意：CustomGBForce 计算密度极耗资源，这里 padding 设为 1.5nm 以加速测试。
    # 正式跑 100ns 时请改回 test.py 中的 3.5nm)
    print("  -> 添加 1.5 nm 水盒子与 0.15 M NaCl...")
    modeller.addSolvent(forcefield, model='tip3p', padding=1.5*unit.nanometer,
                        positiveIon='Na+', negativeIon='Cl-', ionicStrength=0.15*unit.molar)
    normalize_ion_resnames(modeller.topology)
                        
    # 5. 创建 PME 基准系统
    system_pme = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                         nonbondedCutoff=1.0*unit.nanometer, constraints=app.HBonds)
    
    # 6. 💡 核心修改：精准的位置限制 (完美复刻 test.py 的逻辑)
    print("  -> 添加位置限制弹簧 (锁蛋白重原子，放开 Ca2+/水/离子)...")
    restraint = mm.CustomExternalForce("k*periodicdistance(x,y,z,x0,y0,z0)^2")
    restraint.addGlobalParameter("k", 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    
    for atom in modeller.topology.atoms():
        # 只限制氨基酸重原子，绝对不限制 Ca2+、水、Na+、Cl-、H
        if atom.residue.name in AMINO_ACIDS and atom.element.symbol != 'H':
            restraint.addParticle(atom.index, modeller.positions[atom.index])
            
    system_pme.addForce(restraint)
    print(f"  -> ✅ 体系构建完成！总原子数: {modeller.topology.getNumAtoms()}")
    return modeller.topology, system_pme, modeller.positions

def setup_cwld_lips_system(base_system, topology, r_env=0.35, rc=1.2, r_on=0.9, rho0=13.5, k_polar=0.8, dpolar_O=-0.15, k_penalty=180.0):
    print("\n" + "="*60)
    print("⚙️ [2/4] 正在注入 CWLD-L-IPS 隐式极化引擎...")
    print("="*60)
    # 序列化复制系统 (连带 test.py 里的 Restriction Force 和 Barostat 一起完美复制！)
    sys_copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(base_system))
    orig_nb = next(f for f in sys_copy.getForces() if isinstance(f, mm.NonbondedForce))
    
    ONE_4PI_EPS0 = 138.935458
    n_atoms = sys_copy.getNumParticles()
    qbase = np.zeros(n_atoms)
    dpolar = np.zeros(n_atoms)
    is_polar = np.zeros(n_atoms)
    mol_ids = -np.ones(n_atoms) 
    water_mask = np.zeros(n_atoms, dtype=bool) 

    for i in range(n_atoms):
        q, _, _ = orig_nb.getParticleParameters(i)
        qbase[i] = q.value_in_unit(unit.elementary_charge)
        
    for residue in topology.residues():
        atom_indices = [atom.index for atom in residue.atoms()]
        mol_ids[atom_indices] = residue.index
        
        # 1. 处理水分子 (赋予极化率)
        if residue.name in WATER_RESNAMES:
            water_mask[atom_indices] = True
            o_atom = next((a for a in residue.atoms() if a.element.symbol == 'O'), None)
            h_atoms = [a for a in residue.atoms() if a.element.symbol == 'H']
            if o_atom and len(h_atoms) == 2:
                dpolar[o_atom.index] = dpolar_O
                dpolar[h_atoms[0].index] = -0.5 * dpolar_O
                dpolar[h_atoms[1].index] = -0.5 * dpolar_O
                is_polar[atom_indices] = 1.0
                
        # 2. 处理离子：保留力场 qbase，强制不参与自身极化
        elif residue.name in ION_RESNAME_ALIASES.values():
            dpolar[atom_indices] = 0.0
            is_polar[atom_indices] = 0.0

        # 3. 处理蛋白/有机溶质：按元素分配弱隐式极化响应
        else:
            if ENABLE_SOLUTE_POLARIZATION:
                for atom in residue.atoms():
                    elem = atom.element.symbol if atom.element is not None else "C"
                    dpolar[atom.index] = SOLUTE_DPOLAR_BY_ELEMENT.get(elem, 0.0)
                    is_polar[atom.index] = SOLUTE_IS_POLAR_BY_ELEMENT.get(elem, 0.0)
            else:
                dpolar[atom_indices] = 0.0
                is_polar[atom_indices] = 0.0

    high_charge_atoms = np.where(np.abs(qbase) > 1.1)[0]
    if len(high_charge_atoms) > 0:
        print(f"  -> 检测到 {len(high_charge_atoms)} 个 |q|>1.1e 的高电荷原子/离子，按固定电荷环境源处理")
    if ENABLE_SOLUTE_POLARIZATION:
        n_solute_polar = np.sum((is_polar > 0.0) & (~water_mask))
        print(f"  -> 溶质弱极化已开启: {n_solute_polar} 个非水原子获得元素级 dpolar/is_polar")

    # 引擎 A：原生 NonbondedForce
    orig_nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    orig_nb.setCutoffDistance(rc * unit.nanometer)
    orig_nb.setUseSwitchingFunction(True)
    orig_nb.setSwitchingDistance(r_on * unit.nanometer)
    
    # 引擎 B：CWLD 隐式极化力
    lips = mm.CustomGBForce()
    lips.setNonbondedMethod(mm.CustomGBForce.CutoffPeriodic)
    lips.setCutoffDistance(rc * unit.nanometer)

    lips.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
    lips.addGlobalParameter("r_env", r_env)
    lips.addGlobalParameter("rc", rc)
    lips.addGlobalParameter("r_on", r_on)
    lips.addGlobalParameter("rho0", rho0)
    lips.addGlobalParameter("k_penalty", k_penalty)
    lips.addGlobalParameter("k_polar", k_polar)

    # 🛡️ 核心修正：强制仅使用【水分子】计算 qref2，防止蛋白和 Ca2+ 污染基线！
    if np.any(water_mask):
        qref2 = np.mean(qbase[water_mask]**2)
        print(f"  -> 🛡️ 动态锚定【纯溶剂】基线: qref2 = {qref2:.4f} (严格排除蛋白与 Ca2+ 干扰)")
    else:
        qref2 = np.mean(qbase**2)

    lips.addGlobalParameter("a_q2", 0.5)     
    lips.addGlobalParameter("qref2", qref2)  

    lips.addPerParticleParameter("qbase")
    lips.addPerParticleParameter("dpolar")
    lips.addPerParticleParameter("is_polar")
    lips.addPerParticleParameter("mol_id")

    tanh_xmax = 100.0
    x_tanh = np.linspace(0.0, tanh_xmax, 4096)
    y_tanh = np.tanh(k_polar * x_tanh / rho0)
    lips.addTabulatedFunction("tanh_table", mm.Continuous1DFunction(y_tanh.tolist(), 0.0, tanh_xmax))

    x_ips = np.linspace(0.0, rc, 4096)
    y_ips = np.zeros_like(x_ips)
    mask = (x_ips >= r_on) & (x_ips <= rc)
    r_valid = x_ips[mask]
    ips_core = 1.0/r_valid - 1.5/rc + 0.5*(r_valid**2)/(rc**3)
    x_switch = (r_valid - r_on) / (rc - r_on)
    smooth_switch = 1.0 - (1.0 - x_switch)**2 * (1.0 + 2.0*x_switch)
    y_ips[mask] = ips_core * smooth_switch
    lips.addTabulatedFunction("ips_table", mm.Continuous1DFunction(y_ips.tolist(), 0.0, rc))

    w_geom = "(1.0 - (r/r_env)^2)^2 * step(r_env - r)"
    charge_mod = "(1.0 + a_q2 * (qbase2^2 - qref2))"
    density_expr = f"{w_geom} * {charge_mod} * step(abs(mol_id1 - mol_id2) - 0.5)"
    lips.addComputedValue("dens", density_expr, mm.CustomGBForce.ParticlePairNoExclusions)

    q_expr = "qbase + is_polar * dpolar * tanh_table(dens)"
    lips.addComputedValue("Q", q_expr, mm.CustomGBForce.SingleParticle)

    pair_expr = """ONE_4PI_EPS0 * (Q1*Q2 - qbase1*qbase2) * ips_table(r);"""
    lips.addEnergyTerm(pair_expr, mm.CustomGBForce.ParticlePair)

    self_expr = "0.5 * k_penalty * (Q - qbase)^2"
    lips.addEnergyTerm(self_expr, mm.CustomGBForce.SingleParticle)

    for i in range(n_atoms):
        lips.addParticle([float(qbase[i]), float(dpolar[i]), float(is_polar[i]), float(mol_ids[i])])
        
    for exc in range(orig_nb.getNumExceptions()):
        p1, p2, _, _, _ = orig_nb.getExceptionParameters(exc)
        lips.addExclusion(p1, p2)
        
    sys_copy.addForce(lips)
    return sys_copy

def run_multistage_md(label, system, topology, positions):
    print(f"\n[🚀 模拟] 启动 {label} 多阶段平行试验...")
    label_clean = label.replace(" ", "_")
    dcd_file = f"{label_clean}_1ckk.dcd"
    csv_file = f"{label_clean}_1ckk.csv"
    
    # 硬件加速
    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'Precision': 'mixed'}
    except:
        platform = mm.Platform.getPlatformByName('CPU')
        properties = {}
        
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, 1.0/unit.picosecond, DT)
    sim = app.Simulation(topology, system, integrator, platform, properties)
    sim.context.setPositions(positions)
    
    sim.reporters.append(app.DCDReporter(dcd_file, REPORT_INTERVAL))
    sim.reporters.append(app.StateDataReporter(csv_file, REPORT_INTERVAL, step=True, potentialEnergy=True, 
                                               temperature=True, volume=True, density=True, speed=True))
                                               
    # 阶段 1：能量最小化
    print("  -> [1/4] 能量最小化...")
    sim.minimizeEnergy(tolerance=10.0, maxIterations=2000)
    
    # 阶段 2：NVT 平衡 (带限制，k=1000)
    print("  -> [2/4] NVT 平衡 (蛋白受限，Ca2+/水 自由松弛)...")
    sim.step(EQ_STEPS // 2)
    
    # 阶段 3：NPT 平衡 (带限制，k=1000)
    print("  -> [3/4] 注入 Barostat，切换到 NPT 平衡 (稳定体系密度)...")
    system.addForce(mm.MonteCarloBarostat(PRESSURE, TEMPERATURE, 25))
    sim.context.reinitialize(preserveState=True)
    sim.step(EQ_STEPS // 2)
    
    # 阶段 4：解除限制，生产 MD
    print("  -> [4/4] 解除蛋白限制，开始无约束生产 MD (Ca2+ 完全自由)...")
    sim.context.setParameter("k", 0.0) # 💡 完美复刻 test.py 的解除锁定逻辑！
    sim.step(PROD_STEPS)
    
    print(f"  -> 🏁 {label} 模拟顺利跑完！")
    del sim
    gc.collect()
    return dcd_file, csv_file

def analyze_1ckk(label, dcd_file, csv_file, topology, system):
    print(f"\n[📊 分析] 正在提取 {label} 的核心指标...")
    md_top = md.Topology.from_openmm(topology)
    traj = md.load(dcd_file, top=md_top)
    traj = traj.image_molecules(inplace=False)
    times_ps = np.arange(traj.n_frames) * REPORT_INTERVAL * DT.value_in_unit(unit.picoseconds)
    charges = extract_nonbonded_charges(system)
    
    # 1. 蛋白 RMSD (评估 CWLD 是否破坏了蛋白折叠)
    protein_atoms = [a.index for a in traj.topology.atoms if a.residue.name in AMINO_ACIDS]
    protein_heavy_atoms = [
        a.index for a in traj.topology.atoms
        if a.residue.name in AMINO_ACIDS and a.element.symbol != 'H'
    ]
    rmsd = md.rmsd(traj, traj, atom_indices=protein_atoms)
    
    # 2. Ca2+ 配位环境 (评估 Ca2+ 是否与水/蛋白发生异常粘连)
    ca_atoms = [a.index for a in traj.topology.atoms if a.residue.name == 'Ca2+']
    water_o_atoms = [
        a.index
        for res in traj.topology.residues if res.name in WATER_RESNAMES
        for a in res.atoms if a.element.symbol == 'O'
    ]
    all_o_atoms = [a.index for a in traj.topology.atoms if a.element.symbol == 'O']

    def calc_ca_o_rdf(o_atoms):
        if not ca_atoms or not o_atoms:
            return np.array([]), np.array([])

        pairs = np.array([(i, j) for i in ca_atoms for j in o_atoms])
        r, g_r = md.compute_rdf(traj, pairs, r_range=(0.1, 0.6), bin_width=0.01)
        return r, g_r

    r_water, gr_water = calc_ca_o_rdf(water_o_atoms)
    r_all, gr_all = calc_ca_o_rdf(all_o_atoms)

    # 3. 表面水驻留时间：水氧进入蛋白重原子 0.35 nm 壳层的 intermittent ACF
    surface_cutoff_nm = 0.35
    water_residues = [res for res in traj.topology.residues if res.name in WATER_RESNAMES]
    water_o_by_residue = []
    for res in water_residues:
        o_atom = next((a for a in res.atoms if a.element.symbol == 'O'), None)
        if o_atom is not None:
            water_o_by_residue.append(o_atom.index)

    if protein_heavy_atoms and water_o_by_residue:
        surface_occupancy = np.zeros((traj.n_frames, len(water_o_by_residue)), dtype=bool)
        water_o_lookup = {atom_idx: pos for pos, atom_idx in enumerate(water_o_by_residue)}
        for frame_idx in range(traj.n_frames):
            neighbors = md.compute_neighbors(
                traj[frame_idx],
                surface_cutoff_nm,
                query_indices=protein_heavy_atoms,
                haystack_indices=water_o_by_residue,
            )[0]
            for atom_idx in neighbors:
                surface_occupancy[frame_idx, water_o_lookup[atom_idx]] = True

        residence_acf = binary_acf(surface_occupancy)
        residence_tau_ps = np.trapz(
            np.nan_to_num(residence_acf, nan=0.0),
            x=times_ps,
        )
        residence_t_1e_ps = first_crossing_time(times_ps, residence_acf, 1.0 / np.e)
        avg_surface_waters = float(np.mean(np.sum(surface_occupancy, axis=1)))
    else:
        residence_acf = np.array([])
        residence_tau_ps = np.nan
        residence_t_1e_ps = np.nan
        avg_surface_waters = np.nan

    # 4. 盐桥 RDF：Arg/Lys 带正电 N 与 Asp/Glu 带负电 O
    positive_n_atoms = [
        a.index for a in traj.topology.atoms
        if (
            (a.residue.name == "LYS" and a.name == "NZ") or
            (a.residue.name == "ARG" and a.name in {"NE", "NH1", "NH2"})
        )
    ]
    negative_o_atoms = [
        a.index for a in traj.topology.atoms
        if (
            (a.residue.name == "ASP" and a.name in {"OD1", "OD2"}) or
            (a.residue.name == "GLU" and a.name in {"OE1", "OE2"})
        )
    ]
    if positive_n_atoms and negative_o_atoms:
        salt_pairs = np.array([(i, j) for i in positive_n_atoms for j in negative_o_atoms])
        r_salt, gr_salt = md.compute_rdf(traj, salt_pairs, r_range=(0.2, 1.2), bin_width=0.01)
        salt_peak = float(np.max(gr_salt)) if len(gr_salt) else np.nan
        salt_peak_r = float(r_salt[np.argmax(gr_salt)]) if len(gr_salt) else np.nan
    else:
        r_salt, gr_salt = np.array([]), np.array([])
        salt_peak = np.nan
        salt_peak_r = np.nan

    # 5. 蛋白偶极矩波动：用 qbase * 构象坐标后处理；不直接输出 CustomGBForce 内部瞬时 Q
    if protein_atoms:
        protein_xyz = traj.xyz[:, protein_atoms, :]
        protein_charges = charges[protein_atoms]
        protein_charge_sum = np.sum(protein_charges)
        if abs(protein_charge_sum) > 1.0e-6:
            origin = np.average(protein_xyz, axis=1)
        else:
            origin = np.zeros((traj.n_frames, 3))
        centered_xyz = protein_xyz - origin[:, None, :]
        dipole_vectors = np.einsum("i,tij->tj", protein_charges, centered_xyz) * DEBYE_PER_E_NM
        dipole_magnitude = np.linalg.norm(dipole_vectors, axis=1)
        dipole_mean_debye = float(np.mean(dipole_magnitude))
        dipole_std_debye = float(np.std(dipole_magnitude))
    else:
        dipole_vectors = np.empty((0, 3))
        dipole_magnitude = np.array([])
        dipole_mean_debye = np.nan
        dipole_std_debye = np.nan

    metrics = {
        "mean_rmsd_nm": float(np.mean(rmsd)),
        "surface_water_tau_ps": float(residence_tau_ps),
        "surface_water_t_1e_ps": float(residence_t_1e_ps) if np.isfinite(residence_t_1e_ps) else np.nan,
        "avg_surface_waters": avg_surface_waters,
        "salt_bridge_peak_g": salt_peak,
        "salt_bridge_peak_r_nm": salt_peak_r,
        "protein_dipole_mean_debye": dipole_mean_debye,
        "protein_dipole_std_debye": dipole_std_debye,
    }

    analysis = {
        "metrics": metrics,
        "ca_water": (r_water, gr_water),
        "ca_all": (r_all, gr_all),
        "surface_water_acf": (times_ps[:len(residence_acf)], residence_acf),
        "salt_bridge_rdf": (r_salt, gr_salt),
        "dipole": (times_ps[:len(dipole_magnitude)], dipole_magnitude, dipole_vectors),
    }
        
    return analysis

def main():
    # 1. 构建体系 (自带 Restriction Force)
    topology, sys_pme, positions = build_1ckk_system()
    
    # 2. 注入 CWLD 引擎 (自动继承 Restriction Force)
    sys_lips = setup_cwld_lips_system(sys_pme, topology)
    
    # 3. 跑平行试验
    dcd_pme, csv_pme = run_multistage_md("PME_Baseline", sys_pme, topology, positions)
    dcd_lips, csv_lips = run_multistage_md("CWLD_LIPS", sys_lips, topology, positions)
    
    # 4. 分析对比
    ana_pme = analyze_1ckk("PME", dcd_pme, csv_pme, topology, sys_pme)
    ana_lips = analyze_1ckk("CWLD", dcd_lips, csv_lips, topology, sys_lips)
    metrics_df = pd.DataFrame({
        "PME": ana_pme["metrics"],
        "CWLD_LIPS": ana_lips["metrics"],
    }).T
    metrics_df.to_csv("1ckk_cwld_analysis_metrics.csv")
    
    print("\n" + "="*60)
    print(f"📊 最终结果对比:")
    print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("  -> 注：蛋白偶极矩后处理使用 qbase 与构象坐标；未直接读取 CWLD 内部瞬时 Q。")
    print("  -> 注：带净电荷蛋白的偶极矩依赖参考原点，当前以每帧蛋白几何中心为原点。")
    print("  -> ✅ 分析指标已保存至: 1ckk_cwld_analysis_metrics.csv")
    if metrics_df.loc["CWLD_LIPS", "surface_water_t_1e_ps"] > 200.0:
        print("  -> ⚠️ CWLD 表面水 ACF 的 1/e 时间超过 200 ps，建议检查 k_polar/a_q2 是否过度极化。")
    print("="*60)
    
    # 5. 绘图：Ca2+ - Oxygen RDF
    r_pme, gr_pme = ana_pme["ca_water"]
    r_lips, gr_lips = ana_lips["ca_water"]
    if len(r_pme) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(r_pme, gr_pme, label='PME (Ca-waterO)', color='black', lw=2)
        plt.plot(r_lips, gr_lips, label='CWLD-L-IPS (Ca-waterO)', color='purple', lw=2, ls='--')
        plt.title('Ca2+ - Water Oxygen Radial Distribution Function (1CKK)')
        plt.xlabel('r (nm)')
        plt.ylabel('g(r)')
        plt.xlim(0.1, 0.5)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('1ckk_ca_water_o_rdf.png', dpi=300)
        print("✅ Ca-waterO RDF 对比图已保存至: 1ckk_ca_water_o_rdf.png")

    r_pme, gr_pme = ana_pme["ca_all"]
    r_lips, gr_lips = ana_lips["ca_all"]
    if len(r_pme) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(r_pme, gr_pme, label='PME (Ca-allO)', color='black', lw=2)
        plt.plot(r_lips, gr_lips, label='CWLD-L-IPS (Ca-allO)', color='purple', lw=2, ls='--')
        plt.title('Ca2+ - All Oxygen Radial Distribution Function (1CKK)')
        plt.xlabel('r (nm)')
        plt.ylabel('g(r)')
        plt.xlim(0.1, 0.5)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('1ckk_ca_all_o_rdf.png', dpi=300)
        print("✅ Ca-allO RDF 对比图已保存至: 1ckk_ca_all_o_rdf.png")

    # 6. 表面水驻留 ACF
    t_pme, acf_pme = ana_pme["surface_water_acf"]
    t_lips, acf_lips = ana_lips["surface_water_acf"]
    if len(acf_pme) > 0 and len(acf_lips) > 0:
        n = min(len(t_pme), len(t_lips), len(acf_pme), len(acf_lips))
        pd.DataFrame({
            "time_ps": t_pme[:n],
            "PME": acf_pme[:n],
            "CWLD_LIPS": acf_lips[:n],
        }).to_csv("1ckk_surface_water_residence_acf.csv", index=False)
        plt.figure(figsize=(8, 5))
        plt.plot(t_pme, acf_pme, label='PME', color='black', lw=2)
        plt.plot(t_lips, acf_lips, label='CWLD-L-IPS', color='teal', lw=2, ls='--')
        plt.title('Surface Water Residence ACF (<0.35 nm from protein)')
        plt.xlabel('Lag time (ps)')
        plt.ylabel('C(t)')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('1ckk_surface_water_residence_acf.png', dpi=300)
        print("✅ 表面水驻留 ACF 已保存至: 1ckk_surface_water_residence_acf.png")

    # 7. 盐桥 RDF
    r_pme, gr_pme = ana_pme["salt_bridge_rdf"]
    r_lips, gr_lips = ana_lips["salt_bridge_rdf"]
    if len(r_pme) > 0 and len(r_lips) > 0:
        n = min(len(r_pme), len(r_lips), len(gr_pme), len(gr_lips))
        pd.DataFrame({
            "r_nm": r_pme[:n],
            "PME": gr_pme[:n],
            "CWLD_LIPS": gr_lips[:n],
        }).to_csv("1ckk_salt_bridge_rdf.csv", index=False)
        plt.figure(figsize=(8, 5))
        plt.plot(r_pme, gr_pme, label='PME', color='black', lw=2)
        plt.plot(r_lips, gr_lips, label='CWLD-L-IPS', color='crimson', lw=2, ls='--')
        plt.title('Salt-Bridge RDF: Arg/Lys N - Asp/Glu O')
        plt.xlabel('r (nm)')
        plt.ylabel('g(r)')
        plt.xlim(0.2, 1.2)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('1ckk_salt_bridge_rdf.png', dpi=300)
        print("✅ 盐桥 RDF 已保存至: 1ckk_salt_bridge_rdf.png")

    # 8. 蛋白偶极矩波动
    t_pme, dipole_pme, vec_pme = ana_pme["dipole"]
    t_lips, dipole_lips, vec_lips = ana_lips["dipole"]
    if len(dipole_pme) > 0 and len(dipole_lips) > 0:
        n = min(len(t_pme), len(t_lips), len(dipole_pme), len(dipole_lips))
        pd.DataFrame({
            "time_ps": t_pme[:n],
            "PME_debye": dipole_pme[:n],
            "CWLD_LIPS_debye": dipole_lips[:n],
        }).to_csv("1ckk_protein_dipole_timeseries.csv", index=False)
        plt.figure(figsize=(8, 5))
        plt.plot(t_pme, dipole_pme, label='PME', color='black', lw=2)
        plt.plot(t_lips, dipole_lips, label='CWLD-L-IPS', color='navy', lw=2, ls='--')
        plt.title('Protein Dipole Magnitude Fluctuation')
        plt.xlabel('Time (ps)')
        plt.ylabel('|mu| (Debye)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('1ckk_protein_dipole_fluctuation.png', dpi=300)
        print("✅ 蛋白偶极矩波动图已保存至: 1ckk_protein_dipole_fluctuation.png")

if __name__ == "__main__":
    main()
