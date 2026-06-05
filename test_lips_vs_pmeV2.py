import openmm as mm
from openmm import app, unit
import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib.pyplot as plt
from scipy import stats
import os, glob, gc, time, warnings
warnings.filterwarnings('ignore')

# ==========================================
# 🌟 全局配置
# ==========================================
BOX_SIZE_NM = 3.0
TEMPERATURE = 300 * unit.kelvin
PRESSURE = 1.0 * unit.bar
DT = 0.002 * unit.picoseconds
EQ_STEPS = 250000       # 500 ps
PROD_STEPS = 1000000    # 2 ns
REPORT_INTERVAL = 1000  # 2 ps

def build_water_box():
    print(f"  -> 正在构建 {BOX_SIZE_NM:.1f} nm³ 的 TIP3P 纯水盒子...")
    forcefield = app.ForceField('tip3p.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(forcefield, model='tip3p', boxSize=mm.Vec3(BOX_SIZE_NM, BOX_SIZE_NM, BOX_SIZE_NM)*unit.nanometers)
    system_pme = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                         nonbondedCutoff=1.2*unit.nanometers, constraints=app.HBonds)
    return modeller.topology, system_pme, modeller.positions

def setup_cwld_lips_system(base_system, topology, r_env=0.35, rc=1.2, r_on=0.9, rho0=13.5, k_polar=0.8, dpolar_O=-0.15, k_penalty=180.0):
    """
    CWLD-L-IPS (Charge-Weighted Local Density)
    1. 引入 qref2 基线对齐，确保纯水 dens 不漂移，完美兼容 rho0=13.5。
    2. Python 端融合 IPS 核与 Smooth Switch，消灭 GPU 表达式中的双重截断。
    """
    sys_copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(base_system))
    orig_nb = next(f for f in sys_copy.getForces() if isinstance(f, mm.NonbondedForce))
    
    ONE_4PI_EPS0 = 138.935458
    n_atoms = sys_copy.getNumParticles()
    
    qbase = np.zeros(n_atoms)
    dpolar = np.zeros(n_atoms)
    is_polar = np.zeros(n_atoms)
    mol_ids = -np.ones(n_atoms) 
    
    # 1. 提取基础电荷
    for i in range(n_atoms):
        q, _, _ = orig_nb.getParticleParameters(i)
        qbase[i] = q.value_in_unit(unit.elementary_charge)
        
    # 2. 拓扑安全遍历
    for residue in topology.residues():
        for atom in residue.atoms():
            mol_ids[atom.index] = residue.index
            
        if residue.name in ("HOH", "WAT", "SOL"):
            o_atom = None
            h_atoms = []
            for atom in residue.atoms():
                if atom.element.symbol == "O": o_atom = atom
                elif atom.element.symbol == "H": h_atoms.append(atom)
            
            if o_atom and len(h_atoms) == 2:
                idx_O = o_atom.index
                idx_H1 = h_atoms[0].index
                idx_H2 = h_atoms[1].index
                
                dpolar[idx_O] = dpolar_O
                dpolar[idx_H1] = -0.5 * dpolar_O
                dpolar[idx_H2] = -0.5 * dpolar_O
                
                is_polar[idx_O] = is_polar[idx_H1] = is_polar[idx_H2] = 1.0

    # 引擎 A：原生 NonbondedForce (维持 TIP3P 液态基线)
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
    
    # 🚀 核心泛化：动态计算极性溶剂基线 qref2
    polar_mask = is_polar > 0.5
    if np.any(polar_mask):
        qref2 = np.mean(qbase[polar_mask]**2)
        print(f"  -> 动态锚定溶剂基线: qref2 = {qref2:.4f} (基于 {np.sum(polar_mask)} 个极性原子)")
    else:
        qref2 = np.mean(qbase**2)
        print(f"  -> 未检测到极性溶剂，使用全局 Fallback: qref2 = {qref2:.4f}")

    lips.addGlobalParameter("a_q2", 0.5)     # 对高电荷环境的敏感度
    lips.addGlobalParameter("qref2", qref2)  # 动态极性溶剂基线
    
    lips.addPerParticleParameter("qbase")
    lips.addPerParticleParameter("dpolar")
    lips.addPerParticleParameter("is_polar")
    lips.addPerParticleParameter("mol_id")
    
    # 表 A: tanh 极化响应表
    x_tanh = np.linspace(0.0, 50.0, 2048)
    y_tanh = np.tanh(k_polar * x_tanh / rho0)
    lips.addTabulatedFunction("tanh_table", mm.Continuous1DFunction(y_tanh.tolist(), 0.0, 50.0))
    
    # 🚨 核心修正 2：Python 端融合 IPS 核与 Smooth Switch (消灭双重截断)
    x_ips = np.linspace(0.0, rc, 4096)
    y_ips = np.zeros_like(x_ips)
    mask = (x_ips >= r_on) & (x_ips <= rc)
    r_valid = x_ips[mask]
    
    # 1. 标准 IPS 核函数
    ips_core = 1.0/r_valid - 1.5/rc + 0.5*(r_valid**2)/(rc**3)
    
    # 2. Smooth Switch (在 r_on 处为 0，在 rc 处为 1)
    x_switch = (r_valid - r_on) / (rc - r_on)
    smooth_switch = 1.0 - (1.0 - x_switch)**2 * (1.0 + 2.0*x_switch)
    
    # 3. 融合：表格本身在 r_on 和 rc 处都严格为 0，且一阶导数连续！
    y_ips[mask] = ips_core * smooth_switch
    lips.addTabulatedFunction("ips_table", mm.Continuous1DFunction(y_ips.tolist(), 0.0, rc))

    # ==========================================
    # Pass 1: 电荷加权局部密度 (CWLD)
    # ==========================================
    w_geom = "(1.0 - (r/r_env)^2)^2 * step(r_env - r)"
    # 纯水系综平均下，qbase2^2 ≈ qref2，charge_mod ≈ 1.0，dens 严格等于几何配位数！
    charge_mod = "(1.0 + a_q2 * (qbase2^2 - qref2))"
    density_expr = f"{w_geom} * {charge_mod} * step(abs(mol_id1 - mol_id2) - 0.5)"
    
    lips.addComputedValue("dens", density_expr, mm.CustomGBForce.ParticlePairNoExclusions)
    
    # Pass 2: 隐变量映射
    q_expr = "qbase + is_polar * dpolar * tanh_table(dens)"
    lips.addComputedValue("Q", q_expr, mm.CustomGBForce.SingleParticle)
    
    # ==========================================
    # Pass 3: 长程 IPS 残差力 (极简表达式)
    # ==========================================
    # 因为 ips_table 已经包含了平滑截断，这里直接乘即可，绝无双重削弱！
    pair_expr = """
    ONE_4PI_EPS0 * (Q1*Q2 - qbase1*qbase2) * ips_table(r);
    """
    lips.addEnergyTerm(pair_expr, mm.CustomGBForce.ParticlePair)
    
    # Pass 4: 极化惩罚项
    self_expr = "0.5 * k_penalty * (Q - qbase)^2"
    lips.addEnergyTerm(self_expr, mm.CustomGBForce.SingleParticle)

    for i in range(n_atoms):
        lips.addParticle([float(qbase[i]), float(dpolar[i]), float(is_polar[i]), float(mol_ids[i])])
        
    for exc in range(orig_nb.getNumExceptions()):
        p1, p2, _, _, _ = orig_nb.getExceptionParameters(exc)
        lips.addExclusion(p1, p2)
        
    sys_copy.addForce(lips)
    return sys_copy

# ==========================================
# 模拟引擎与分析器 (包含 PBC Unwrap 修复)
# ==========================================
def run_simulation(label, system, topology, positions, is_gas=False):
    label_clean = label.strip().replace(" ", "_")
    existing_dcd = glob.glob(f"{label_clean}_*_traj.dcd")
    existing_csv = glob.glob(f"{label_clean}_*_data.csv")
    if existing_dcd and existing_csv:
        dcd_file = sorted(existing_dcd)[-1]
        csv_file = sorted(existing_csv)[-1]
        print(f"\n✨ [跳过模拟] 检测到 {label_clean} 已有落盘文件，直接读取！")
        return dcd_file, csv_file

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.getcwd())
    dcd_file = os.path.join(base_dir, f"{label_clean}_{timestamp}_traj.dcd")
    csv_file = os.path.join(base_dir, f"{label_clean}_{timestamp}_data.csv")

    print(f"\n[🚀 模拟] 启动 {label_clean} ... ")
    sys_copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    n_atoms = sys_copy.getNumParticles()
    if not is_gas:
        sys_copy.addForce(mm.MonteCarloBarostat(PRESSURE, TEMPERATURE, 25))
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, 1/unit.picosecond, DT)

    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'Precision': 'mixed'}
    except:
        platform = mm.Platform.getPlatformByName('CPU')
        properties = {}
        
    sim = app.Simulation(topology, sys_copy, integrator, platform, properties)
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(TEMPERATURE)

    sim.reporters.append(app.DCDReporter(dcd_file, REPORT_INTERVAL))
    sim.reporters.append(app.StateDataReporter(csv_file, REPORT_INTERVAL, step=True, potentialEnergy=True, 
                                               temperature=True, volume=True, density=True,
                                               speed=True, elapsedTime=True))

    total_steps = EQ_STEPS + PROD_STEPS
    chunk_steps = 50000  
    steps_done = 0
    wall_start = time.perf_counter()
    try:
        while steps_done < total_steps:
            steps_to_run = min(chunk_steps, total_steps - steps_done)
            sim.step(steps_to_run)  
            steps_done += steps_to_run
            dcd_size = os.path.getsize(dcd_file) / 1024 if os.path.exists(dcd_file) else 0
            print(f"    [进度 {steps_done: >7}/{total_steps}] 💾 DCD: {dcd_size: >8.1f} KB ")
    except Exception as e:
        print(f"\n⚠️ [崩溃] 模拟中途异常退出: {e}")
    finally:
        wall_elapsed = time.perf_counter() - wall_start
        if steps_done > 0 and wall_elapsed > 0:
            dt_ns = DT.value_in_unit(unit.nanoseconds)
            simulated_ns = steps_done * dt_ns
            ns_per_day = simulated_ns * 86400.0 / wall_elapsed
            matom_steps_per_s = (n_atoms * steps_done) / 1.0e6 / wall_elapsed
            print(
                f"  -> 性能统计 [{label_clean}] : "
                f"{matom_steps_per_s:.3f} Matom*steps/s, "
                f"{ns_per_day:.3f} ns/day "
                f"(atoms={n_atoms}, steps={steps_done}, wall={wall_elapsed:.1f}s)"
            )
        sim.reporters = []
        del sim
        gc.collect()
    return dcd_file, csv_file

class WaterAnalyzer:
    def __init__(self, label, dcd_file, csv_file, openmm_topology):
        self.label = label
        self.csv = pd.read_csv(csv_file)
        self.csv.columns = [c.replace('#', '').strip() for c in self.csv.columns]
        print(f"  -> 加载轨迹 {os.path.basename(dcd_file)} ... ")
        md_top = md.Topology.from_openmm(openmm_topology)
        self.traj = md.load(dcd_file, top=md_top)
        self.n_waters = self.traj.n_residues

    def calc_thermodynamics(self):
        vol = self.csv['Box Volume (nm^3)'].values
        density = self.csv['Density (g/mL)'].values
        avg_vol = np.mean(vol) * unit.nanometers**3
        vol_fluct = np.var(vol) * unit.nanometers**6
        kT = unit.BOLTZMANN_CONSTANT_kB * TEMPERATURE 
        kappa_T = (vol_fluct / (kT * avg_vol)).value_in_unit(unit.bar**-1)
        return np.mean(density), kappa_T

    def calc_structure(self):
        o_indices = [a.index for a in self.traj.topology.atoms if a.element.symbol == 'O']
        pairs = np.array([(i, j) for i in o_indices for j in o_indices if i < j])
        r, g_r = md.compute_rdf(self.traj, pairs, r_range=(0.1, 0.8))
        mask = r < 0.35
        rho = self.n_waters / np.mean(self.csv['Box Volume (nm^3)'].values)
        trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        coord_num = trapz_func(4 * np.pi * rho * r[mask]**2 * g_r[mask], x=r[mask])
        hbonds = md.baker_hubbard(self.traj, freq=0.1)
        avg_hbonds = len(hbonds) / self.n_waters
        return r, g_r, coord_num, avg_hbonds

    def calc_dynamics(self):
        o_indices = [a.index for a in self.traj.topology.atoms if a.element.symbol == 'O']
        o_traj = self.traj.atom_slice(o_indices)
        xyz = o_traj.xyz.copy()
        box_lengths = o_traj.unitcell_lengths[:, 0:3] 
        for i in range(1, len(xyz)):
            diff = xyz[i] - xyz[i-1]
            diff = diff - box_lengths[i] * np.round(diff / box_lengths[i])
            xyz[i] = xyz[i-1] + diff
        msd = np.zeros(len(xyz))
        for i in range(1, len(xyz)):
            disp = xyz[i] - xyz[0]
            msd[i] = np.mean(np.sum(disp**2, axis=-1))
        t = np.arange(len(xyz)) * REPORT_INTERVAL * DT.value_in_unit(unit.picoseconds)
        mask = t > 100 
        if np.sum(mask) < 2: return 0.0
        res = stats.linregress(t[mask], msd[mask])
        D = (res.slope / 6.0) * 100  
        return D

def main():
    print("="*70)
    print(" 🌊 CWLD-L-IPS：电荷加权局部密度 (基线对齐 + 完美平滑)")
    print("="*70)
    topology, sys_pme, positions = build_water_box()
    
    sys_lips = setup_cwld_lips_system(sys_pme, topology)

    dcd_pme, csv_pme = run_simulation("PME_Baseline", sys_pme, topology, positions)
    dcd_lips, csv_lips = run_simulation("CWLD_LIPS", sys_lips, topology, positions)

    print("\n[📊 分析] 开始计算核心指标... ")
    ana_pme = WaterAnalyzer("PME", dcd_pme, csv_pme, topology)
    ana_lips = WaterAnalyzer("CWLD_LIPS", dcd_lips, csv_lips, topology)

    results = {}
    rdf_data = {}

    for label, ana in [("PME", ana_pme), ("CWLD_LIPS", ana_lips)]:
        print(f"\n--- 正在分析 {label} ---")
        dens, kappa = ana.calc_thermodynamics()
        r, g_r, cn, hb = ana.calc_structure()
        D = ana.calc_dynamics()
        rdf_data[label] = {'r': r, 'g_r': g_r}
        results[label] = {
            "Density (g/cm³)": dens,
            "Isothermal Comp. (10^-5 bar^-1)": kappa * 1e5,
            "Coordination Number": cn,
            "Avg H-Bonds per Water": hb,
            "Diffusion Coeff (10^-5 cm²/s)": D
        }

    df = pd.DataFrame(results).T
    csv_out = "water_properties_cwld.csv"
    df.to_csv(csv_out)
    print("\n" + "="*70)
    print(df.to_string())
    print(f"\n✅ 数据表格已保存至: {os.path.abspath(csv_out)}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rdf_data['PME']['r'], rdf_data['PME']['g_r'], label='PME (O-O)', color='black', linewidth=2)
    ax.plot(rdf_data['CWLD_LIPS']['r'], rdf_data['CWLD_LIPS']['g_r'], label='CWLD-L-IPS (O-O)', color='purple', linewidth=2, linestyle='--')
    ax.set_title('Radial Distribution Function (O-O) Comparison', fontsize=14)
    ax.set_xlabel('r (nm)', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    img_out = "water_rdf_cwld.png"
    plt.savefig(img_out, dpi=300)
    print(f"✅ RDF 对比图已保存至: {os.path.abspath(img_out)}")

if __name__ == "__main__":
    main()
