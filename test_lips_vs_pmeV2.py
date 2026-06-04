import openmm as mm
from openmm import app, unit
import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob
import gc
import time
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 🌟 全局配置 (Global Config)
# ==========================================
BOX_SIZE_NM = 3.0
BOX_SIZE = BOX_SIZE_NM * unit.nanometers
TEMPERATURE = 300 * unit.kelvin
PRESSURE = 1.0 * unit.bar
DT = 0.002 * unit.picoseconds
EQ_STEPS = 250000       # 500 ps 平衡
PROD_STEPS = 1000000    # 2 ns 采样
REPORT_INTERVAL = 1000  # 2 ps 保存一帧

# ==========================================
# 1. 系统构建模块
# ==========================================
def build_water_box():
    print(f"  -> 正在构建 {BOX_SIZE_NM:.1f} x {BOX_SIZE_NM:.1f} x {BOX_SIZE_NM:.1f} nm³ 的 TIP3P 纯水盒子...")
    forcefield = app.ForceField('tip3p.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(forcefield, model='tip3p', boxSize=mm.Vec3(BOX_SIZE_NM, BOX_SIZE_NM, BOX_SIZE_NM)*unit.nanometers)
    system_pme = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                         nonbondedCutoff=1.2*unit.nanometers, constraints=app.HBonds)
    return modeller.topology, system_pme, modeller.positions

def setup_hybrid_lips_system(base_system, topology, r_env=0.35, rc=1.2, r_on=0.9, rho_bulk=13.5, k_polar=0.8, dpolar_O=-0.15, k_penalty=180.0):
    """
    Hybrid L-IPS v2.0 (Production Ready)
    Reference (Nonbonded) + Perturbation (CustomGB) 架构
    """
    sys_copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(base_system))
    orig_nb = next(f for f in sys_copy.getForces() if isinstance(f, mm.NonbondedForce))
    
    ONE_4PI_EPS0 = 138.935458
    n_atoms = sys_copy.getNumParticles()
    
    qbase = np.zeros(n_atoms)
    qbulk = np.zeros(n_atoms)
    dpolar = np.zeros(n_atoms)
    is_polar = np.zeros(n_atoms)
    mol_ids = -np.ones(n_atoms) # 修复: 默认 -1 防止 ID 碰撞
    
    # 1. 提取所有原子的基础电荷
    for i in range(n_atoms):
        q, _, _ = orig_nb.getParticleParameters(i)
        qbase[i] = q.value_in_unit(unit.elementary_charge)
        qbulk[i] = qbase[i] 
        
    # 2. 遍历所有残基分配 mol_id，并处理水分子极化
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

    tanh_bulk = np.tanh(k_polar)
    qbulk = qbase + dpolar * tanh_bulk

    # 引擎 A：修改原生 NonbondedForce (参考态大头)
    for i in range(n_atoms):
        _, sig, eps = orig_nb.getParticleParameters(i)
        orig_nb.setParticleParameters(i, qbulk[i] * unit.elementary_charge, sig, eps)
        
    orig_nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    orig_nb.setCutoffDistance(rc * unit.nanometer)
    orig_nb.setUseSwitchingFunction(True)
    orig_nb.setSwitchingDistance(r_on * unit.nanometer)

    # 引擎 B：构建残差微扰力 (CustomGBForce)
    lips = mm.CustomGBForce()
    lips.setNonbondedMethod(mm.CustomGBForce.CutoffPeriodic)
    lips.setCutoffDistance(rc * unit.nanometer)
    
    lips.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
    lips.addGlobalParameter("r_env", r_env)
    lips.addGlobalParameter("rho_bulk", rho_bulk)
    lips.addGlobalParameter("r_on", r_on)
    lips.addGlobalParameter("rc", rc)
    lips.addGlobalParameter("k_penalty", k_penalty)
    lips.addGlobalParameter("k_polar", k_polar)
    
    lips.addPerParticleParameter("qbase")
    lips.addPerParticleParameter("qbulk")
    lips.addPerParticleParameter("dpolar")
    lips.addPerParticleParameter("is_polar")
    lips.addPerParticleParameter("mol_id")
    
    x_tanh = np.linspace(0.0, 50.0, 2048)
    y_tanh = np.tanh(k_polar * x_tanh / rho_bulk)
    lips.addTabulatedFunction("tanh_table", mm.Continuous1DFunction(y_tanh.tolist(), 0.0, 50.0))
    
    x_ips = np.linspace(0.0, rc, 4096)
    y_ips = np.zeros_like(x_ips)
    mask = x_ips > 1e-6
    y_ips[mask] = 1.0 / x_ips[mask]
    lips.addTabulatedFunction("ips_table", mm.Continuous1DFunction(y_ips.tolist(), 0.0, rc))

    # Pass 1: 局部密度 (使用 ParticlePairNoExclusions 避免误伤)
    density_expr = "(1.0 - (r/r_env)^2)^2 * step(r_env - r) * step(abs(mol_id1 - mol_id2) - 0.5)"
    lips.addComputedValue("dens", density_expr, mm.CustomGBForce.ParticlePairNoExclusions)
    
    # Pass 2: 残差电荷 Delta_Q
    delta_q_expr = "is_polar * dpolar * (tanh_table(dens) - tanh_table(rho_bulk))"
    lips.addComputedValue("Delta_Q", delta_q_expr, mm.CustomGBForce.SingleParticle)
    
    # Pass 3: 残差相互作用
    pair_expr = """
    ONE_4PI_EPS0 * (Delta_Q1*qbulk2 + Delta_Q2*qbulk1 + Delta_Q1*Delta_Q2) * S_long * ips_table(r);
    S_long = step(r - r_on) * smooth_switch;
    smooth_switch = 1.0 - (1.0 - x)^2 * (1.0 + 2.0*x);
    x = (r - r_on) / (rc - r_on);
    """
    lips.addEnergyTerm(pair_expr, mm.CustomGBForce.ParticlePair)
    
    # Pass 4: 极化惩罚项 (惩罚 residual，确保体相中能量和力严格为 0)
    self_expr = "0.5 * k_penalty * Delta_Q^2"
    lips.addEnergyTerm(self_expr, mm.CustomGBForce.SingleParticle)

    for i in range(n_atoms):
        lips.addParticle([float(qbase[i]), float(qbulk[i]), float(dpolar[i]), float(is_polar[i]), float(mol_ids[i])])
        
    for exc in range(orig_nb.getNumExceptions()):
        p1, p2, _, _, _ = orig_nb.getExceptionParameters(exc)
        lips.addExclusion(p1, p2)
        
    sys_copy.addForce(lips)
    return sys_copy

# ==========================================
# 2. 模拟引擎 (防弹落盘版)
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
                                               temperature=True, volume=True, density=True))

    total_steps = EQ_STEPS + PROD_STEPS
    chunk_steps = 50000  
    steps_done = 0
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
        sim.reporters = []
        del sim
        gc.collect()

    return dcd_file, csv_file

# ==========================================
# 3. 核心指标分析引擎 (全兼容修复版)
# ==========================================
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
        
        # 兼容 NumPy 2.0+ 与旧版本
        trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        coord_num = trapz_func(4 * np.pi * rho * r[mask]**2 * g_r[mask], x=r[mask])
        
        hbonds = md.baker_hubbard(self.traj, freq=0.1)
        avg_hbonds = len(hbonds) / self.n_waters
        
        return r, g_r, coord_num, avg_hbonds

    def calc_dynamics(self):
        o_indices = [a.index for a in self.traj.topology.atoms if a.element.symbol == 'O']
        o_traj = self.traj.atom_slice(o_indices)
        
        # 🚨 致命修复：PBC Unwrap (防止水分子跨盒子导致 MSD 饱和)
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
        mask = t > 100 # 取 100ps 后的线性区
        if np.sum(mask) < 2: return 0.0
        
        res = stats.linregress(t[mask], msd[mask])
        D = (res.slope / 6.0) * 100  # nm^2/ps -> 10^-5 cm^2/s
        return D

# ==========================================
# 4. 主函数与文件导出
# ==========================================
def main():
    print("="*70)
    print(" 🌊 终极水模型物理化学性质全景分析 Pipeline (PME vs Hybrid L-IPS v2.0)")
    print("="*70)
    topology, sys_pme, positions = build_water_box()
    
    # 使用我们推导出的完美混合架构
    sys_lips = setup_hybrid_lips_system(sys_pme, topology)

    # 1. 运行/加载 模拟
    dcd_pme, csv_pme = run_simulation("PME_Baseline", sys_pme, topology, positions)
    dcd_lips, csv_lips = run_simulation("Hybrid_LIPS_v2", sys_lips, topology, positions)

    # 2. 初始化分析器
    print("\n[📊 分析] 开始计算核心指标... ")
    ana_pme = WaterAnalyzer("PME", dcd_pme, csv_pme, topology)
    ana_lips = WaterAnalyzer("Hybrid_LIPS_v2", dcd_lips, csv_lips, topology)

    results = {}
    rdf_data = {}

    for label, ana in [("PME", ana_pme), ("Hybrid_LIPS_v2", ana_lips)]:
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

    # 3. 导出 CSV 数据文件 💾
    df = pd.DataFrame(results).T
    csv_out = "water_properties_comparison.csv"
    df.to_csv(csv_out)

    print("\n" + "="*70)
    print(" 📈 核心物理化学性质对比报告  ")
    print("="*70)
    print(df.to_string())
    print(f"\n✅ 数据表格已保存至: {os.path.abspath(csv_out)}")

    # 4. 可视化 RDF 对比图并导出 📊
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rdf_data['PME']['r'], rdf_data['PME']['g_r'], label='PME (O-O)', color='black', linewidth=2)
    ax.plot(rdf_data['Hybrid_LIPS_v2']['r'], rdf_data['Hybrid_LIPS_v2']['g_r'], label='Hybrid L-IPS v2.0 (O-O)', color='purple', linewidth=2, linestyle='--')

    ax.set_title('Radial Distribution Function (O-O) Comparison', fontsize=14)
    ax.set_xlabel('r (nm)', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    img_out = "water_rdf_comparison.png"
    plt.savefig(img_out, dpi=300)
    print(f"✅ RDF 对比图已保存至: {os.path.abspath(img_out)}")

if __name__ == "__main__":
    main()