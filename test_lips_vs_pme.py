import openmm as mm
from openmm import app, unit
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def build_water_box():
    print("  -> 正在构建 3x3x3 nm 的 TIP3P 纯水盒子...")
    forcefield = app.ForceField('tip3p.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(forcefield, model='tip3p', boxSize=mm.Vec3(3.0, 3.0, 3.0)*unit.nanometers)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                     nonbondedCutoff=1.2*unit.nanometers, constraints=app.HBonds)
    return modeller.topology, system, modeller.positions

def setup_ultimate_lips_system(base_system, topology):
    sys_copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(base_system))
    
    # 1. 改造原始 NonbondedForce (短程 q0 引擎)
    orig_nb = next((f for f in sys_copy.getForces() if isinstance(f, mm.NonbondedForce)), None)
    orig_nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    orig_nb.setCutoffDistance(1.2*unit.nanometers)
    orig_nb.setUseSwitchingFunction(True)
    orig_nb.setSwitchingDistance(0.9*unit.nanometers)
    
    orig_charges = []
    for i in range(orig_nb.getNumParticles()):
        q, sig, eps = orig_nb.getParticleParameters(i)
        orig_charges.append(q.value_in_unit(unit.elementary_charge))
        
    # ==========================================
    # 2. 构建终极 LIPS 引擎 (Texture & Fused 模拟)
    # ==========================================
    lips_force = mm.CustomGBForce()
    lips_force.setNonbondedMethod(mm.CustomGBForce.CutoffPeriodic)
    lips_force.setCutoffDistance(1.2*unit.nanometers)
    
    # 全局参数
    r_env = 0.35
    rc = 1.2
    r_on = 0.9
    rho0 = 13.5
    k_polar = 0.8
    k_penalty = 180.0
    
    lips_force.addGlobalParameter("r_env", r_env)
    lips_force.addGlobalParameter("rc", rc)
    lips_force.addGlobalParameter("r_on", r_on)
    lips_force.addGlobalParameter("ONE_4PI_EPS0", 138.935458)
    lips_force.addGlobalParameter("k_penalty", k_penalty)
    
    # 每粒子参数
    lips_force.addPerParticleParameter("qbase")
    lips_force.addPerParticleParameter("dpolar")
    lips_force.addPerParticleParameter("is_polar") 
    lips_force.addPerParticleParameter("mol_id") 
    
    # ==========================================
    # 🚀 核心核爆点 1：Texture Memory 表格化 (修正 API)
    # ==========================================
    # 表 A: tanh 极化响应表 (只需传 Y 值列表, X_min, X_max)
    x_tanh = np.linspace(0.0, 50.0, 1024)
    y_tanh = np.tanh(k_polar * x_tanh / rho0)
    # 【修正】：只传 y_tanh 的 list，OpenMM 会自动处理等间距 X
    tanh_table = mm.Continuous1DFunction(y_tanh.tolist(), 0.0, 50.0)
    lips_force.addTabulatedFunction("tanh_table", tanh_table)
    
    # 表 B: IPS 核函数表 
    x_ips = np.linspace(0.0, rc, 1024)
    y_ips = np.zeros_like(x_ips)
    mask = (x_ips >= r_on) & (x_ips <= rc) 
    r_valid = x_ips[mask]
    y_ips[mask] = 1.0/r_valid - 1.5/rc + 0.5*(r_valid**2)/(rc**3)
    # 【修正】：只传 y_ips 的 list
    ips_table = mm.Continuous1DFunction(y_ips.tolist(), 0.0, rc)
    lips_force.addTabulatedFunction("ips_table", ips_table)
    
    # ==========================================
    # 🚀 核心核爆点 2：计算图降维 (Fused 模拟)
    # ==========================================
    # Pass 1: 局部环境感知
    density_expr = "(1.0 - (r/r_env)^2)^2 * step(r_env - r) * step(abs(mol_id1 - mol_id2) - 0.5)"
    lips_force.addComputedValue("dens", density_expr, mm.CustomGBForce.ParticlePair)
    
    # Pass 2: 隐变量映射 (Single 循环，查表代替 tanh)
    q_expr = "qbase + is_polar * dpolar * tanh_table(dens)"
    lips_force.addComputedValue("Q", q_expr, mm.CustomGBForce.SingleParticle)
    
    # Pass 3: 远程 IPS 残差力 (Pair 循环，查表代替复杂多项式)
    pair_expr = """
    ONE_4PI_EPS0 * (Q1*Q2 - qbase1*qbase2) * S_LR * ips_table(r);
    
    S_LR = step(r - r_on) * smooth_switch;
    smooth_switch = 1.0 - (1.0 - x)^2 * (1.0 + 2.0*x);
    x = (r - r_on) / (rc - r_on);
    """
    lips_force.addEnergyTerm(pair_expr, mm.CustomGBForce.ParticlePair)
    
    # Pass 4: 极化惩罚 (Single 循环)
    self_expr = "0.5 * k_penalty * (Q - qbase)^2"
    lips_force.addEnergyTerm(self_expr, mm.CustomGBForce.SingleParticle)
    
    # 添加粒子
    atoms = list(topology.atoms())
    for i, q_val in enumerate(orig_charges):
        is_polar = 1.0 if atoms[i].element.symbol in ['O', 'H', 'N'] else 0.0
        dpolar_val = -0.15 if atoms[i].element.symbol == 'O' else 0.075
        mol_idx = atoms[i].residue.index
        lips_force.addParticle([q_val, dpolar_val, is_polar, mol_idx])
        
    # 复制排除列表
    for exc in range(orig_nb.getNumExceptions()):
        p1, p2, _, _, _ = orig_nb.getExceptionParameters(exc)
        lips_force.addExclusion(p1, p2)
        
    sys_copy.addForce(lips_force)
    return sys_copy

def run_npt_diagnostic(system, topology, positions, platform, properties, label=""):
    print(f"\n[诊断] 运行 {label} NPT (50ps)...")
    sys_copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    sys_copy.addForce(mm.MonteCarloBarostat(1.0*unit.bar, 300*unit.kelvin, 25))
    
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    sim = app.Simulation(topology, sys_copy, integrator, platform, properties)
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(300*unit.kelvin)
    
    integrator.step(10000) # 平衡 20ps
    
    volumes = []
    for _ in range(30): # 采样 30ps
        integrator.step(1000)
        box = sim.context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometers)
        volumes.append(np.linalg.det(box))
        
    avg_vol = np.mean(volumes)
    n_waters = system.getNumParticles() / 3
    density = (n_waters * 18.015 / 6.022e23) / (avg_vol * 1e-21) 
    
    print(f"  -> ✅ {label} 密度: {density:.3f} g/cm³")
    return density

def main():
    print("="*60)
    print(" L-IPS 终极硬件架构版：Texture Fetch & Fused Graph")
    print("="*60)
    topology, pme_system, positions = build_water_box()
    
    sys_lips = setup_ultimate_lips_system(pme_system, topology)
    
    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'Precision': 'mixed'}
    except:
        platform = mm.Platform.getPlatformByName('CPU')
        properties = {}
        
    dens_pme = run_npt_diagnostic(pme_system, topology, positions, platform, properties, "1. PME (Baseline)")
    dens_lips = run_npt_diagnostic(sys_lips, topology, positions, platform, properties, "2. L-IPS (Texture & Fused)")
    
    print("\n[可视化] 生成密度对照图...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['PME\n(Baseline)', 'L-IPS\n(Texture/Fused)']
    densities = [dens_pme, dens_lips]
    colors = ['black', 'purple']
    
    bars = ax.bar(labels, densities, color=colors, alpha=0.8, width=0.5)
    ax.axhline(0.98, color='blue', linestyle='--', lw=2, label='Ideal TIP3P (~0.98)')
    
    for bar, dens in zip(bars, densities):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{dens:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    ax.set_title('NPT Density: L-IPS Hardware-Optimized\n(Texture Memory & O(N) Graph Reduction)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density (g/cm³)', fontsize=12)
    ax.set_ylim(0.8, 1.1)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('lips_hardware_optimized.png', dpi=300)
    print("✅ 诊断完成！图表已保存为: lips_hardware_optimized.png")

if __name__ == "__main__":
    main()