import openmm as mm
from openmm import app, unit
import numpy as np
import matplotlib.pyplot as plt

def build_water_box():
    """构建一个简单的纯水盒子用于测试"""
    forcefield = app.ForceField('tip3p.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(forcefield, model='tip3p', boxSize=mm.Vec3(3.0, 3.0, 3.0)*unit.nanometers)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME, 
                                     nonbondedCutoff=1.0*unit.nanometers, constraints=app.HBonds)
    return modeller.topology, system, modeller.positions

def setup_lips_system(system, topology):
    """将传统 PME 系统改造为 L-IPS 系统"""
    # 1. 找到并禁用原有的 NonbondedForce (将其电荷设为0，只保留 LJ)
    lips_system = mm.System()
    lips_system.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    
    # 复制粒子和约束
    for i in range(system.getNumParticles()):
        lips_system.addParticle(system.getParticleMass(i))
    for i in range(system.getNumConstraints()):
        p1, p2, dist = system.getConstraintParameters(i)
        lips_system.addConstraint(p1, p2, dist)

    nb_force = None
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            nb_force = force
            break
            
    # 复制 Lennard-Jones (将电荷设为0)
    lj_force = mm.NonbondedForce()
    lj_force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    lj_force.setCutoffDistance(1.0*unit.nanometers)
    lj_force.setUseSwitchingFunction(True)
    lj_force.setSwitchingDistance(0.9*unit.nanometers)
    
    # 2. 构建核心的 L-IPS CustomGBForce
    lips_force = mm.CustomGBForce()
    lips_force.setNonbondedMethod(mm.CustomGBForce.CutoffPeriodic)
    lips_force.setCutoffDistance(1.0*unit.nanometers) # IPS 的长程 cutoff (rc)
    
    # 添加全局参数
    lips_force.addGlobalParameter("r_local", 0.35) # 局部环境感知半径 (3.5 A)
    lips_force.addGlobalParameter("rc", 1.0)       # IPS 截断半径 (10 A)
    lips_force.addGlobalParameter("ONE_4PI_EPS0", 138.935458) # 库仑常数
    
    # 添加每个粒子的参数 (基础电荷 q0, 极化响应系数 delta_q)
    lips_force.addPerParticleParameter("q0")
    lips_force.addPerParticleParameter("delta_q")
    
    # Pass 1: 计算局部环境密度 (Density)
    # 使用平滑的钟形截断函数，保证在 r_local 处一阶导为 0
    density_expr = "(1.0 - (r/r_local)^2)^2 * step(r_local - r)"
    lips_force.addComputedValue("Density", density_expr, mm.CustomGBForce.ParticlePairNoExclusions)
    
    # Pass 2: 计算 L-IPS 能量
    # IPS 势函数: 1/r + r^2/(2*rc^3) - 1.5/rc (保证在 rc 处能量和力都为 0)
    # Q_i = q0_i + delta_q_i * Density_i (隐变量电荷更新)
    energy_expr = """
    ONE_4PI_EPS0 * Q_i * Q_j * ips_pot;
    ips_pot = 1/r + r^2/(2*rc^3) - 1.5/rc;
    Q_i = q0_i + delta_q_i * Density_i;
    Q_j = q0_j + delta_q_j * Density_j;
    """
    lips_force.addEnergyTerm(energy_expr, mm.CustomGBForce.ParticlePair)
    
    # 3. 同步参数和 Exclusions (排除列表)
    for i in range(nb_force.getNumParticles()):
        charge, sigma, epsilon = nb_force.getParticleParameters(i)
        # 将电荷转移到 L-IPS，LJ 保留在 NonbondedForce
        lj_force.addParticle(0.0, sigma, epsilon) 
        
        # 设置 L-IPS 参数 (这里为了演示，给 O 和 H 设置不同的极化率，且保持分子内中性)
        # TIP3P 中 O 是偶数索引，H 是奇数索引 (简化处理，实际应按 atom name)
        q0_val = charge.value_in_unit(unit.elementary_charge)
        if i % 3 == 0: # 假设是 Oxygen
            delta_q = 0.10  # 氧原子对密度敏感
        else:          # 假设是 Hydrogen
            delta_q = -0.05 # 氢原子响应，保证 0.10 + 2*(-0.05) = 0 (分子内守恒)
            
        lips_force.addParticle([q0_val, delta_q])
        
    # 严格复制 Exclusions (1-2, 1-3 排除)
    for i in range(nb_force.getNumExceptions()):
        p1, p2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(i)
        lj_force.addException(p1, p2, 0.0, sigma, epsilon)
        lips_force.addExclusion(p1, p2)
        
    lips_system.addForce(lj_force)
    lips_system.addForce(lips_force)
    
    return lips_system

def run_comparison():
    print("1. 构建纯水体系...")
    topology, pme_system, positions = build_water_box()
    lips_system = setup_lips_system(pme_system, topology)
    
    integrator_pme = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    integrator_lips = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    
    platform = mm.Platform.getPlatformByName('CUDA') # 如果有 GPU 用 CUDA，否则用 CPU
    properties = {'Precision': 'mixed'} if platform.getName() == 'CUDA' else {}
    
    sim_pme = app.Simulation(topology, pme_system, integrator_pme, platform, properties)
    sim_lips = app.Simulation(topology, lips_system, integrator_lips, platform, properties)
    
    sim_pme.context.setPositions(positions)
    sim_lips.context.setPositions(positions)
    
    print("2. 计算 PME 能量与受力...")
    state_pme = sim_pme.context.getState(getEnergy=True, getForces=True)
    pe_pme = state_pme.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces_pme = state_pme.getForces(asNumpy=True)
    
    print("3. 计算 L-IPS 能量与受力...")
    state_lips = sim_lips.context.getState(getEnergy=True, getForces=True)
    pe_lips = state_lips.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces_lips = state_lips.getForces(asNumpy=True)
    
    print("\n" + "="*40)
    print(f"总势能对比:")
    print(f"  PME 势能:   {pe_pme:12.2f} kJ/mol")
    print(f"  L-IPS 势能: {pe_lips:12.2f} kJ/mol")
    print(f"  差异:       {abs(pe_pme - pe_lips):12.2f} kJ/mol")
    print("="*40)
    
    # 对比受力
    forces_pme_flat = forces_pme.flatten()
    forces_lips_flat = forces_lips.flatten()
    
    rmse = np.sqrt(np.mean((forces_pme_flat - forces_lips_flat)**2))
    correlation = np.corrcoef(forces_pme_flat, forces_lips_flat)[0, 1]
    
    print(f"\n受力对比 (Forces):")
    print(f"  力的 RMSE:  {rmse:.4f} kJ/mol/nm")
    print(f"  皮尔逊相关系数 (Pearson R): {correlation:.4f}")
    
    # 画图
    plt.figure(figsize=(6, 6))
    plt.scatter(forces_pme_flat, forces_lips_flat, s=1, alpha=0.5, color='blue')
    min_f, max_f = min(forces_pme_flat.min(), forces_lips_flat.min()), max(forces_pme_flat.max(), forces_lips_flat.max())
    plt.plot([min_f, max_f], [min_f, max_f], 'r--', label='y = x (Perfect Match)')
    plt.xlabel('PME Forces (kJ/mol/nm)')
    plt.ylabel('L-IPS Forces (kJ/mol/nm)')
    plt.title(f'L-IPS vs PME Forces (R = {correlation:.3f})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lips_vs_pme_forces.png', dpi=300)
    print("\n受力对比散点图已保存为 'lips_vs_pme_forces.png'")

if __name__ == "__main__":
    run_comparison()