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
                                     nonbondedCutoff=1.0*unit.nanometers, constraints=app.HBonds)
    return modeller.topology, system, modeller.positions

def setup_lips_system(system, topology):
    """构建物理严密的 L-IPS 系统 (带归一化与纯衰减核)"""
    lips_system = mm.System()
    lips_system.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    
    for i in range(system.getNumParticles()):
        lips_system.addParticle(system.getParticleMass(i))
    for i in range(system.getNumConstraints()):
        p1, p2, dist = system.getConstraintParameters(i)
        lips_system.addConstraint(p1, p2, dist)

    nb_force = next((f for f in system.getForces() if isinstance(f, mm.NonbondedForce)), None)
            
    lj_force = mm.NonbondedForce()
    lj_force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    lj_force.setCutoffDistance(1.0*unit.nanometers)
    lj_force.setUseSwitchingFunction(True)
    lj_force.setSwitchingDistance(0.9*unit.nanometers)
    
    lips_force = mm.CustomGBForce()
    lips_force.setNonbondedMethod(mm.CustomGBForce.CutoffPeriodic)
    lips_force.setCutoffDistance(1.0*unit.nanometers) 
    
    lips_force.addGlobalParameter("r_local", 0.35) 
    lips_force.addGlobalParameter("rc", 1.0)       
    lips_force.addGlobalParameter("ONE_4PI_EPS0", 138.935458) 
    lips_force.addGlobalParameter("rho0", 5.0) # 参考密度，防止极化爆炸
    
    # 彻底抛弃下划线，使用纯字母命名避开 OpenMM 解析器暗礁
    lips_force.addPerParticleParameter("qbase")
    lips_force.addPerParticleParameter("dpolar")
    
    # Pass 1: 局部环境密度
    density_expr = "(1.0 - (r/r_local)^2)^2 * step(r_local - r)"
    lips_force.addComputedValue("dens", density_expr, mm.CustomGBForce.ParticlePairNoExclusions)
    
    # Pass 2: 双体能量 (纯衰减核，在 rc 处能量和力完美归零，无全局常数黑洞)
    energy_expr = """
    ONE_4PI_EPS0 * Q1 * Q2 * ips_pot;
    ips_pot = (1/r) * (1 - r/rc)^2;
    Q1 = qbase1 + dpolar1 * (dens1 / rho0);
    Q2 = qbase2 + dpolar2 * (dens2 / rho0);
    """
    lips_force.addEnergyTerm(energy_expr, mm.CustomGBForce.ParticlePair)
    
    for i in range(nb_force.getNumParticles()):
        charge, sigma, epsilon = nb_force.getParticleParameters(i)
        lj_force.addParticle(0.0, sigma, epsilon) 
        q_val = charge.value_in_unit(unit.elementary_charge)
        dpolar_val = 0.10 if i % 3 == 0 else -0.05 
        lips_force.addParticle([q_val, dpolar_val])
        
    for i in range(nb_force.getNumExceptions()):
        p1, p2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(i)
        lj_force.addException(p1, p2, 0.0, sigma, epsilon)
        lips_force.addExclusion(p1, p2) 
        
    lips_system.addForce(lj_force)
    lips_system.addForce(lips_force)
    return lips_system

def get_platform():
    for p_name in ['CUDA', 'OpenCL', 'CPU']:
        try:
            platform = mm.Platform.getPlatformByName(p_name)
            properties = {'Precision': 'mixed'} if p_name in ['CUDA', 'OpenCL'] else {}
            return platform, properties, p_name
        except Exception:
            continue
    raise RuntimeError("找不到可用的 OpenMM 计算平台！")

def compute_rdf_oxygen(trajectory_O, box_lengths, r_max=0.8, n_bins=80):
    """纯 NumPy 实现的带 PBC 的 O-O RDF 计算引擎"""
    hist = np.zeros(n_bins)
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    
    n_frames = len(trajectory_O)
    n_particles = trajectory_O[0].shape[0]
    volume = box_lengths[0] * box_lengths[1] * box_lengths[2]
    rho = n_particles / volume
    
    for frame in trajectory_O:
        diff = frame[:, np.newaxis, :] - frame[np.newaxis, :, :]
        diff = diff - box_lengths * np.round(diff / box_lengths) # 最小镜像约定
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, r_max + 1.0) # 排除自身
        counts, _ = np.histogram(dist, bins=r_bins)
        hist += counts
        
    shell_vols = 4/3 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    rdf = hist / (n_frames * n_particles * rho * shell_vols)
    return r_centers, rdf

def run_nve_test(system, topology, positions, platform, properties):
    """NVE 能量守恒测试与线性漂移率计算"""
    print("\n[3] 运行 L-IPS NVE 能量守恒测试 (50,000步 / 50ps)...")
    integrator = mm.VerletIntegrator(0.001*unit.picoseconds)
    sim = app.Simulation(topology, system, integrator, platform, properties)
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(300*unit.kelvin)
    
    integrator.step(1000) # 短暂平衡
    
    n_steps = 50000
    save_interval = 500 
    n_frames = n_steps // save_interval
    
    times = []
    energies = []
    
    for i in range(n_frames):
        integrator.step(save_interval)
        state = sim.context.getState(getEnergy=True)
        ke = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        times.append(i * save_interval * 0.001) 
        energies.append(ke + pe)
        
    times = np.array(times)
    energies = np.array(energies)
    
    slope, intercept = np.polyfit(times, energies, 1)
    n_atoms = system.getNumParticles()
    drift_rate_per_atom = (slope * 1000) / n_atoms 
    
    print(f"  -> 线性漂移率 (Drift Rate): {drift_rate_per_atom:.6f} kJ/mol/ns/atom")
    print(f"  -> 结论: {'🟢 完美守恒' if abs(drift_rate_per_atom) < 0.05 else '🔴 存在系统性泄漏'}")
    
    return times, energies, drift_rate_per_atom

def run_nvt_sampling(system, topology, positions, platform, properties, label=""):
    """NVT 采样收集轨迹"""
    print(f"\n[*] 运行 {label} NVT 采样 (20ps, 收集50帧)...")
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    sim = app.Simulation(topology, system, integrator, platform, properties)
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(300*unit.kelvin)
    
    integrator.step(1000) # 平衡 2ps
    
    traj_O = []
    boxes = []
    for _ in range(50):
        integrator.step(200)
        state = sim.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometers)
        traj_O.append(pos[0::3]) # 只取 Oxygen
        boxes.append(np.diag(box))
        
    return traj_O, np.mean(boxes, axis=0)

def main():
    print("="*50)
    print(" L-IPS 终极验证流水线 (单点力 + NVE + RDF)")
    print("="*50)
    
    topology, pme_system, positions = build_water_box()
    lips_system = setup_lips_system(pme_system, topology)
    platform, properties, p_name = get_platform()
    print(f"  -> 使用计算平台: {p_name}")
    
    # 1. 单点受力对比
    print("\n[1] 计算单点受力对比...")
    sim_pme = app.Simulation(topology, pme_system, mm.LangevinMiddleIntegrator(300, 1, 0.002), platform, properties)
    sim_lips = app.Simulation(topology, lips_system, mm.LangevinMiddleIntegrator(300, 1, 0.002), platform, properties)
    sim_pme.context.setPositions(positions)
    sim_lips.context.setPositions(positions)
    
    f_pme = sim_pme.context.getState(getForces=True).getForces(asNumpy=True)[0::3].flatten()
    f_lips = sim_lips.context.getState(getForces=True).getForces(asNumpy=True)[0::3].flatten()
    r_val = np.corrcoef(f_pme, f_lips)[0, 1]
    print(f"  -> 氧原子受力 Pearson R: {r_val:.4f}")
    
    # 2. NVE 能量守恒测试
    times, nve_energies, drift_rate = run_nve_test(lips_system, topology, positions, platform, properties)
    
    # 3. NVT 采样与 RDF 计算
    print("\n[4] 运行 NVT 采样与 RDF 计算...")
    traj_pme, box_pme = run_nvt_sampling(pme_system, topology, positions, platform, properties, "PME")
    traj_lips, box_lips = run_nvt_sampling(lips_system, topology, positions, platform, properties, "L-IPS")
    
    r_bins, rdf_pme = compute_rdf_oxygen(traj_pme, box_pme)
    _, rdf_lips = compute_rdf_oxygen(traj_lips, box_lips)
    print("  -> RDF 计算完成！")
    
    # 4. 生成可视化图表
    print("\n[5] 生成可视化图表...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1: 受力散点图
    axs[0].scatter(f_pme, f_lips, s=2, alpha=0.5, color='blue')
    min_f, max_f = min(f_pme.min(), f_lips.min()), max(f_pme.max(), f_lips.max())
    axs[0].plot([min_f, max_f], [min_f, max_f], 'r--', lw=2)
    axs[0].set_title(f'Forces (R = {r_val:.3f})')
    axs[0].set_xlabel('PME O-Forces')
    axs[0].set_ylabel('L-IPS O-Forces')
    axs[0].grid(True, alpha=0.5)
    
    # 图2: NVE 能量漂移
    axs[1].plot(times, nve_energies, color='green', lw=1.5, alpha=0.8)
    fit_y = np.polyfit(times, nve_energies, 1)
    axs[1].plot(times, np.polyval(fit_y, times), 'r--', lw=2, label=f'Fit: {drift_rate:.4f}')
    axs[1].set_title(f'NVE Total Energy (50 ps)\nDrift: {drift_rate:.4f} kJ/mol/ns/atom')
    axs[1].set_xlabel('Time (ps)')
    axs[1].set_ylabel('Total Energy (kJ/mol)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.5)
    
    # 图3: O-O RDF
    axs[2].plot(r_bins, rdf_pme, 'k-', lw=2, label='PME')
    axs[2].plot(r_bins, rdf_lips, 'r--', lw=2, label='L-IPS')
    axs[2].set_title('O-O Radial Distribution Function')
    axs[2].set_xlabel('r (nm)')
    axs[2].set_ylabel('g(r)')
    axs[2].legend()
    axs[2].grid(True, alpha=0.5)
    axs[2].set_xlim(0.2, 0.8)
    
    plt.tight_layout()
    out_file = 'lips_full_validation.png'
    plt.savefig(out_file, dpi=300)
    print(f"\n✅ 验证完成！综合图表已保存为: {out_file}")

if __name__ == "__main__":
    main()