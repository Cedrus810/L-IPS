import openmm as mm
from openmm import app, unit
import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib.pyplot as plt
import os, glob, gc, time, warnings
from pdbfixer import PDBFixer
from scipy.spatial import cKDTree
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
DEFAULT_A_Q2 = 0.5
PURE_DENSITY_A_Q2 = 0.0
INCLUDE_BASELINE_IPS = True
USE_SELF_CONSISTENT_IPS_PAIR = True
CWLD_ENGINE = "customgb_exact"  # "customgb_exact" or "cpu_kdtree_fast"
Q_UPDATE_INTERVAL = 50
REPORT_POTENTIAL_ENERGY = False
NPT_BAROSTAT_INTERVAL = 100
DISABLE_BAROSTAT_DURING_PRODUCTION = True
Q_PROFILE_MAX_POINTS = 60000
Q_PROFILE_FRAME_STRIDE = 5
TABULATED_POINTS = 1024
TRAPEZOID = getattr(np, "trapezoid", getattr(np, "trapz", None))
ENABLE_SOLUTE_POLARIZATION = True
FAST_ACTIVE_DENSITY = True
SOLUTE_DPOLAR_BY_ELEMENT = {
    "O": 0.012,
    "N": 0.012,
    "C": 0.0,
    "S": 0.015,
    "H": 0.0,
}
SOLUTE_IS_POLAR_BY_ELEMENT = {
    "O": 1.0,
    "N": 1.0,
    "C": 0.0,
    "S": 1.0,
    "H": 0.0,
}
CHARGED_SIDECHAIN_POLAR_ATOMS = {
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "HIS": {"ND1", "NE2"},
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

def assign_force_groups(system):
    for idx, force in enumerate(system.getForces()):
        force.setForceGroup(min(idx, 31))

def snapshot_force_energies(label, system, topology, dcd_file):
    if not os.path.exists(dcd_file):
        return {}

    md_top = md.Topology.from_openmm(topology)
    traj = md.load(dcd_file, top=md_top)
    if traj.n_frames == 0:
        return {}

    assign_force_groups(system)
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = mm.Platform.getPlatformByName('CPU')
    sim = app.Simulation(topology, system, integrator, platform)
    sim.context.setPositions(traj.xyz[-1] * unit.nanometers)
    if traj.unitcell_vectors is not None:
        a, b, c = traj.unitcell_vectors[-1]
        sim.context.setPeriodicBoxVectors(
            mm.Vec3(*a) * unit.nanometers,
            mm.Vec3(*b) * unit.nanometers,
            mm.Vec3(*c) * unit.nanometers,
        )

    energies = {}
    for idx, force in enumerate(system.getForces()):
        group = force.getForceGroup()
        state = sim.context.getState(getEnergy=True, groups={group})
        key = f"{idx}:{force.__class__.__name__}"
        energies[key] = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    del sim, integrator
    return energies

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

class IntervalSpeedReporter:
    def __init__(self, file, report_interval):
        self._file = open(file, "w")
        self._report_interval = report_interval
        self._last_step = None
        self._last_clock = None
        self._file.write("Step,Interval Speed (ns/day)\n")

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return {"steps": steps, "periodic": None, "include": []}

    def report(self, simulation, state):
        now = time.perf_counter()
        step = simulation.currentStep
        if self._last_step is None:
            speed = 0.0
        else:
            elapsed = now - self._last_clock
            dt_ns = DT.value_in_unit(unit.nanoseconds)
            simulated_ns = (step - self._last_step) * dt_ns
            speed = simulated_ns * 86400.0 / elapsed if elapsed > 0 else 0.0
        self._file.write(f"{step},{speed:.3f}\n")
        self._file.flush()
        self._last_step = step
        self._last_clock = now

    def __del__(self):
        try:
            self._file.close()
        except Exception:
            pass

class FastQUpdateReporter:
    def __init__(self, force, meta, a_q2, update_interval):
        self.force = force
        self.meta = meta
        self.a_q2 = a_q2
        self.update_interval = update_interval

    def describeNextReport(self, simulation):
        steps = self.update_interval - simulation.currentStep % self.update_interval
        return {"steps": steps, "periodic": None, "include": ["positions"]}

    def report(self, simulation, state):
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        box = state.getPeriodicBoxVectors(asNumpy=True)
        box_nm = None
        if box is not None:
            box_nm = box.value_in_unit(unit.nanometers)
        q_eff = compute_fast_cwld_q_from_positions(pos, box_nm, self.meta, self.a_q2)
        for i, q in enumerate(q_eff):
            self.force.setParticleParameters(i, [float(self.meta["qbase"][i]), float(q)])
        self.force.updateParametersInContext(simulation.context)

def ca_water_first_peak(ca_water_rdf):
    r, g_r = ca_water_rdf
    if len(r) == 0 or len(g_r) == 0:
        return np.nan, np.nan

    mask = (r >= 0.18) & (r <= 0.35)
    if not np.any(mask):
        return np.nan, np.nan

    local_idx = np.argmax(g_r[mask])
    return float(r[mask][local_idx]), float(g_r[mask][local_idx])

def build_cwld_particle_arrays(system, topology, dpolar_O=-0.15):
    qbase = extract_nonbonded_charges(system)
    n_atoms = len(qbase)
    dpolar = np.zeros(n_atoms)
    is_polar = np.zeros(n_atoms)
    dens_source = np.zeros(n_atoms)
    mol_ids = -np.ones(n_atoms)
    water_mask = np.zeros(n_atoms, dtype=bool)
    water_o_indices = []
    protein_heavy_indices = []

    for residue in topology.residues():
        atom_indices = [atom.index for atom in residue.atoms()]
        mol_ids[atom_indices] = residue.index

        if residue.name in WATER_RESNAMES:
            water_mask[atom_indices] = True
            o_atom = next((a for a in residue.atoms() if a.element.symbol == "O"), None)
            h_atoms = [a for a in residue.atoms() if a.element.symbol == "H"]
            if o_atom is not None:
                water_o_indices.append(o_atom.index)
            if o_atom is not None and len(h_atoms) == 2:
                dpolar[o_atom.index] = dpolar_O
                dpolar[h_atoms[0].index] = -0.5 * dpolar_O
                dpolar[h_atoms[1].index] = -0.5 * dpolar_O
                is_polar[atom_indices] = 1.0
                dens_source[o_atom.index] = 1.0
        elif residue.name in ION_RESNAME_ALIASES.values():
            dens_source[atom_indices] = 1.0
            continue
        else:
            if residue.name in AMINO_ACIDS:
                protein_heavy_indices.extend(
                    atom.index for atom in residue.atoms()
                    if atom.element is not None and atom.element.symbol != "H"
                )
            if ENABLE_SOLUTE_POLARIZATION:
                for atom in residue.atoms():
                    elem = atom.element.symbol if atom.element is not None else "C"
                    is_charged_site = atom.name in CHARGED_SIDECHAIN_POLAR_ATOMS.get(residue.name, set())
                    is_active_heavy = elem in ("O", "N", "S") or is_charged_site
                    if FAST_ACTIVE_DENSITY and not is_active_heavy:
                        dpolar[atom.index] = 0.0
                        is_polar[atom.index] = 0.0
                        dens_source[atom.index] = 0.0
                    else:
                        dpolar[atom.index] = SOLUTE_DPOLAR_BY_ELEMENT.get(elem, 0.0)
                        is_polar[atom.index] = SOLUTE_IS_POLAR_BY_ELEMENT.get(elem, 0.0)
                        dens_source[atom.index] = 1.0 if elem != "H" else 0.0

    qref2 = np.mean(qbase[water_mask]**2) if np.any(water_mask) else np.mean(qbase**2)
    return qbase, dpolar, is_polar, dens_source, mol_ids, qref2, water_o_indices, protein_heavy_indices

def build_fast_cwld_metadata(system, topology, dpolar_O=-0.15):
    qbase = extract_nonbonded_charges(system)
    n_atoms = len(qbase)
    dpolar = np.zeros(n_atoms)
    is_polar = np.zeros(n_atoms)
    dens_source = np.zeros(n_atoms)
    q_driver = -np.ones(n_atoms, dtype=int)
    mol_ids = -np.ones(n_atoms)
    water_mask = np.zeros(n_atoms, dtype=bool)

    for residue in topology.residues():
        atom_indices = [atom.index for atom in residue.atoms()]
        mol_ids[atom_indices] = residue.index

        if residue.name in WATER_RESNAMES:
            water_mask[atom_indices] = True
            o_atom = next((a for a in residue.atoms() if a.element.symbol == "O"), None)
            h_atoms = [a for a in residue.atoms() if a.element.symbol == "H"]
            if o_atom is not None and len(h_atoms) == 2:
                dpolar[o_atom.index] = dpolar_O
                dpolar[h_atoms[0].index] = -0.5 * dpolar_O
                dpolar[h_atoms[1].index] = -0.5 * dpolar_O
                is_polar[[o_atom.index, h_atoms[0].index, h_atoms[1].index]] = 1.0
                dens_source[o_atom.index] = 1.0
                q_driver[[o_atom.index, h_atoms[0].index, h_atoms[1].index]] = o_atom.index

        elif residue.name in ION_RESNAME_ALIASES.values():
            dens_source[atom_indices] = 1.0

        elif ENABLE_SOLUTE_POLARIZATION:
            for atom in residue.atoms():
                elem = atom.element.symbol if atom.element is not None else "C"
                is_charged_site = atom.name in CHARGED_SIDECHAIN_POLAR_ATOMS.get(residue.name, set())
                is_active_heavy = elem in ("O", "N", "S") or is_charged_site
                if FAST_ACTIVE_DENSITY and not is_active_heavy:
                    continue
                dpolar[atom.index] = SOLUTE_DPOLAR_BY_ELEMENT.get(elem, 0.0)
                is_polar[atom.index] = SOLUTE_IS_POLAR_BY_ELEMENT.get(elem, 0.0)
                dens_source[atom.index] = 1.0 if elem != "H" else 0.0
                if is_polar[atom.index] > 0.0:
                    q_driver[atom.index] = atom.index

    qref2 = np.mean(qbase[water_mask]**2) if np.any(water_mask) else np.mean(qbase**2)
    return {
        "qbase": qbase,
        "qbase_sq": qbase**2,
        "dpolar": dpolar,
        "is_polar": is_polar,
        "dens_source": dens_source,
        "q_driver": q_driver,
        "mol_ids": mol_ids,
        "qref2": qref2,
    }

def compute_fast_cwld_q_from_positions(positions_nm, box_vectors_nm, meta, a_q2, r_env=0.35, rho0=13.5, k_polar=0.8):
    qbase = meta["qbase"]
    q_eff = qbase.copy()
    driver_atoms = np.where(meta["q_driver"] >= 0)[0]
    source_atoms = np.where(meta["dens_source"] > 0.0)[0]
    if len(driver_atoms) == 0 or len(source_atoms) == 0:
        return q_eff

    if box_vectors_nm is not None:
        box_lengths = np.array([box_vectors_nm[0][0], box_vectors_nm[1][1], box_vectors_nm[2][2]], dtype=float)
    else:
        box_lengths = None

    if box_lengths is not None and np.all(np.isfinite(box_lengths)) and np.all(box_lengths > 0.0):
        coords = np.mod(positions_nm, box_lengths)
        tree = cKDTree(coords[source_atoms], boxsize=box_lengths)
    else:
        coords = positions_nm
        tree = cKDTree(coords[source_atoms])

    driver_dens = {}
    for atom_idx in np.unique(meta["q_driver"][driver_atoms]):
        o_pos = coords[atom_idx]
        local_source_pos = tree.query_ball_point(o_pos, r_env)
        dens = 0.0
        for local_j in local_source_pos:
            j = source_atoms[local_j]
            if j == atom_idx or abs(meta["mol_ids"][atom_idx] - meta["mol_ids"][j]) <= 0.5:
                continue
            delta = coords[j] - o_pos
            if box_lengths is not None and np.all(np.isfinite(box_lengths)) and np.all(box_lengths > 0.0):
                delta -= box_lengths * np.round(delta / box_lengths)
            r = np.linalg.norm(delta)
            if 0.0 < r < r_env:
                x = r / r_env
                dens += (1.0 - x*x)**2 * (1.0 + a_q2 * (meta["qbase_sq"][j] - meta["qref2"]))
        driver_dens[atom_idx] = dens

    for i in driver_atoms:
        dens = driver_dens.get(meta["q_driver"][i], 0.0)
        q_eff[i] = qbase[i] + meta["is_polar"][i] * meta["dpolar"][i] * np.tanh(k_polar * dens / rho0)
    return q_eff

def compute_water_q_profile(traj, system, topology, a_q2, r_env=0.35, rho0=13.5, k_polar=0.8):
    """Recompute CWLD water-oxygen Q from saved coordinates for visualization."""
    qbase, dpolar, is_polar, dens_source, mol_ids, qref2, water_o_indices, protein_heavy_indices = build_cwld_particle_arrays(system, topology)
    if not water_o_indices or not protein_heavy_indices:
        return pd.DataFrame(columns=["frame", "distance_to_protein_nm", "water_oxygen_Qe", "dens"])

    q2 = qbase**2
    rows = []
    frame_indices = list(range(0, traj.n_frames, Q_PROFILE_FRAME_STRIDE))
    max_per_frame = max(1, Q_PROFILE_MAX_POINTS // max(1, len(frame_indices)))

    for frame_idx in frame_indices:
        xyz = traj.xyz[frame_idx]
        box = traj.unitcell_lengths[frame_idx] if traj.unitcell_lengths is not None else None
        if box is not None and np.all(np.isfinite(box)) and np.all(box > 0.0):
            coords = np.mod(xyz, box)
            tree = cKDTree(coords, boxsize=box)
            query_points = coords[water_o_indices]
        else:
            coords = xyz
            tree = cKDTree(coords)
            query_points = coords[water_o_indices]

        protein_tree = cKDTree(xyz[protein_heavy_indices])
        sampled_waters = water_o_indices
        if len(sampled_waters) > max_per_frame:
            pick = np.linspace(0, len(sampled_waters) - 1, max_per_frame, dtype=int)
            sampled_waters = [sampled_waters[i] for i in pick]

        for o_idx in sampled_waters:
            o_pos = coords[o_idx]
            neighbor_indices = tree.query_ball_point(o_pos, r_env)
            dens = 0.0
            for j in neighbor_indices:
                if j == o_idx or dens_source[j] <= 0.0 or abs(mol_ids[o_idx] - mol_ids[j]) <= 0.5:
                    continue
                delta = coords[j] - o_pos
                if box is not None and np.all(np.isfinite(box)) and np.all(box > 0.0):
                    delta -= box * np.round(delta / box)
                r = np.linalg.norm(delta)
                if 0.0 < r < r_env:
                    w_geom = (1.0 - (r / r_env)**2)**2
                    charge_mod = 1.0 + a_q2 * (q2[j] - qref2)
                    dens += w_geom * charge_mod

            q_eff = qbase[o_idx] + is_polar[o_idx] * dpolar[o_idx] * np.tanh(k_polar * dens / rho0)
            protein_dist, _ = protein_tree.query(xyz[o_idx], k=1)
            rows.append({
                "frame": frame_idx,
                "distance_to_protein_nm": float(protein_dist),
                "water_oxygen_Qe": float(q_eff),
                "dens": float(dens),
            })

    return pd.DataFrame(rows)

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
                                         nonbondedCutoff=1.0*unit.nanometer,
                                         constraints=app.HBonds,
                                         rigidWater=True)
    
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

def setup_cwld_lips_system(base_system, topology, r_env=0.35, rc=1.2, r_on=0.9, rho0=13.5, k_polar=0.8, dpolar_O=-0.15, k_penalty=180.0, a_q2=DEFAULT_A_Q2, include_baseline_ips=INCLUDE_BASELINE_IPS):
    print("\n" + "="*60)
    print(f"⚙️ [2/4] 正在注入 CWLD-L-IPS 隐式极化引擎 (a_q2={a_q2:.3f}, baseline_ips={include_baseline_ips})...")
    if USE_SELF_CONSISTENT_IPS_PAIR:
        print("  -> 使用自洽解析 IPS pair: target(Q) - shifted_cutoff(qbase)")
    print("="*60)
    # 序列化复制系统 (连带 test.py 里的 Restriction Force 和 Barostat 一起完美复制！)
    sys_copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(base_system))
    orig_nb = next(f for f in sys_copy.getForces() if isinstance(f, mm.NonbondedForce))
    
    ONE_4PI_EPS0 = 138.935458
    n_atoms = sys_copy.getNumParticles()
    qbase = np.zeros(n_atoms)
    dpolar = np.zeros(n_atoms)
    is_polar = np.zeros(n_atoms)
    dens_source = np.zeros(n_atoms)
    dens_sink = np.zeros(n_atoms)
    mol_ids = -np.ones(n_atoms) 
    water_mask = np.zeros(n_atoms, dtype=bool) 

    for i in range(n_atoms):
        q, _, _ = orig_nb.getParticleParameters(i)
        qbase[i] = q.value_in_unit(unit.elementary_charge)
    qbase_sq = qbase**2
        
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
                dens_source[o_atom.index] = 1.0
                dens_sink[atom_indices] = 1.0
                
        # 2. 处理离子：保留力场 qbase，强制不参与自身极化
        elif residue.name in ION_RESNAME_ALIASES.values():
            dpolar[atom_indices] = 0.0
            is_polar[atom_indices] = 0.0
            dens_source[atom_indices] = 1.0
            dens_sink[atom_indices] = 0.0

        # 3. 处理蛋白/有机溶质：按元素分配弱隐式极化响应
        else:
            if ENABLE_SOLUTE_POLARIZATION:
                for atom in residue.atoms():
                    elem = atom.element.symbol if atom.element is not None else "C"
                    is_charged_site = atom.name in CHARGED_SIDECHAIN_POLAR_ATOMS.get(residue.name, set())
                    is_active_heavy = elem in ("O", "N", "S") or is_charged_site
                    if FAST_ACTIVE_DENSITY and not is_active_heavy:
                        dpolar[atom.index] = 0.0
                        is_polar[atom.index] = 0.0
                        dens_source[atom.index] = 0.0
                        dens_sink[atom.index] = 0.0
                    else:
                        dpolar[atom.index] = SOLUTE_DPOLAR_BY_ELEMENT.get(elem, 0.0)
                        is_polar[atom.index] = SOLUTE_IS_POLAR_BY_ELEMENT.get(elem, 0.0)
                        dens_source[atom.index] = 1.0 if elem != "H" else 0.0
                        dens_sink[atom.index] = 1.0 if is_polar[atom.index] > 0.0 else 0.0
            else:
                dpolar[atom_indices] = 0.0
                is_polar[atom_indices] = 0.0
                dens_source[atom_indices] = 0.0
                dens_sink[atom_indices] = 0.0

    high_charge_atoms = np.where(np.abs(qbase) > 1.1)[0]
    if len(high_charge_atoms) > 0:
        print(f"  -> 检测到 {len(high_charge_atoms)} 个 |q|>1.1e 的高电荷原子/离子，按固定电荷环境源处理")
    if ENABLE_SOLUTE_POLARIZATION:
        n_solute_polar = np.sum((is_polar > 0.0) & (~water_mask))
        print(f"  -> 溶质弱极化已开启: {n_solute_polar} 个非水原子获得元素级 dpolar/is_polar")
    if FAST_ACTIVE_DENSITY:
        print(f"  -> Fast density mask: sources={np.sum(dens_source > 0.0)}, sinks={np.sum(dens_sink > 0.0)}, total={n_atoms}")

    # 引擎 A：原生 NonbondedForce
    orig_nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    orig_nb.setCutoffDistance(rc * unit.nanometer)
    orig_nb.setReactionFieldDielectric(1.0)
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
    lips.addGlobalParameter("include_baseline_ips", 1.0 if include_baseline_ips else 0.0)

    # 🛡️ 核心修正：强制仅使用【水分子】计算 qref2，防止蛋白和 Ca2+ 污染基线！
    if np.any(water_mask):
        qref2 = np.mean(qbase[water_mask]**2)
        print(f"  -> 🛡️ 动态锚定【纯溶剂】基线: qref2 = {qref2:.4f} (严格排除蛋白与 Ca2+ 干扰)")
    else:
        qref2 = np.mean(qbase**2)

    lips.addGlobalParameter("a_q2", a_q2)
    lips.addGlobalParameter("qref2", qref2)  

    lips.addPerParticleParameter("qbase")
    lips.addPerParticleParameter("qbase_sq")
    lips.addPerParticleParameter("dpolar")
    lips.addPerParticleParameter("is_polar")
    lips.addPerParticleParameter("dens_source")
    lips.addPerParticleParameter("dens_sink")
    lips.addPerParticleParameter("mol_id")

    tanh_xmax = 100.0
    x_tanh = np.linspace(0.0, tanh_xmax, TABULATED_POINTS)
    y_tanh = np.tanh(k_polar * x_tanh / rho0)
    lips.addTabulatedFunction("tanh_table", mm.Continuous1DFunction(y_tanh.tolist(), 0.0, tanh_xmax))

    x_density = np.linspace(0.0, rc, TABULATED_POINTS)
    y_density = np.zeros_like(x_density)
    density_mask = x_density < r_env
    x_density_valid = x_density[density_mask] / r_env
    y_density[density_mask] = (1.0 - x_density_valid**2)**2
    lips.addTabulatedFunction("density_kernel", mm.Continuous1DFunction(y_density.tolist(), 0.0, rc))

    if not USE_SELF_CONSISTENT_IPS_PAIR:
        x_ips = np.linspace(0.0, rc, TABULATED_POINTS)
        y_ips = np.zeros_like(x_ips)
        mask = (x_ips >= r_on) & (x_ips <= rc)
        r_valid = x_ips[mask]
        ips_core = 1.0/r_valid - 1.5/rc + 0.5*(r_valid**2)/(rc**3)
        x_switch = (r_valid - r_on) / (rc - r_on)
        smooth_switch = 1.0 - (1.0 - x_switch)**2 * (1.0 + 2.0*x_switch)
        y_ips[mask] = ips_core * smooth_switch
        lips.addTabulatedFunction("ips_table", mm.Continuous1DFunction(y_ips.tolist(), 0.0, rc))

    charge_mod = "(1.0 + a_q2 * (qbase_sq2 - qref2))"
    density_expr = f"dens_sink1 * dens_source2 * density_kernel(r) * {charge_mod} * step(abs(mol_id1 - mol_id2) - 0.5)"
    lips.addComputedValue("dens", density_expr, mm.CustomGBForce.ParticlePair)

    q_expr = "qbase + is_polar * dpolar * tanh_table(dens)"
    lips.addComputedValue("Q", q_expr, mm.CustomGBForce.SingleParticle)

    if USE_SELF_CONSISTENT_IPS_PAIR:
        pair_expr = """
        ONE_4PI_EPS0 * (
            (Q1*Q2 - qbase1*qbase2) / r
          + Q1*Q2 * (r^2/(2*rc^3) - 1.5/rc)
          + qbase1*qbase2 / rc
        );
        """
    else:
        pair_expr = """
        ONE_4PI_EPS0 * (
            (Q1*Q2 - qbase1*qbase2)
            + include_baseline_ips * qbase1*qbase2
        ) * ips_table(r);
        """
    lips.addEnergyTerm(pair_expr, mm.CustomGBForce.ParticlePair)

    self_expr = "0.5 * k_penalty * (Q - qbase)^2"
    lips.addEnergyTerm(self_expr, mm.CustomGBForce.SingleParticle)

    for i in range(n_atoms):
        lips.addParticle([
            float(qbase[i]),
            float(qbase_sq[i]),
            float(dpolar[i]),
            float(is_polar[i]),
            float(dens_source[i]),
            float(dens_sink[i]),
            float(mol_ids[i]),
        ])
        
    for exc in range(orig_nb.getNumExceptions()):
        p1, p2, _, _, _ = orig_nb.getExceptionParameters(exc)
        lips.addExclusion(p1, p2)
        
    sys_copy.addForce(lips)
    return sys_copy

def setup_cwld_fast_system(base_system, topology, rc=1.2, a_q2=DEFAULT_A_Q2):
    print("\n" + "="*60)
    print(f"⚙️ [2/4] 正在注入 CPU-KDTree Fast CWLD 引擎 (a_q2={a_q2:.3f}, update={Q_UPDATE_INTERVAL} steps)...")
    print("="*60)
    sys_copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(base_system))
    orig_nb = next(f for f in sys_copy.getForces() if isinstance(f, mm.NonbondedForce))
    orig_nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    orig_nb.setCutoffDistance(rc * unit.nanometer)
    orig_nb.setReactionFieldDielectric(1.0)
    orig_nb.setUseSwitchingFunction(True)
    orig_nb.setSwitchingDistance(0.9 * unit.nanometer)

    meta = build_fast_cwld_metadata(sys_copy, topology)
    print(
        f"  -> Fast Q metadata: sources={np.sum(meta['dens_source'] > 0.0)}, "
        f"drivers={np.sum(meta['q_driver'] >= 0)}, total={len(meta['qbase'])}"
    )

    expr = """
    ONE_4PI_EPS0 * (
        (Q1*Q2 - qbase1*qbase2) / r
      + Q1*Q2 * (r^2/(2*rc^3) - 1.5/rc)
      + qbase1*qbase2 / rc
    );
    """
    ips_force = mm.CustomNonbondedForce(expr)
    ips_force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    ips_force.setCutoffDistance(rc * unit.nanometer)
    ips_force.addGlobalParameter("ONE_4PI_EPS0", 138.935458)
    ips_force.addGlobalParameter("rc", rc)
    ips_force.addPerParticleParameter("qbase")
    ips_force.addPerParticleParameter("Q")

    for q in meta["qbase"]:
        ips_force.addParticle([float(q), float(q)])
    for exc in range(orig_nb.getNumExceptions()):
        p1, p2, _, _, _ = orig_nb.getExceptionParameters(exc)
        ips_force.addExclusion(p1, p2)

    sys_copy.addForce(ips_force)
    updater = FastQUpdateReporter(ips_force, meta, a_q2, Q_UPDATE_INTERVAL)
    return sys_copy, updater

def run_multistage_md(label, system, topology, positions, extra_reporters=None):
    print(f"\n[🚀 模拟] 启动 {label} 多阶段平行试验...")
    label_clean = label.replace(" ", "_")
    dcd_file = f"{label_clean}_1ckk.dcd"
    csv_file = f"{label_clean}_1ckk.csv"
    speed_file = f"{label_clean}_speed.csv"

    if os.path.exists(dcd_file) and os.path.exists(csv_file):
        print(f"  -> 检测到已有 {label_clean} 轨迹与 CSV，跳过模拟直接分析。")
        return dcd_file, csv_file
    
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
    sim.reporters.append(app.StateDataReporter(csv_file, REPORT_INTERVAL,
                                               step=True,
                                               potentialEnergy=REPORT_POTENTIAL_ENERGY,
                                               temperature=True, volume=True,
                                               density=True, speed=True))
    sim.reporters.append(IntervalSpeedReporter(speed_file, REPORT_INTERVAL))
    if extra_reporters:
        sim.reporters.extend(extra_reporters)
                                               
    # 阶段 1：能量最小化
    print("  -> [1/4] 能量最小化...")
    sim.minimizeEnergy(tolerance=10.0, maxIterations=2000)
    
    # 阶段 2：NVT 平衡 (带限制，k=1000)
    print("  -> [2/4] NVT 平衡 (蛋白受限，Ca2+/水 自由松弛)...")
    sim.step(EQ_STEPS // 2)
    
    # 阶段 3：NPT 平衡 (带限制，k=1000)
    print("  -> [3/4] 注入 Barostat，切换到 NPT 平衡 (稳定体系密度)...")
    barostat = next((force for force in system.getForces() if isinstance(force, mm.MonteCarloBarostat)), None)
    if barostat is None:
        barostat = mm.MonteCarloBarostat(PRESSURE, TEMPERATURE, NPT_BAROSTAT_INTERVAL)
        system.addForce(barostat)
    else:
        barostat.setFrequency(NPT_BAROSTAT_INTERVAL)
    sim.context.reinitialize(preserveState=True)
    sim.step(EQ_STEPS // 2)
    
    # 阶段 4：解除限制，生产 MD
    if DISABLE_BAROSTAT_DURING_PRODUCTION and barostat is not None:
        print("  -> [4/4] 关闭 Barostat，固定 NPT 平衡后的盒子，开始 NVT 生产 MD...")
        barostat.setFrequency(0)
        sim.context.reinitialize(preserveState=True)
    else:
        print("  -> [4/4] 保持 NPT，开始无约束生产 MD...")
    sim.context.setParameter("k", 0.0) # 💡 完美复刻 test.py 的解除锁定逻辑！
    sim.step(PROD_STEPS)
    
    print(f"  -> 🏁 {label} 模拟顺利跑完！")
    del sim
    gc.collect()
    return dcd_file, csv_file

def analyze_1ckk(label, dcd_file, csv_file, topology, system, a_q2=None):
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
    if protein_heavy_atoms:
        aligned = traj[:]
        aligned.superpose(aligned, 0, atom_indices=protein_heavy_atoms)
        protein_rmsf = md.rmsf(aligned, aligned, 0, atom_indices=protein_heavy_atoms)
        mean_protein_rmsf = float(np.mean(protein_rmsf))
    else:
        protein_rmsf = np.array([])
        mean_protein_rmsf = np.nan
    
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
    ca_water_peak_r, ca_water_peak_g = ca_water_first_peak((r_water, gr_water))

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
        if TRAPEZOID is None:
            raise RuntimeError("NumPy does not provide trapezoid/trapz integration.")
        residence_tau_ps = TRAPEZOID(
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

    q_profile = compute_water_q_profile(traj, system, topology, a_q2) if a_q2 is not None else pd.DataFrame()

    metrics = {
        "mean_rmsd_nm": float(np.mean(rmsd)),
        "mean_protein_rmsf_nm": mean_protein_rmsf,
        "ca_water_peak_r_nm": ca_water_peak_r,
        "ca_water_peak_g": ca_water_peak_g,
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
        "protein_rmsf": (np.array(protein_heavy_atoms), protein_rmsf),
        "water_q_profile": q_profile,
        "surface_water_acf": (times_ps[:len(residence_acf)], residence_acf),
        "salt_bridge_rdf": (r_salt, gr_salt),
        "dipole": (times_ps[:len(dipole_magnitude)], dipole_magnitude, dipole_vectors),
    }
        
    return analysis

def main():
    # 1. 构建体系 (自带 Restriction Force)
    topology, sys_pme, positions = build_1ckk_system()
    
    # 2. A/B/C 消融：Full CWLD vs Pure Density vs PME
    if CWLD_ENGINE == "cpu_kdtree_fast":
        sys_full, updater_full = setup_cwld_fast_system(sys_pme, topology, a_q2=DEFAULT_A_Q2)
        sys_density, updater_density = setup_cwld_fast_system(sys_pme, topology, a_q2=PURE_DENSITY_A_Q2)
    else:
        sys_full = setup_cwld_lips_system(sys_pme, topology, a_q2=DEFAULT_A_Q2, include_baseline_ips=INCLUDE_BASELINE_IPS)
        sys_density = setup_cwld_lips_system(sys_pme, topology, a_q2=PURE_DENSITY_A_Q2, include_baseline_ips=INCLUDE_BASELINE_IPS)
        updater_full = None
        updater_density = None

    experiments = {
        "PME": {
            "label": "PME_Baseline",
            "system": sys_pme,
            "reporters": [],
            "a_q2": None,
            "include_baseline_ips": False,
            "color": "black",
            "linestyle": "-",
        },
        "Full_CWLD_aq0p5": {
            "label": "CWLD_Fast_Full_aq0p5" if CWLD_ENGINE == "cpu_kdtree_fast" else "CWLD_Full_aq0p5",
            "system": sys_full,
            "reporters": [updater_full] if updater_full is not None else [],
            "a_q2": DEFAULT_A_Q2,
            "include_baseline_ips": INCLUDE_BASELINE_IPS,
            "color": "purple",
            "linestyle": "--",
        },
        "Pure_Density_aq0": {
            "label": "CWLD_Fast_PureDensity_aq0" if CWLD_ENGINE == "cpu_kdtree_fast" else "CWLD_PureDensity_aq0",
            "system": sys_density,
            "reporters": [updater_density] if updater_density is not None else [],
            "a_q2": PURE_DENSITY_A_Q2,
            "include_baseline_ips": INCLUDE_BASELINE_IPS,
            "color": "teal",
            "linestyle": ":",
        },
    }

    analyses = {}
    energy_rows = []
    for name, cfg in experiments.items():
        dcd_file, csv_file = run_multistage_md(cfg["label"], cfg["system"], topology, positions, extra_reporters=cfg["reporters"])
        analyses[name] = analyze_1ckk(name, dcd_file, csv_file, topology, cfg["system"], a_q2=cfg["a_q2"])
        force_energies = snapshot_force_energies(name, cfg["system"], topology, dcd_file)
        for force_name, energy in force_energies.items():
            energy_rows.append({"group": name, "force": force_name, "energy_kj_mol": energy})

    metrics_df = pd.DataFrame({name: ana["metrics"] for name, ana in analyses.items()}).T
    for name, cfg in experiments.items():
        metrics_df.loc[name, "a_q2"] = np.nan if cfg["a_q2"] is None else cfg["a_q2"]
        metrics_df.loc[name, "include_baseline_ips"] = float(cfg["include_baseline_ips"])
    metrics_df.to_csv("1ckk_v23_charge_mod_ablation_metrics.csv")
    if energy_rows:
        pd.DataFrame(energy_rows).to_csv("1ckk_v23_force_group_energies.csv", index=False)
    
    print("\n" + "="*60)
    print("📊 V2.3 charge_mod 消融结果:")
    print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("  -> 注：蛋白偶极矩后处理使用 qbase 与构象坐标；未直接读取 CWLD 内部瞬时 Q。")
    print("  -> 注：带净电荷蛋白的偶极矩依赖参考原点，当前以每帧蛋白几何中心为原点。")
    print("  -> ✅ 分析指标已保存至: 1ckk_v23_charge_mod_ablation_metrics.csv")
    print("="*60)
    
    # Ca2+ - water oxygen RDF: 直接检验 a_q2 是否让 Ca2+ 成为强扰动源
    plt.figure(figsize=(8, 5))
    rdf_rows = []
    for name, cfg in experiments.items():
        r, gr = analyses[name]["ca_water"]
        if len(r) == 0:
            continue
        plt.plot(r, gr, label=name, color=cfg["color"], lw=2, ls=cfg["linestyle"])
        rdf_rows.append(pd.DataFrame({"group": name, "r_nm": r, "g_r": gr}))
    if rdf_rows:
        pd.concat(rdf_rows, ignore_index=True).to_csv("1ckk_v23_ca_water_rdf.csv", index=False)
        plt.title('V2.3 Ablation: Ca2+ - Water Oxygen RDF')
        plt.xlabel('r (nm)')
        plt.ylabel('g(r)')
        plt.xlim(0.1, 0.5)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('1ckk_v23_ca_water_rdf_ablation.png', dpi=300)
        print("✅ A/B/C Ca-water RDF 已保存至: 1ckk_v23_ca_water_rdf_ablation.png")

    # Protein RMSF: 检查 charge_mod 是否导致构象非物理锁死或柔性塌缩
    plt.figure(figsize=(8, 5))
    rmsf_rows = []
    for name, cfg in experiments.items():
        atom_indices, rmsf = analyses[name]["protein_rmsf"]
        if len(rmsf) == 0:
            continue
        x = np.arange(len(rmsf))
        plt.plot(x, rmsf, label=name, color=cfg["color"], lw=1.5, ls=cfg["linestyle"])
        rmsf_rows.append(pd.DataFrame({"group": name, "protein_heavy_atom_order": x, "atom_index": atom_indices, "rmsf_nm": rmsf}))
    if rmsf_rows:
        pd.concat(rmsf_rows, ignore_index=True).to_csv("1ckk_v23_protein_rmsf.csv", index=False)
        plt.title('V2.3 Ablation: Protein Heavy-Atom RMSF')
        plt.xlabel('Protein heavy atom order')
        plt.ylabel('RMSF (nm)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('1ckk_v23_protein_rmsf_ablation.png', dpi=300)
        print("✅ A/B/C 蛋白 RMSF 已保存至: 1ckk_v23_protein_rmsf_ablation.png")

    # Water oxygen effective Q profile: 可视化空间依赖局部极化响应
    q_frames = []
    for name in ("Full_CWLD_aq0p5", "Pure_Density_aq0"):
        q_df = analyses[name]["water_q_profile"].copy()
        if q_df.empty:
            continue
        q_df["group"] = name
        q_frames.append(q_df)
    if q_frames:
        q_all = pd.concat(q_frames, ignore_index=True)
        q_all.to_csv("1ckk_v23_water_oxygen_q_profile.csv", index=False)
        fig, axes = plt.subplots(1, len(q_frames), figsize=(7 * len(q_frames), 5), squeeze=False)
        for ax, (name, q_df) in zip(axes.ravel(), [(df["group"].iloc[0], df) for df in q_frames]):
            mask = q_df["distance_to_protein_nm"].between(0.0, 2.0)
            hb = ax.hexbin(
                q_df.loc[mask, "distance_to_protein_nm"],
                q_df.loc[mask, "water_oxygen_Qe"],
                gridsize=70,
                mincnt=1,
                cmap="viridis",
            )
            ax.axhline(-0.834, color="white", lw=1.5, ls="--", label="TIP3P O baseline")
            ax.set_title(f'{name}: Water O effective Q')
            ax.set_xlabel('Distance to nearest protein heavy atom (nm)')
            ax.set_ylabel('Water oxygen Q (e)')
            ax.set_xlim(0.0, 2.0)
            ax.legend()
            fig.colorbar(hb, ax=ax, label='count')
        plt.tight_layout()
        plt.savefig("1ckk_v23_water_q_profile.png", dpi=300)
        print("✅ 水氧有效电荷 Q 剖面已保存至: 1ckk_v23_water_q_profile.png")

if __name__ == "__main__":
    main()
