import openmm as mm
from openmm import app, unit
try:
    from openmm.mtsintegrator import MTSLangevinIntegrator
except Exception:
    MTSLangevinIntegrator = None
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
CHARGE_MOD_MIN = 0.25
CHARGE_MOD_MAX = 2.5
CWLD_ENGINE = "customgb_exact"  # "customgb_exact" or "cpu_kdtree_fast"
Q_UPDATE_INTERVAL = 50
REPORT_POTENTIAL_ENERGY = False
NPT_BAROSTAT_INTERVAL = 100
DISABLE_BAROSTAT_DURING_PRODUCTION = True
Q_PROFILE_MAX_POINTS = 60000
Q_PROFILE_FRAME_STRIDE = 5
TABULATED_POINTS = 1024
BASE_GROUP = 0
CWLD_GROUP = 1
USE_MTS = False
MTS_RATIO = 1  # allowed: 1, 2, 4
RUN_MTS_VALIDATION_MATRIX = False
MTS_VALIDATION_RATIOS = (1, 2, 4)
ENABLE_FORCE_NORM_REPORTER = RUN_MTS_VALIDATION_MATRIX
FORCE_NORM_WARN_RMS_RATIO = 0.2
FORCE_NORM_WARN_MAX_RATIO = 1.0
TRAPEZOID = getattr(np, "trapezoid", getattr(np, "trapz", None))
ENABLE_SOLUTE_POLARIZATION = True
ENABLE_WATER_RESPONSE = True
FAST_ACTIVE_DENSITY = True
Q_DELTA_CLAMP = 0.20
ENABLE_Q_PENALTY = False
WATER_SOURCE_WEIGHT = 0.5
ION_SOURCE_WEIGHT = 2.0
POLAR_SOURCE_WEIGHT = 1.0
CHARGED_SOURCE_WEIGHT = 1.5
LIGAND_CHARGED_SOURCE_WEIGHT = 1.5
LIPID_HEADGROUP_SOURCE_WEIGHT = 0.3
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
LIPID_RESNAMES = {
    "POPC", "POPE", "POPG", "POPS", "POPA", "DPPC", "DOPC", "DOPE", "DOPS", "DOPG",
    "DLPC", "DMPC", "DSPC", "PIP", "PIP2", "PIP3", "PI", "PSM", "SM", "CER",
    "CHL", "CHOL", "CLR",
}
PHASE_INTERFACE_RULES = {
    "protein_ligand": (0.60, 0.90),
    "ion": (0.35, 0.60),
    "headgroup": (0.50, 0.80),
}

def normalize_ion_resnames(topology):
    for res in topology.residues():
        res.name = ION_RESNAME_ALIASES.get(res.name, res.name)

def is_lipid_headgroup_atom(atom):
    elem = atom.element.symbol if atom.element is not None else ""
    if atom.residue.name not in LIPID_RESNAMES:
        return False
    return elem in {"O", "N", "P", "S"}

def smooth_interface_weight(distance_nm, r_inner, r_outer):
    distance_nm = np.asarray(distance_nm, dtype=float)
    weight = np.zeros_like(distance_nm)
    weight[distance_nm <= r_inner] = 1.0
    shell = (distance_nm > r_inner) & (distance_nm < r_outer)
    if np.any(shell):
        x = (distance_nm[shell] - r_inner) / (r_outer - r_inner)
        weight[shell] = 1.0 - (x*x*x * (10.0 - 15.0*x + 6.0*x*x))
    return weight

def nearest_distance_weights(coords, target_indices, ref_indices, box_lengths, r_inner, r_outer):
    weights = np.zeros(len(target_indices))
    if len(target_indices) == 0 or len(ref_indices) == 0:
        return weights

    if box_lengths is not None and np.all(np.isfinite(box_lengths)) and np.all(box_lengths > 0.0):
        tree = cKDTree(coords[ref_indices], boxsize=box_lengths)
    else:
        tree = cKDTree(coords[ref_indices])
    distances, _ = tree.query(coords[target_indices], k=1, distance_upper_bound=r_outer)
    finite = np.isfinite(distances)
    weights[finite] = smooth_interface_weight(distances[finite], r_inner, r_outer)
    return weights

def build_phase_cwld_metadata(system, topology, dpolar_O=-0.15, a_q2=DEFAULT_A_Q2, enable_water_response=ENABLE_WATER_RESPONSE):
    qbase = extract_nonbonded_charges(system)
    n_atoms = len(qbase)
    dpolar = np.zeros(n_atoms)
    is_polar = np.zeros(n_atoms)
    dens_source = np.zeros(n_atoms)
    dens_sink = np.zeros(n_atoms)
    source_class_weight = np.zeros(n_atoms)
    static_phase = np.zeros(n_atoms)
    phase_ref_protein_ligand = np.zeros(n_atoms)
    phase_ref_ion = np.zeros(n_atoms)
    phase_ref_headgroup = np.zeros(n_atoms)
    q_driver = -np.ones(n_atoms, dtype=int)
    mol_ids = -np.ones(n_atoms)
    water_mask = np.zeros(n_atoms, dtype=bool)
    water_o_indices = []
    water_h_indices = []
    protein_heavy_indices = []
    ligand_heavy_indices = []
    ion_indices = []
    lipid_headgroup_indices = []
    lipid_tail_indices = []
    charged_site_indices = []

    for residue in topology.residues():
        atom_indices = [atom.index for atom in residue.atoms()]
        mol_ids[atom_indices] = residue.index
        residue_name = residue.name

        if residue_name in WATER_RESNAMES:
            water_mask[atom_indices] = True
            o_atom = next((a for a in residue.atoms() if a.element is not None and a.element.symbol == "O"), None)
            h_atoms = [a for a in residue.atoms() if a.element is not None and a.element.symbol == "H"]
            if o_atom is not None:
                water_o_indices.append(o_atom.index)
            if o_atom is not None and len(h_atoms) == 2:
                water_h_indices.extend([h_atoms[0].index, h_atoms[1].index])
                dens_source[o_atom.index] = 1.0
                source_class_weight[o_atom.index] = WATER_SOURCE_WEIGHT
                if enable_water_response:
                    dpolar[o_atom.index] = dpolar_O
                    dpolar[h_atoms[0].index] = -0.5 * dpolar_O
                    dpolar[h_atoms[1].index] = -0.5 * dpolar_O
                    is_polar[[o_atom.index, h_atoms[0].index, h_atoms[1].index]] = 1.0
                    dens_sink[[o_atom.index, h_atoms[0].index, h_atoms[1].index]] = 1.0
                    static_phase[[o_atom.index, h_atoms[0].index, h_atoms[1].index]] = 1.0
                    q_driver[[o_atom.index, h_atoms[0].index, h_atoms[1].index]] = o_atom.index

        elif residue_name in ION_RESNAME_ALIASES.values():
            dens_source[atom_indices] = 1.0
            source_class_weight[atom_indices] = ION_SOURCE_WEIGHT
            phase_ref_ion[atom_indices] = 1.0
            ion_indices.extend(atom_indices)

        elif residue_name in LIPID_RESNAMES:
            for atom in residue.atoms():
                elem = atom.element.symbol if atom.element is not None else "C"
                if is_lipid_headgroup_atom(atom):
                    dens_source[atom.index] = 1.0
                    source_class_weight[atom.index] = LIPID_HEADGROUP_SOURCE_WEIGHT
                    phase_ref_headgroup[atom.index] = 1.0
                    lipid_headgroup_indices.append(atom.index)
                elif elem in {"C", "H"}:
                    lipid_tail_indices.append(atom.index)

        else:
            is_protein = residue_name in AMINO_ACIDS
            if is_protein:
                protein_heavy_indices.extend(
                    atom.index for atom in residue.atoms()
                    if atom.element is not None and atom.element.symbol != "H"
                )
            else:
                ligand_heavy_indices.extend(
                    atom.index for atom in residue.atoms()
                    if atom.element is not None and atom.element.symbol != "H"
                )

            for atom in residue.atoms():
                elem = atom.element.symbol if atom.element is not None else "C"
                if elem != "H":
                    phase_ref_protein_ligand[atom.index] = 1.0

            if ENABLE_SOLUTE_POLARIZATION:
                for atom in residue.atoms():
                    elem = atom.element.symbol if atom.element is not None else "C"
                    is_charged_site = atom.name in CHARGED_SIDECHAIN_POLAR_ATOMS.get(residue_name, set())
                    is_active_heavy = elem in ("O", "N", "S") or is_charged_site
                    if is_charged_site:
                        charged_site_indices.append(atom.index)
                    if FAST_ACTIVE_DENSITY and not is_active_heavy:
                        continue
                    dpolar[atom.index] = SOLUTE_DPOLAR_BY_ELEMENT.get(elem, 0.0)
                    is_polar[atom.index] = SOLUTE_IS_POLAR_BY_ELEMENT.get(elem, 0.0)
                    dens_source[atom.index] = 1.0 if elem != "H" else 0.0
                    if dens_source[atom.index] > 0.0:
                        if is_charged_site:
                            source_class_weight[atom.index] = CHARGED_SOURCE_WEIGHT if is_protein else LIGAND_CHARGED_SOURCE_WEIGHT
                        else:
                            source_class_weight[atom.index] = POLAR_SOURCE_WEIGHT
                    dens_sink[atom.index] = 1.0 if is_polar[atom.index] > 0.0 else 0.0
                    if is_polar[atom.index] > 0.0:
                        static_phase[atom.index] = 1.0
                        q_driver[atom.index] = atom.index

    qref2, _ = active_water_source_qref2(qbase, water_mask, dens_source)
    charge_mod_array = clamp_charge_mod(1.0 + a_q2 * (qbase**2 - qref2))
    return {
        "qbase": qbase,
        "qbase_sq": qbase**2,
        "charge_mod_array": charge_mod_array,
        "dpolar": dpolar,
        "is_polar": is_polar,
        "dens_source": dens_source,
        "dens_sink": dens_sink,
        "source_class_weight": source_class_weight,
        "static_phase": static_phase,
        "phase_ref_protein_ligand": phase_ref_protein_ligand,
        "phase_ref_ion": phase_ref_ion,
        "phase_ref_headgroup": phase_ref_headgroup,
        "q_driver": q_driver,
        "mol_ids": mol_ids,
        "qref2": qref2,
        "water_mask": water_mask,
        "water_o_indices": np.array(water_o_indices, dtype=int),
        "water_h_indices": np.array(water_h_indices, dtype=int),
        "protein_heavy_indices": np.array(protein_heavy_indices, dtype=int),
        "ligand_heavy_indices": np.array(ligand_heavy_indices, dtype=int),
        "ion_indices": np.array(ion_indices, dtype=int),
        "lipid_headgroup_indices": np.array(lipid_headgroup_indices, dtype=int),
        "lipid_tail_indices": np.array(lipid_tail_indices, dtype=int),
        "charged_site_indices": np.array(charged_site_indices, dtype=int),
    }

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

def assign_cwld_mts_force_groups(system):
    for force in system.getForces():
        if isinstance(force, mm.CustomGBForce):
            force.setForceGroup(CWLD_GROUP)
        else:
            force.setForceGroup(BASE_GROUP)

def make_2to1_mts_integrator():
    if MTSLangevinIntegrator is None:
        raise RuntimeError("MTSLangevinIntegrator is not available in this OpenMM installation.")
    return MTSLangevinIntegrator(
        TEMPERATURE,
        1.0 / unit.picosecond,
        0.004 * unit.picoseconds,
        [(CWLD_GROUP, 1), (BASE_GROUP, 2)],
    )

def make_4to1_mts_integrator():
    if MTSLangevinIntegrator is None:
        raise RuntimeError("MTSLangevinIntegrator is not available in this OpenMM installation.")
    return MTSLangevinIntegrator(
        TEMPERATURE,
        1.0 / unit.picosecond,
        0.008 * unit.picoseconds,
        [(CWLD_GROUP, 1), (BASE_GROUP, 4)],
    )

def make_md_integrator(mts_ratio=None):
    mts_ratio = MTS_RATIO if mts_ratio is None else mts_ratio
    if mts_ratio == 1:
        return mm.LangevinMiddleIntegrator(TEMPERATURE, 1.0 / unit.picosecond, DT)
    if mts_ratio == 2:
        return make_2to1_mts_integrator()
    if mts_ratio == 4:
        return make_4to1_mts_integrator()
    raise ValueError("MTS_RATIO must be one of 1, 2, or 4.")

def scaled_step_count(reference_steps, integrator):
    effective_dt_ps = integrator.getStepSize().value_in_unit(unit.picoseconds)
    base_dt_ps = DT.value_in_unit(unit.picoseconds)
    if effective_dt_ps <= 0.0:
        raise ValueError("Integrator step size must be positive.")
    return max(1, int(round(reference_steps * base_dt_ps / effective_dt_ps)))

def scaled_report_interval(integrator):
    return scaled_step_count(REPORT_INTERVAL, integrator)

class ForceNormReporter:
    def __init__(self, file_path, report_interval):
        self.file_path = file_path
        self.report_interval = report_interval
        self._header_written = os.path.exists(file_path) and os.path.getsize(file_path) > 0

    def describeNextReport(self, simulation):
        steps = self.report_interval - simulation.currentStep % self.report_interval
        return {"steps": steps, "periodic": None, "include": []}

    def report(self, simulation, state):
        rows = []
        for group, name in ((BASE_GROUP, "base"), (CWLD_GROUP, "cwld")):
            group_state = simulation.context.getState(getForces=True, groups={group})
            forces = group_state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
            norms = np.linalg.norm(forces, axis=1)
            rows.append({
                "step": simulation.currentStep,
                "group": name,
                "force_rms": float(np.sqrt(np.mean(norms * norms))) if len(norms) else np.nan,
                "force_max": float(np.max(norms)) if len(norms) else np.nan,
            })
        base = rows[0]
        cwld = rows[1]
        base_rms = base["force_rms"]
        base_max = base["force_max"]
        rms_ratio = cwld["force_rms"] / base_rms if np.isfinite(base_rms) and base_rms > 0.0 else np.nan
        max_ratio = cwld["force_max"] / base_max if np.isfinite(base_max) and base_max > 0.0 else np.nan
        summary = {
            "step": simulation.currentStep,
            "base_force_rms": base["force_rms"],
            "base_force_max": base["force_max"],
            "cwld_force_rms": cwld["force_rms"],
            "cwld_force_max": cwld["force_max"],
            "cwld_rms_over_base": float(rms_ratio),
            "cwld_max_over_base": float(max_ratio),
            "mts_warning": int(
                (np.isfinite(rms_ratio) and rms_ratio > FORCE_NORM_WARN_RMS_RATIO)
                or (np.isfinite(max_ratio) and max_ratio > FORCE_NORM_WARN_MAX_RATIO)
            ),
        }
        pd.DataFrame([summary]).to_csv(self.file_path, mode="a", header=not self._header_written, index=False)
        self._header_written = True

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

def active_water_source_qref2(qbase, water_mask, dens_source):
    source_mask = water_mask & (dens_source > 0.0)
    if np.any(source_mask):
        return float(np.mean(qbase[source_mask]**2)), int(np.sum(source_mask))
    if np.any(water_mask):
        return float(np.mean(qbase[water_mask]**2)), int(np.sum(water_mask))
    return float(np.mean(qbase**2)), int(len(qbase))

def clamp_charge_mod(charge_mod):
    return np.clip(charge_mod, CHARGE_MOD_MIN, CHARGE_MOD_MAX)

def smooth_clamp_delta_q(delta_q):
    return Q_DELTA_CLAMP * np.tanh(delta_q / Q_DELTA_CLAMP)

def build_cwld_particle_arrays(system, topology, dpolar_O=-0.15, a_q2=DEFAULT_A_Q2, enable_water_response=ENABLE_WATER_RESPONSE):
    meta = build_phase_cwld_metadata(system, topology, dpolar_O, a_q2, enable_water_response)
    interface_refs = np.unique(np.concatenate([
        meta["protein_heavy_indices"],
        meta["ligand_heavy_indices"],
        meta["ion_indices"],
        meta["lipid_headgroup_indices"],
    ])).astype(int)
    return (
        meta["qbase"],
        meta["dpolar"],
        meta["is_polar"],
        meta["dens_source"],
        meta["mol_ids"],
        meta["qref2"],
        meta["water_o_indices"].tolist(),
        interface_refs.tolist(),
    )

def build_fast_cwld_metadata(system, topology, dpolar_O=-0.15, a_q2=DEFAULT_A_Q2, enable_water_response=ENABLE_WATER_RESPONSE):
    return build_phase_cwld_metadata(system, topology, dpolar_O, a_q2, enable_water_response)

def compute_fast_cwld_q_from_positions(positions_nm, box_vectors_nm, meta, a_q2, r_env=0.35, rho0=13.5, k_polar=0.8):
    qbase = meta["qbase"]
    q_eff = qbase.copy()
    source_atoms = np.where(meta["dens_source"] > 0.0)[0]
    if len(source_atoms) == 0:
        return q_eff

    if box_vectors_nm is not None:
        box_lengths = np.array([box_vectors_nm[0][0], box_vectors_nm[1][1], box_vectors_nm[2][2]], dtype=float)
    else:
        box_lengths = None

    has_periodic_box = box_lengths is not None and np.all(np.isfinite(box_lengths)) and np.all(box_lengths > 0.0)
    coords = np.mod(positions_nm, box_lengths) if has_periodic_box else positions_nm

    driver_phase = meta["static_phase"].copy()
    water_o_indices = meta["water_o_indices"]
    if len(water_o_indices) > 0:
        water_phase = np.zeros(len(water_o_indices))
        protein_ligand_refs = np.unique(np.concatenate([
            meta["protein_heavy_indices"],
            meta["ligand_heavy_indices"],
        ])).astype(int)
        for key, refs in (
            ("protein_ligand", protein_ligand_refs),
            ("ion", meta["ion_indices"]),
            ("headgroup", meta["lipid_headgroup_indices"]),
        ):
            r_inner, r_outer = PHASE_INTERFACE_RULES[key]
            water_phase = np.maximum(
                water_phase,
                nearest_distance_weights(coords, water_o_indices, refs, box_lengths, r_inner, r_outer),
            )
        driver_phase[water_o_indices] = np.maximum(driver_phase[water_o_indices], water_phase)

    all_driver_atoms = np.where(meta["q_driver"] >= 0)[0]
    active_driver_roots = all_driver_atoms[driver_phase[meta["q_driver"][all_driver_atoms]] > 0.0]
    if len(active_driver_roots) == 0:
        return q_eff

    if has_periodic_box:
        tree = cKDTree(coords[source_atoms], boxsize=box_lengths)
    else:
        tree = cKDTree(coords[source_atoms])

    driver_dens = {}
    for atom_idx in np.unique(meta["q_driver"][active_driver_roots]):
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
                dens += meta["source_class_weight"][j] * (1.0 - x*x)**2 * float(meta["charge_mod_array"][j])
        driver_dens[atom_idx] = dens

    for i in active_driver_roots:
        dens = driver_dens.get(meta["q_driver"][i], 0.0)
        phase = driver_phase[meta["q_driver"][i]]
        delta_q = phase * meta["is_polar"][i] * meta["dpolar"][i] * np.tanh(k_polar * dens / rho0)
        q_eff[i] = qbase[i] + smooth_clamp_delta_q(delta_q)
    return q_eff

def compute_water_q_profile(traj, system, topology, a_q2, r_env=0.35, rho0=13.5, k_polar=0.8, enable_water_response=ENABLE_WATER_RESPONSE, profile_mode="exact"):
    """Recompute CWLD water-oxygen Q from saved coordinates for visualization."""
    meta = build_phase_cwld_metadata(system, topology, a_q2=a_q2, enable_water_response=enable_water_response)
    qbase = meta["qbase"]
    dpolar = meta["dpolar"]
    is_polar = meta["is_polar"]
    dens_source = meta["dens_source"]
    source_class_weight = meta["source_class_weight"]
    charge_mod_array = meta["charge_mod_array"]
    mol_ids = meta["mol_ids"]
    static_phase = meta["static_phase"]
    qref2 = meta["qref2"]
    water_o_indices = meta["water_o_indices"].tolist()
    ca_indices = [
        atom.index for atom in topology.atoms()
        if atom.residue.name == "Ca2+"
    ]
    protein_heavy_indices = meta["protein_heavy_indices"].tolist()
    interface_ref_indices = np.unique(np.concatenate([
        meta["protein_heavy_indices"],
        meta["ligand_heavy_indices"],
        meta["ion_indices"],
        meta["lipid_headgroup_indices"],
    ])).astype(int).tolist()
    if not water_o_indices or not interface_ref_indices:
        return pd.DataFrame(columns=["frame", "distance_to_interface_nm", "distance_to_ca_nm", "water_oxygen_Qe", "dens", "s_phase"])

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

        interface_tree = cKDTree(xyz[interface_ref_indices])
        if ca_indices:
            ca_tree = cKDTree(coords[ca_indices], boxsize=box) if box is not None and np.all(np.isfinite(box)) and np.all(box > 0.0) else cKDTree(coords[ca_indices])
        else:
            ca_tree = None
        sampled_waters_all = np.array(water_o_indices, dtype=int)
        water_phase_by_o = {}
        water_phase = np.zeros(len(sampled_waters_all))
        protein_ligand_refs = np.unique(np.concatenate([
            meta["protein_heavy_indices"],
            meta["ligand_heavy_indices"],
        ])).astype(int)
        for key, refs in (
            ("protein_ligand", protein_ligand_refs),
            ("ion", meta["ion_indices"]),
            ("headgroup", meta["lipid_headgroup_indices"]),
        ):
            r_inner, r_outer = PHASE_INTERFACE_RULES[key]
            water_phase = np.maximum(
                water_phase,
                nearest_distance_weights(coords, sampled_waters_all, refs, box, r_inner, r_outer),
            )
        for atom_idx, phase in zip(sampled_waters_all, water_phase):
            water_phase_by_o[int(atom_idx)] = float(phase)

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
                    dens += source_class_weight[j] * w_geom * charge_mod_array[j]

            if profile_mode == "exact":
                s_phase = static_phase[o_idx]
            else:
                s_phase = water_phase_by_o.get(o_idx, 0.0)
            delta_q = s_phase * is_polar[o_idx] * dpolar[o_idx] * np.tanh(k_polar * dens / rho0)
            q_eff = qbase[o_idx] + smooth_clamp_delta_q(delta_q)
            interface_dist, _ = interface_tree.query(xyz[o_idx], k=1)
            ca_dist = np.nan
            if ca_tree is not None:
                ca_dist, _ = ca_tree.query(coords[o_idx], k=1)
            rows.append({
                "frame": frame_idx,
                "distance_to_interface_nm": float(interface_dist),
                "distance_to_ca_nm": float(ca_dist),
                "water_oxygen_Qe": float(q_eff),
                "dens": float(dens),
                "s_phase": float(s_phase),
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

def setup_cwld_lips_system(base_system, topology, r_env=0.35, rc=1.2, r_on=0.9, rho0=13.5, k_polar=0.8, dpolar_O=-0.15, k_penalty=180.0, a_q2=DEFAULT_A_Q2, enable_water_response=ENABLE_WATER_RESPONSE):
    print("\n" + "="*60)
    print(f"⚙️ [2/4] 正在注入 CWLD-L-IPS 隐式极化引擎 (a_q2={a_q2:.3f})...")
    print("  -> 使用自洽解析 IPS pair: target(Q) - shifted_cutoff(qbase)")
    print("  -> CustomGBForce 保留 dens→Q→U 的链式 dQ/dR 极化响应力")
    print("="*60)
    # 序列化复制系统 (连带 test.py 里的 Restriction Force 和 Barostat 一起完美复制！)
    sys_copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(base_system))
    orig_nb = next(f for f in sys_copy.getForces() if isinstance(f, mm.NonbondedForce))
    
    ONE_4PI_EPS0 = 138.935458
    n_atoms = sys_copy.getNumParticles()
    meta = build_phase_cwld_metadata(sys_copy, topology, dpolar_O, a_q2, enable_water_response)
    qbase = meta["qbase"]
    charge_mod_array = meta["charge_mod_array"]
    dpolar = meta["dpolar"]
    is_polar = meta["is_polar"]
    dens_source = meta["dens_source"]
    dens_sink = meta["dens_sink"]
    source_class_weight = meta["source_class_weight"]
    mol_ids = meta["mol_ids"]
    water_mask = meta["water_mask"]
    static_phase = meta["static_phase"]

    high_charge_atoms = np.where(np.abs(qbase) > 1.1)[0]
    if len(high_charge_atoms) > 0:
        print(f"  -> 检测到 {len(high_charge_atoms)} 个 |q|>1.1e 的高电荷原子/离子，按固定电荷环境源处理")
    if ENABLE_SOLUTE_POLARIZATION:
        n_solute_polar = np.sum((is_polar > 0.0) & (~water_mask))
        print(f"  -> 溶质弱极化已开启: {n_solute_polar} 个非水原子获得元素级 dpolar/is_polar")
    if FAST_ACTIVE_DENSITY:
        print(f"  -> Fast density mask: sources={np.sum(dens_source > 0.0)}, sinks={np.sum(dens_sink > 0.0)}, total={n_atoms}")
    print(f"  -> Water response: {'on' if enable_water_response else 'off/fixed-charge source-only'} (source_weight={WATER_SOURCE_WEIGHT:.2f})")
    for source_label, source_indices in (
        ("water_O", meta["water_o_indices"]),
        ("ions", meta["ion_indices"]),
        ("protein_ligand_refs", np.where(meta["phase_ref_protein_ligand"] > 0.0)[0]),
        ("lipid_head", meta["lipid_headgroup_indices"]),
    ):
        active = np.array([idx for idx in source_indices if dens_source[idx] > 0.0], dtype=int)
        if len(active) > 0:
            print(
                f"  -> charge_mod[{source_label}]: "
                f"N={len(active)}, mean={np.mean(charge_mod_array[active]):.3f}, "
                f"min={np.min(charge_mod_array[active]):.3f}, max={np.max(charge_mod_array[active]):.3f}"
            )
    print(
        "  -> Phase/interface mask: "
        f"water_O={len(meta['water_o_indices'])}, protein_ref={len(meta['protein_heavy_indices'])}, "
        f"ligand_ref={len(meta['ligand_heavy_indices'])}, ions={len(meta['ion_indices'])}, "
        f"lipid_head={len(meta['lipid_headgroup_indices'])}, lipid_tail={len(meta['lipid_tail_indices'])}"
    )

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
    if ENABLE_Q_PENALTY:
        lips.addGlobalParameter("k_penalty", k_penalty)

    # qref2 must match the active density source population. For TIP3P this is water O,
    # not the O+H+H average, otherwise ordinary water is artificially up-weighted.
    qref2, qref2_count = active_water_source_qref2(qbase, water_mask, dens_source)
    print(f"  -> 动态锚定 active water-source 基线: qref2 = {qref2:.4f} (N={qref2_count})")

    lips.addGlobalParameter("q_delta_clamp", Q_DELTA_CLAMP)

    lips.addPerParticleParameter("qbase")
    lips.addPerParticleParameter("charge_mod")
    lips.addPerParticleParameter("dpolar")
    lips.addPerParticleParameter("is_polar")
    lips.addPerParticleParameter("dens_source")
    lips.addPerParticleParameter("dens_sink")
    lips.addPerParticleParameter("source_class_weight")
    lips.addPerParticleParameter("static_phase")
    lips.addPerParticleParameter("mol_id")

    x_density = np.linspace(0.0, rc, TABULATED_POINTS)
    y_density = np.zeros_like(x_density)
    density_mask = x_density < r_env
    x_density_valid = x_density[density_mask] / r_env
    y_density[density_mask] = (1.0 - x_density_valid**2)**2
    lips.addTabulatedFunction("density_kernel", mm.Continuous1DFunction(y_density.tolist(), 0.0, rc))

    density_expr = "dens_sink1 * dens_source2 * source_class_weight2 * charge_mod2 * density_kernel(r) * step(abs(mol_id1 - mol_id2) - 0.5)"
    lips.addComputedValue("dens", density_expr, mm.CustomGBForce.ParticlePair)

    q_expr = f"qbase + q_delta_clamp * tanh((static_phase * is_polar * dpolar * tanh({k_polar:.16g}*dens/{rho0:.16g})) / q_delta_clamp)"
    lips.addComputedValue("Q", q_expr, mm.CustomGBForce.SingleParticle)

    pair_expr = """
    ONE_4PI_EPS0 * (
        (Q1*Q2 - qbase1*qbase2) / r
      + Q1*Q2 * (r^2/(2*rc^3) - 1.5/rc)
      + qbase1*qbase2 / rc
    );
    """
    lips.addEnergyTerm(pair_expr, mm.CustomGBForce.ParticlePair)

    if ENABLE_Q_PENALTY:
        self_expr = "0.5 * k_penalty * (Q - qbase)^2"
        lips.addEnergyTerm(self_expr, mm.CustomGBForce.SingleParticle)

    for i in range(n_atoms):
        lips.addParticle([
            float(qbase[i]),
            float(charge_mod_array[i]),
            float(dpolar[i]),
            float(is_polar[i]),
            float(dens_source[i]),
            float(dens_sink[i]),
            float(source_class_weight[i]),
            float(static_phase[i]),
            float(mol_ids[i]),
        ])
        
    for exc in range(orig_nb.getNumExceptions()):
        p1, p2, _, _, _ = orig_nb.getExceptionParameters(exc)
        lips.addExclusion(p1, p2)
        
    sys_copy.addForce(lips)
    return sys_copy

def setup_cwld_fast_system(base_system, topology, rc=1.2, a_q2=DEFAULT_A_Q2, enable_water_response=ENABLE_WATER_RESPONSE):
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

    meta = build_fast_cwld_metadata(sys_copy, topology, a_q2=a_q2, enable_water_response=enable_water_response)
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

def run_multistage_md(label, system, topology, positions, extra_reporters=None, mts_ratio=None):
    print(f"\n[🚀 模拟] 启动 {label} 多阶段平行试验...")
    label_clean = label.replace(" ", "_")
    dcd_file = f"{label_clean}_1ckk.dcd"
    csv_file = f"{label_clean}_1ckk.csv"
    force_norm_file = f"{label_clean}_force_norms.csv"

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

    mts_ratio = MTS_RATIO if mts_ratio is None else mts_ratio
    has_exact_cwld = any(isinstance(force, mm.CustomGBForce) for force in system.getForces())
    use_mts_for_run = (USE_MTS or RUN_MTS_VALIDATION_MATRIX) and mts_ratio in (2, 4) and has_exact_cwld
    if use_mts_for_run:
        assign_cwld_mts_force_groups(system)
        integrator = make_md_integrator(mts_ratio)
        print(f"  -> MTS enabled: ratio={mts_ratio}:1, base_group={BASE_GROUP}, cwld_group={CWLD_GROUP}")
    else:
        integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, 1.0/unit.picosecond, DT)
        if (USE_MTS or RUN_MTS_VALIDATION_MATRIX) and mts_ratio in (2, 4) and not has_exact_cwld:
            print("  -> MTS requested but no CustomGB exact force found; using ordinary LangevinMiddleIntegrator.")
    nvt_steps = scaled_step_count(EQ_STEPS // 2, integrator)
    npt_steps = scaled_step_count(EQ_STEPS // 2, integrator)
    prod_steps = scaled_step_count(PROD_STEPS, integrator)
    report_interval = scaled_report_interval(integrator)
    effective_dt_ps = integrator.getStepSize().value_in_unit(unit.picoseconds)
    print(
        f"  -> Integrator dt={effective_dt_ps:.4f} ps; "
        f"scaled steps: NVT={nvt_steps}, NPT={npt_steps}, PROD={prod_steps}, report={report_interval}"
    )
    sim = app.Simulation(topology, system, integrator, platform, properties)
    sim.context.setPositions(positions)
    
    sim.reporters.append(app.DCDReporter(dcd_file, report_interval))
    sim.reporters.append(app.StateDataReporter(csv_file, report_interval,
                                               step=True,
                                               time=True,
                                               potentialEnergy=REPORT_POTENTIAL_ENERGY,
                                               temperature=True, volume=True,
                                               density=True, speed=True))
    if extra_reporters:
        sim.reporters.extend(extra_reporters)
    if ENABLE_FORCE_NORM_REPORTER and has_exact_cwld:
        sim.reporters.append(ForceNormReporter(force_norm_file, report_interval))
                                               
    # 阶段 1：能量最小化
    print("  -> [1/4] 能量最小化...")
    sim.minimizeEnergy(tolerance=10.0, maxIterations=2000)
    
    # 阶段 2：NVT 平衡 (带限制，k=1000)
    print("  -> [2/4] NVT 平衡 (蛋白受限，Ca2+/水 自由松弛)...")
    sim.step(nvt_steps)
    
    # 阶段 3：NPT 平衡 (带限制，k=1000)
    print("  -> [3/4] 注入 Barostat，切换到 NPT 平衡 (稳定体系密度)...")
    barostat = next((force for force in system.getForces() if isinstance(force, mm.MonteCarloBarostat)), None)
    if barostat is None:
        barostat = mm.MonteCarloBarostat(PRESSURE, TEMPERATURE, NPT_BAROSTAT_INTERVAL)
        system.addForce(barostat)
    else:
        barostat.setFrequency(NPT_BAROSTAT_INTERVAL)
    sim.context.reinitialize(preserveState=True)
    sim.step(npt_steps)
    
    # 阶段 4：解除限制，生产 MD
    if DISABLE_BAROSTAT_DURING_PRODUCTION and barostat is not None:
        print("  -> [4/4] 关闭 Barostat，固定 NPT 平衡后的盒子，开始 NVT 生产 MD...")
        barostat.setFrequency(0)
        sim.context.reinitialize(preserveState=True)
    else:
        print("  -> [4/4] 保持 NPT，开始无约束生产 MD...")
    sim.context.setParameter("k", 0.0) # 💡 完美复刻 test.py 的解除锁定逻辑！
    sim.step(prod_steps)
    
    print(f"  -> 🏁 {label} 模拟顺利跑完！")
    del sim
    gc.collect()
    return dcd_file, csv_file

def analyze_1ckk(label, dcd_file, csv_file, topology, system, a_q2=None, force_norm_file=None, enable_water_response=ENABLE_WATER_RESPONSE, q_profile_mode="exact"):
    print(f"\n[📊 分析] 正在提取 {label} 的核心指标...")
    md_top = md.Topology.from_openmm(topology)
    traj = md.load(dcd_file, top=md_top)
    traj = traj.image_molecules(inplace=False)
    charges = extract_nonbonded_charges(system)
    if force_norm_file is None:
        force_norm_file = dcd_file.replace("_1ckk.dcd", "_force_norms.csv")

    state_df = None
    state_metrics = {
        "temperature_mean_K": np.nan,
        "temperature_std_K": np.nan,
        "volume_mean_nm3": np.nan,
        "volume_std_nm3": np.nan,
        "density_mean_g_ml": np.nan,
        "density_std_g_ml": np.nan,
    }
    if os.path.exists(csv_file):
        state_df = pd.read_csv(csv_file)
        state_df.columns = [c.strip().strip('"').lstrip("#").strip() for c in state_df.columns]

        def column_containing(*parts):
            parts_lower = [p.lower() for p in parts]
            for col in state_df.columns:
                col_lower = col.lower()
                if all(part in col_lower for part in parts_lower):
                    return col
            return None

        for metric_prefix, parts in (
            ("temperature", ("temperature",)),
            ("volume", ("volume",)),
            ("density", ("density",)),
        ):
            col = column_containing(*parts)
            if col is not None:
                values = pd.to_numeric(state_df[col], errors="coerce").dropna()
                if not values.empty:
                    unit_suffix = {"temperature": "K", "volume": "nm3", "density": "g_ml"}[metric_prefix]
                    state_metrics[f"{metric_prefix}_mean_{unit_suffix}"] = float(values.mean())
                    state_metrics[f"{metric_prefix}_std_{unit_suffix}"] = float(values.std(ddof=0))

    times_ps = np.arange(traj.n_frames) * REPORT_INTERVAL * DT.value_in_unit(unit.picoseconds)
    if state_df is not None:
        time_col = next((col for col in state_df.columns if "time" in col.lower()), None)
        if time_col is not None:
            time_values = pd.to_numeric(state_df[time_col], errors="coerce").dropna().to_numpy()
            if len(time_values) >= traj.n_frames:
                times_ps = time_values[:traj.n_frames].astype(float)
                if len(times_ps) > 0 and times_ps[0] != 0.0:
                    times_ps = times_ps - times_ps[0]
        step_col = next((col for col in state_df.columns if col.lower() == "step"), None)
        if time_col is None and step_col is not None:
            steps = pd.to_numeric(state_df[step_col], errors="coerce").dropna().to_numpy()
            if len(steps) >= traj.n_frames and len(steps) > 1:
                time_per_frame_ps = float(np.median(np.diff(steps[:traj.n_frames]))) * DT.value_in_unit(unit.picoseconds)
                times_ps = np.arange(traj.n_frames) * time_per_frame_ps

    force_norm_metrics = {
        "cwld_rms_over_base_mean": np.nan,
        "cwld_rms_over_base_max": np.nan,
        "cwld_max_over_base_mean": np.nan,
        "cwld_max_over_base_max": np.nan,
        "force_norm_warning_count": np.nan,
    }
    if os.path.exists(force_norm_file):
        force_df = pd.read_csv(force_norm_file)
        for col, out_mean, out_max in (
            ("cwld_rms_over_base", "cwld_rms_over_base_mean", "cwld_rms_over_base_max"),
            ("cwld_max_over_base", "cwld_max_over_base_mean", "cwld_max_over_base_max"),
        ):
            if col in force_df.columns:
                values = pd.to_numeric(force_df[col], errors="coerce").dropna()
                if not values.empty:
                    force_norm_metrics[out_mean] = float(values.mean())
                    force_norm_metrics[out_max] = float(values.max())
        if "mts_warning" in force_df.columns:
            force_norm_metrics["force_norm_warning_count"] = int(pd.to_numeric(force_df["mts_warning"], errors="coerce").fillna(0).sum())
    
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
    protein_o_atoms = [
        a.index
        for a in traj.topology.atoms
        if a.residue.name in AMINO_ACIDS and a.element.symbol == 'O'
    ]
    all_o_atoms = [a.index for a in traj.topology.atoms if a.element.symbol == 'O']

    def calc_ca_o_rdf(o_atoms):
        if not ca_atoms or not o_atoms:
            return np.array([]), np.array([])

        pairs = np.array([(i, j) for i in ca_atoms for j in o_atoms])
        r, g_r = md.compute_rdf(traj, pairs, r_range=(0.1, 0.6), bin_width=0.01)
        return r, g_r

    def calc_ca_coordination(o_atoms, cutoff_nm=0.30):
        if not ca_atoms or not o_atoms:
            return np.nan

        pairs = np.array([(i, j) for i in ca_atoms for j in o_atoms], dtype=int)
        distances = md.compute_distances(traj, pairs, periodic=True)
        counts = (distances < cutoff_nm).reshape(traj.n_frames, len(ca_atoms), len(o_atoms)).sum(axis=2)
        return float(np.mean(counts))

    def calc_coordination_by_cutoff(o_atoms, cutoffs_nm=(0.28, 0.30, 0.32, 0.35)):
        out = {}
        if not ca_atoms or not o_atoms:
            for cutoff_nm in cutoffs_nm:
                out[f"coord_{cutoff_nm:.2f}_nm"] = np.nan
            return out

        pairs = np.array([(i, j) for i in ca_atoms for j in o_atoms], dtype=int)
        distances = md.compute_distances(traj, pairs, periodic=True)
        distances = distances.reshape(traj.n_frames, len(ca_atoms), len(o_atoms))
        for cutoff_nm in cutoffs_nm:
            counts = (distances < cutoff_nm).sum(axis=2)
            out[f"coord_{cutoff_nm:.2f}_nm"] = float(np.mean(counts))
        return out

    def rdf_auc(r, gr, r_min=0.20, r_max=0.32):
        if len(r) == 0 or len(gr) == 0:
            return np.nan
        mask = (r >= r_min) & (r <= r_max)
        if not np.any(mask):
            return np.nan
        if TRAPEZOID is None:
            raise RuntimeError("NumPy does not provide trapezoid/trapz integration.")
        return float(TRAPEZOID(gr[mask], x=r[mask]))

    carboxylate_o_atoms = [
        a.index for a in traj.topology.atoms
        if (
            (a.residue.name == "ASP" and a.name in {"OD1", "OD2"}) or
            (a.residue.name == "GLU" and a.name in {"OE1", "OE2"})
        )
    ]
    backbone_o_atoms = [
        a.index for a in traj.topology.atoms
        if a.residue.name in AMINO_ACIDS and a.name == "O"
    ]
    carboxylate_o_set = set(carboxylate_o_atoms)
    sidechain_o_atoms = [
        a.index for a in traj.topology.atoms
        if (
            a.residue.name in AMINO_ACIDS
            and a.element.symbol == "O"
            and a.name != "O"
            and a.index not in carboxylate_o_set
        )
    ]

    r_water, gr_water = calc_ca_o_rdf(water_o_atoms)
    r_all, gr_all = calc_ca_o_rdf(all_o_atoms)
    ca_water_peak_r, ca_water_peak_g = ca_water_first_peak((r_water, gr_water))
    ca_water_coord_0p30 = calc_ca_coordination(water_o_atoms)
    ca_protein_o_coord_0p30 = calc_ca_coordination(protein_o_atoms)
    ca_total_o_coord_0p30 = calc_ca_coordination(all_o_atoms)
    ca_water_multi_coord = calc_coordination_by_cutoff(water_o_atoms)
    ca_all_o_multi_coord = calc_coordination_by_cutoff(all_o_atoms)
    ca_water_first_peak_auc = rdf_auc(r_water, gr_water, 0.20, 0.32)
    ca_all_o_first_peak_auc = rdf_auc(r_all, gr_all, 0.20, 0.32)
    ca_carboxylate_o_coord_0p30 = calc_ca_coordination(carboxylate_o_atoms, cutoff_nm=0.30)
    ca_backbone_o_coord_0p30 = calc_ca_coordination(backbone_o_atoms, cutoff_nm=0.30)
    ca_sidechain_o_coord_0p30 = calc_ca_coordination(sidechain_o_atoms, cutoff_nm=0.30)

    # 3. 表面水驻留时间：水氧进入蛋白重原子 0.35 nm 壳层的 intermittent ACF
    surface_cutoff_nm = 0.35
    water_residues = [res for res in traj.topology.residues if res.name in WATER_RESNAMES]
    water_o_by_residue = []
    for res in water_residues:
        o_atom = next((a for a in res.atoms if a.element.symbol == 'O'), None)
        if o_atom is not None:
            water_o_by_residue.append(o_atom.index)

    def calc_ca_water_residence(cutoff_nm=0.30):
        if not ca_atoms or not water_o_by_residue:
            return np.nan, np.nan, np.nan

        pairs = np.array([(ca, wo) for ca in ca_atoms for wo in water_o_by_residue], dtype=int)
        distances = md.compute_distances(traj, pairs, periodic=True)
        distances = distances.reshape(traj.n_frames, len(ca_atoms), len(water_o_by_residue))
        occupancy = np.any(distances < cutoff_nm, axis=1)
        acf = binary_acf(occupancy)
        if TRAPEZOID is None:
            raise RuntimeError("NumPy does not provide trapezoid/trapz integration.")
        tau_ps = TRAPEZOID(np.nan_to_num(acf, nan=0.0), x=times_ps[:len(acf)])
        t_1e_ps = first_crossing_time(times_ps[:len(acf)], acf, 1.0 / np.e)
        avg_shell_waters = float(np.mean(np.sum(occupancy, axis=1)))
        return (
            float(tau_ps),
            float(t_1e_ps) if np.isfinite(t_1e_ps) else np.nan,
            avg_shell_waters,
        )

    ca_water_residence_tau_ps, ca_water_residence_t_1e_ps, ca_avg_shell_waters_0p30 = calc_ca_water_residence(0.30)

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

    q_profile = compute_water_q_profile(
        traj,
        system,
        topology,
        a_q2,
        enable_water_response=enable_water_response,
        profile_mode=q_profile_mode,
    ) if a_q2 is not None else pd.DataFrame()
    ca_near_water_qo_mean = np.nan
    ca_near_water_qo_std = np.nan
    ca_near_water_dens_mean = np.nan
    ca_near_water_delta_qo_mean = np.nan
    if not q_profile.empty and "distance_to_ca_nm" in q_profile.columns:
        ca_near_q = q_profile[q_profile["distance_to_ca_nm"] <= 0.35].copy()
        if not ca_near_q.empty:
            ca_near_water_qo_mean = float(ca_near_q["water_oxygen_Qe"].mean())
            ca_near_water_qo_std = float(ca_near_q["water_oxygen_Qe"].std(ddof=0))
            ca_near_water_dens_mean = float(ca_near_q["dens"].mean())
            water_o_qbase = float(np.mean(charges[water_o_atoms])) if water_o_atoms else np.nan
            if np.isfinite(water_o_qbase):
                ca_near_water_delta_qo_mean = float((ca_near_q["water_oxygen_Qe"] - water_o_qbase).mean())

    metrics = {
        **state_metrics,
        **force_norm_metrics,
        "mean_rmsd_nm": float(np.mean(rmsd)),
        "mean_protein_rmsf_nm": mean_protein_rmsf,
        "ca_water_peak_r_nm": ca_water_peak_r,
        "ca_water_peak_g": ca_water_peak_g,
        "ca_water_coord_0p30_nm": ca_water_coord_0p30,
        "ca_protein_o_coord_0p30_nm": ca_protein_o_coord_0p30,
        "ca_total_o_coord_0p30_nm": ca_total_o_coord_0p30,
        "ca_water_coord_0p28_nm": ca_water_multi_coord["coord_0.28_nm"],
        "ca_water_coord_0p32_nm": ca_water_multi_coord["coord_0.32_nm"],
        "ca_water_coord_0p35_nm": ca_water_multi_coord["coord_0.35_nm"],
        "ca_all_o_coord_0p28_nm": ca_all_o_multi_coord["coord_0.28_nm"],
        "ca_all_o_coord_0p32_nm": ca_all_o_multi_coord["coord_0.32_nm"],
        "ca_all_o_coord_0p35_nm": ca_all_o_multi_coord["coord_0.35_nm"],
        "ca_water_first_peak_auc_0p20_0p32": ca_water_first_peak_auc,
        "ca_all_o_first_peak_auc_0p20_0p32": ca_all_o_first_peak_auc,
        "ca_carboxylate_o_coord_0p30_nm": ca_carboxylate_o_coord_0p30,
        "ca_backbone_o_coord_0p30_nm": ca_backbone_o_coord_0p30,
        "ca_sidechain_o_coord_0p30_nm": ca_sidechain_o_coord_0p30,
        "ca_water_residence_tau_ps": ca_water_residence_tau_ps,
        "ca_water_residence_t_1e_ps": ca_water_residence_t_1e_ps,
        "ca_avg_shell_waters_0p30_nm": ca_avg_shell_waters_0p30,
        "ca_near_water_QO_mean": ca_near_water_qo_mean,
        "ca_near_water_QO_std": ca_near_water_qo_std,
        "ca_near_water_dens_mean": ca_near_water_dens_mean,
        "ca_near_water_delta_QO_mean": ca_near_water_delta_qo_mean,
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

    if RUN_MTS_VALIDATION_MATRIX:
        if CWLD_ENGINE != "customgb_exact":
            raise ValueError("RUN_MTS_VALIDATION_MATRIX requires CWLD_ENGINE='customgb_exact'.")
        experiments = {}
        mts_styles = {
            1: ("exact_every_step", "purple", "-"),
            2: ("MTS_2to1", "teal", "--"),
            4: ("MTS_4to1", "orange", ":"),
        }
        for ratio in MTS_VALIDATION_RATIOS:
            label_suffix, color, linestyle = mts_styles[ratio]
            experiments[label_suffix] = {
                "label": f"CWLD_{label_suffix}",
                "system": setup_cwld_lips_system(sys_pme, topology, a_q2=DEFAULT_A_Q2, enable_water_response=ENABLE_WATER_RESPONSE),
                "reporters": [],
                "a_q2": DEFAULT_A_Q2,
                "enable_water_response": ENABLE_WATER_RESPONSE,
                "q_profile_mode": "exact",
                "mts_ratio": ratio,
                "color": color,
                "linestyle": linestyle,
            }

        analyses = {}
        energy_rows = []
        for name, cfg in experiments.items():
            dcd_file, csv_file = run_multistage_md(
                cfg["label"],
                cfg["system"],
                topology,
                positions,
                extra_reporters=cfg["reporters"],
                mts_ratio=cfg["mts_ratio"],
            )
            analyses[name] = analyze_1ckk(
                name,
                dcd_file,
                csv_file,
                topology,
                cfg["system"],
                a_q2=cfg["a_q2"],
                enable_water_response=cfg["enable_water_response"],
                q_profile_mode=cfg["q_profile_mode"],
            )
            force_energies = snapshot_force_energies(name, cfg["system"], topology, dcd_file)
            for force_name, energy in force_energies.items():
                energy_rows.append({"group": name, "force": force_name, "energy_kj_mol": energy})

        metrics_df = pd.DataFrame({name: ana["metrics"] for name, ana in analyses.items()}).T
        for name, cfg in experiments.items():
            metrics_df.loc[name, "a_q2"] = cfg["a_q2"]
            metrics_df.loc[name, "enable_water_response"] = float(cfg["enable_water_response"])
            metrics_df.loc[name, "mts_ratio"] = cfg["mts_ratio"]
        metrics_df.to_csv("1ckk_mts_validation_metrics.csv")
        if energy_rows:
            pd.DataFrame(energy_rows).to_csv("1ckk_mts_validation_force_group_energies.csv", index=False)
        print("📊 MTS validation metrics:")
        print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
        print("  -> 判据：若 MTS_4to1 的 Ca coordination、salt bridge peak、RMSF 或 force norms 偏离 exact_every_step，则不要使用 4:1。")
        return
    
    # 2. A/B/C/D 消融：PME vs water response off vs unweighted/charge-weighted density
    if CWLD_ENGINE == "cpu_kdtree_fast":
        sys_aq05_on, updater_aq05_on = setup_cwld_fast_system(sys_pme, topology, a_q2=DEFAULT_A_Q2, enable_water_response=True)
        sys_aq00_on, updater_aq00_on = setup_cwld_fast_system(sys_pme, topology, a_q2=PURE_DENSITY_A_Q2, enable_water_response=True)
        sys_aq05_water_off, updater_aq05_water_off = setup_cwld_fast_system(sys_pme, topology, a_q2=DEFAULT_A_Q2, enable_water_response=False)
    else:
        sys_aq05_on = setup_cwld_lips_system(sys_pme, topology, a_q2=DEFAULT_A_Q2, enable_water_response=True)
        sys_aq00_on = setup_cwld_lips_system(sys_pme, topology, a_q2=PURE_DENSITY_A_Q2, enable_water_response=True)
        sys_aq05_water_off = setup_cwld_lips_system(sys_pme, topology, a_q2=DEFAULT_A_Q2, enable_water_response=False)
        updater_aq05_on = None
        updater_aq00_on = None
        updater_aq05_water_off = None

    experiments = {
        "PME": {
            "label": "PME_Baseline",
            "system": sys_pme,
            "reporters": [],
            "a_q2": None,
            "enable_water_response": False,
            "q_profile_mode": "exact",
            "color": "black",
            "linestyle": "-",
        },
        "CWLD_water_off_aq0p5": {
            "label": "CWLD_Fast_water_off_aq0p5" if CWLD_ENGINE == "cpu_kdtree_fast" else "CWLD_water_off_aq0p5",
            "system": sys_aq05_water_off,
            "reporters": [updater_aq05_water_off] if updater_aq05_water_off is not None else [],
            "a_q2": DEFAULT_A_Q2,
            "enable_water_response": False,
            "q_profile_mode": "exact",
            "color": "gray",
            "linestyle": "-.",
        },
        "CWLD_aq0_unweighted_density": {
            "label": "CWLD_Fast_aq0_unweighted_density" if CWLD_ENGINE == "cpu_kdtree_fast" else "CWLD_aq0_unweighted_density",
            "system": sys_aq00_on,
            "reporters": [updater_aq00_on] if updater_aq00_on is not None else [],
            "a_q2": PURE_DENSITY_A_Q2,
            "enable_water_response": True,
            "q_profile_mode": "exact",
            "color": "teal",
            "linestyle": ":",
        },
        "CWLD_aq0p5_charge_weighted_density": {
            "label": "CWLD_Fast_aq0p5_charge_weighted_density" if CWLD_ENGINE == "cpu_kdtree_fast" else "CWLD_aq0p5_charge_weighted_density",
            "system": sys_aq05_on,
            "reporters": [updater_aq05_on] if updater_aq05_on is not None else [],
            "a_q2": DEFAULT_A_Q2,
            "enable_water_response": True,
            "q_profile_mode": "exact",
            "color": "purple",
            "linestyle": "--",
        },
    }

    analyses = {}
    energy_rows = []
    for name, cfg in experiments.items():
        dcd_file, csv_file = run_multistage_md(
            cfg["label"],
            cfg["system"],
            topology,
            positions,
            extra_reporters=cfg["reporters"],
            mts_ratio=cfg.get("mts_ratio"),
        )
        analyses[name] = analyze_1ckk(
            name,
            dcd_file,
            csv_file,
            topology,
            cfg["system"],
            a_q2=cfg["a_q2"],
            enable_water_response=cfg["enable_water_response"],
            q_profile_mode=cfg["q_profile_mode"],
        )
        force_energies = snapshot_force_energies(name, cfg["system"], topology, dcd_file)
        for force_name, energy in force_energies.items():
            energy_rows.append({"group": name, "force": force_name, "energy_kj_mol": energy})

    metrics_df = pd.DataFrame({name: ana["metrics"] for name, ana in analyses.items()}).T
    for name, cfg in experiments.items():
        metrics_df.loc[name, "a_q2"] = np.nan if cfg["a_q2"] is None else cfg["a_q2"]
        metrics_df.loc[name, "enable_water_response"] = float(cfg["enable_water_response"])
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
    for name in ("CWLD_water_off_aq0p5", "CWLD_aq0_unweighted_density", "CWLD_aq0p5_charge_weighted_density"):
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
            mask = q_df["distance_to_interface_nm"].between(0.0, 2.0)
            hb = ax.hexbin(
                q_df.loc[mask, "distance_to_interface_nm"],
                q_df.loc[mask, "water_oxygen_Qe"],
                gridsize=70,
                mincnt=1,
                cmap="viridis",
            )
            ax.axhline(-0.834, color="white", lw=1.5, ls="--", label="TIP3P O baseline")
            ax.set_title(f'{name}: Water O effective Q')
            ax.set_xlabel('Distance to nearest CWLD interface atom (nm)')
            ax.set_ylabel('Water oxygen Q (e)')
            ax.set_xlim(0.0, 2.0)
            ax.legend()
            fig.colorbar(hb, ax=ax, label='count')
        plt.tight_layout()
        plt.savefig("1ckk_v23_water_q_profile.png", dpi=300)
        print("✅ 水氧有效电荷 Q 剖面已保存至: 1ckk_v23_water_q_profile.png")

if __name__ == "__main__":
    main()
