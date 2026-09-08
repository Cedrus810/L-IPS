"""Generate tests/fixtures/water_ca_cluster.json: a small synthetic Ca2+ +
3x TIP3P-like water cluster, cross-checked between:
  (a) localcwld_reference.compute_local_cwld (analytic density kernel), and
  (b) the REAL test_lips_vs_pmeV2.6.py CustomGBForce path (1024-point
      Continuous1DFunction density kernel), evaluated via a single static
      OpenMM Context.getState() call -- no Integrator step, no MD.

The fixture's "expected" values are taken from (b), the actual v2.6
CustomGBForce (the plan's designated ground-truth reference implementation,
see plan section 3); (a)'s analytic numbers are recorded alongside as
"analytic_comparison" for DEC-002, not substituted in as ground truth.
"""
import importlib.util
import hashlib
import json
import pathlib
import sys

import numpy as np
import openmm as mm
from openmm import app, unit

REPORTS_DIR = pathlib.Path(__file__).resolve().parent
LCWLD_ROOT = REPORTS_DIR.parents[1]
REPO_ROOT = REPORTS_DIR.parents[2]
FIXTURES_DIR = LCWLD_ROOT / "tests" / "fixtures"

sys.path.insert(0, str(LCWLD_ROOT / "python"))
from openmm_localcwld import metadata as md
from localcwld_reference import Globals, ParticleArrays, compute_local_cwld

spec = importlib.util.spec_from_file_location("v26", str(REPO_ROOT / "test_lips_vs_pmeV2.6.py"))
v26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v26)


def element(sym):
    return app.Element.getBySymbol(sym)


def build_cluster():
    topology = app.Topology()
    chain = topology.addChain()
    system = mm.System()
    nb = mm.NonbondedForce()
    system.addForce(nb)

    def add_atom(residue, name, sym, q, mass=None):
        elem = element(sym)
        atom = topology.addAtom(name, elem, residue)
        idx = system.addParticle(mass if mass is not None else elem.mass)
        nb.addParticle(q * unit.elementary_charge, 0.3 * unit.nanometer, 0.5 * unit.kilojoule_per_mole)
        assert atom.index == idx
        return idx

    positions = []

    # NOTE: must be the canonical POST-alias name "Ca2+", not the raw PDB-style
    # "CA"/"Ca". v2.6's build_phase_cwld_metadata() checks
    # `residue_name in ION_RESNAME_ALIASES.values()` -- it does NOT resolve
    # aliases itself, that's normalize_ion_resnames()'s job, which the real
    # production pipeline (build_1ckk_system) always calls first. A residue
    # named "CA" here would silently fall through to the solute/ligand branch
    # and (with FAST_ACTIVE_DENSITY default True) end up as an all-zero
    # particle -- not a density source at all. openmm_localcwld.metadata
    # resolves the alias internally (non-mutating) so it's robust to this,
    # but this script must match v2.6's actual precondition to get a fair
    # comparison.
    ca_res = topology.addResidue("Ca2+", chain)
    ca_idx = add_atom(ca_res, "CA", "Ca", 2.0)
    positions.append([0.0, 0.0, 0.0])

    # Water 1: O within r_env of Ca (0.30 nm)
    def add_water(o_pos, label):
        res = topology.addResidue("HOH", chain)
        o_idx = add_atom(res, "O", "O", -0.834)
        h1_idx = add_atom(res, "H1", "H", 0.417)
        h2_idx = add_atom(res, "H2", "H", 0.417)
        # simple, chemically-plausible-enough (not equilibrated) TIP3P-like
        # local geometry: O-H ~0.0957 nm, H-O-H ~104.5 deg, oriented in the xy
        # plane offset from o_pos.
        oh = 0.0957
        half_angle = np.deg2rad(104.52 / 2.0)
        h1 = np.array(o_pos) + oh * np.array([np.cos(half_angle), np.sin(half_angle), 0.0])
        h2 = np.array(o_pos) + oh * np.array([np.cos(half_angle), -np.sin(half_angle), 0.0])
        positions.append(list(o_pos))
        positions.append(h1.tolist())
        positions.append(h2.tolist())
        # exclude intramolecular nonbonded interactions, mirroring what a real
        # force field / constraint scheme would set up (see plan section 30
        # step 3, section 38 audit: base NonbondedForce exceptions are what
        # LocalCWLD exclusions derive from).
        nb.addException(o_idx, h1_idx, 0.0 * unit.elementary_charge ** 2, 1.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
        nb.addException(o_idx, h2_idx, 0.0 * unit.elementary_charge ** 2, 1.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
        nb.addException(h1_idx, h2_idx, 0.0 * unit.elementary_charge ** 2, 1.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
        return o_idx, h1_idx, h2_idx

    # water A: O at r(Ca-O)=0.30 nm, inside r_env=0.35
    add_water([0.30, 0.0, 0.0], "A")
    # water B: O at r(Ca-O)=sqrt(0.2^2+0.15^2+0.1^2)=~0.269 nm, inside r_env
    add_water([-0.2, 0.15, 0.1], "B")
    # water C: O at r(Ca-O)=sqrt(0.6^2+0.4^2+0.1^2)=~0.728 nm, outside r_env, inside rc
    add_water([0.6, 0.4, 0.1], "C")

    box = np.eye(3) * 3.0  # 3 nm cubic box, generous, no PBC wraparound expected
    topology.setPeriodicBoxVectors(box * unit.nanometer)
    system.setDefaultPeriodicBoxVectors(*(box * unit.nanometer))

    positions = np.array(positions)
    return system, topology, positions, box


def main():
    system, topology, positions, box = build_cluster()
    n = system.getNumParticles()
    print(f"cluster built: {n} particles (1 Ca2+ + 3 waters)")

    # --- (a) analytic reference via localcwld_reference ---
    params = md.build_particle_parameters(system, topology)  # v0.1 frozen defaults
    particles = ParticleArrays(
        qbase=params.qbase, charge_mod=params.charge_mod, dpolar=params.dpolar, is_polar=params.is_polar,
        dens_source=params.dens_source, dens_sink=params.dens_sink, source_class_weight=params.source_class_weight,
        static_phase=params.static_phase, residue_id=params.residue_id,
    )
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    exclusions = [nb.getExceptionParameters(k)[:2] for k in range(nb.getNumExceptions())]
    g = Globals(r_env=0.35, rc=1.2, zmm_order=2, rho0=13.5, k_polar=0.8, q_delta_clamp=0.20)
    analytic = compute_local_cwld(positions, box, particles, exclusions, g)

    # --- (b) real v2.6 CustomGBForce, single static Context evaluation ---
    sys_cwld = v26.setup_cwld_lips_system(system, topology)  # v0.1 frozen defaults: a_q2=0.5, ca_source_weight=2.0
    ISOLATED_GROUP = 7
    for f in sys_cwld.getForces():
        # NOTE: getForces() returns fresh Python wrapper objects each call, so
        # comparing by identity (`f is gb_force`) against an earlier-fetched
        # reference is unreliable -- compare by type instead.
        if isinstance(f, mm.CustomGBForce):
            f.setForceGroup(ISOLATED_GROUP)
        else:
            f.setForceGroup(0)

    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)  # never stepped
    platform = mm.Platform.getPlatformByName("Reference")
    context = mm.Context(sys_cwld, integrator, platform)
    context.setPositions(positions * unit.nanometer)
    context.setPeriodicBoxVectors(*(box * unit.nanometer))
    state = context.getState(getEnergy=True, getForces=True, groups={ISOLATED_GROUP})
    tabulated_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    tabulated_forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

    # --- compare ---
    e_abs_err = abs(analytic.energy_total - tabulated_energy)
    e_rel_err = e_abs_err / max(abs(analytic.energy_total), abs(tabulated_energy), 1e-10)
    f_abs_err = np.abs(analytic.force_total - tabulated_forces)
    f_rel_err = f_abs_err / np.maximum(np.maximum(np.abs(analytic.force_total), np.abs(tabulated_forces)), 1e-8)

    print(f"energy: analytic={analytic.energy_total:.6f} tabulated={tabulated_energy:.6f} abs_err={e_abs_err:.6e} rel_err={e_rel_err:.6e}")
    print(f"force max abs err: {f_abs_err.max():.6e} at particle {int(np.unravel_index(np.argmax(f_abs_err), f_abs_err.shape)[0])}")
    print(f"force max rel err (where force > 1e-3): ", end="")
    mask = np.maximum(np.abs(analytic.force_total), np.abs(tabulated_forces)) > 1e-3
    if np.any(mask):
        print(float(f_rel_err[mask].max()))
    else:
        print("n/a (all forces below floor)")

    payload = {
        "fixture_name": "water_ca_cluster",
        "fixture_schema_version": 1,
        "formula_version": "v2.6-compatible-1",
        "description": (
            "Synthetic (not equilibrated) cluster: 1 Ca2+ + 3 TIP3P-like waters "
            "(one within r_env of Ca, one within r_env, one outside r_env but "
            "inside rc), each water's intramolecular O-H/H-H pairs excluded via "
            "NonbondedForce exceptions (mirroring what a real force field would "
            "set up). Cross-checks localcwld_reference's analytic density kernel "
            "against the REAL v2.6 CustomGBForce (1024-point tabulated kernel), "
            "isolated to its own force group and evaluated with a single static "
            "OpenMM Context (no Integrator step). 'expected' values below are "
            "from the real CustomGBForce (v0.1 frozen defaults: a_q2=0.5, "
            "ca_source_weight=2.0, enable_water_response=True, "
            "enable_solute_polarization=True) -- the plan's designated "
            "ground-truth reference implementation -- with the analytic "
            "reference numbers recorded alongside under 'analytic_comparison' "
            "for DEC-002, not substituted in as ground truth."
        ),
        "generator": {
            "method": "test_lips_vs_pmeV2.6.py::setup_cwld_lips_system() -> CustomGBForce isolated to its own force group, evaluated via a single static openmm.Context.getState() call (no Integrator.step() ever called)",
            "script": "openmm-localcwld/docs/reports/generate_water_ca_cluster_fixture.py",
        },
        "globals": {
            "r_env_nm": g.r_env, "rc_nm": g.rc, "zmm_order": g.zmm_order, "rho0": g.rho0,
            "k_polar": g.k_polar, "q_delta_clamp_e": g.q_delta_clamp, "ONE_4PI_EPS0": g.one_4pi_eps0,
            "use_q_penalty": g.use_q_penalty, "q_penalty_strength": g.q_penalty_strength,
            "builder_a_q2": 0.5, "builder_ca_source_weight": 2.0,
            "builder_enable_water_response": True, "builder_enable_solute_polarization": True,
        },
        "box_vectors_nm": box.tolist(),
        "num_particles": n,
        "particles": [
            {
                "index": i,
                "position_nm": positions[i].tolist(),
                "qbase": float(params.qbase[i]),
                "charge_mod": float(params.charge_mod[i]),
                "dpolar": float(params.dpolar[i]),
                "is_polar": float(params.is_polar[i]),
                "dens_source": float(params.dens_source[i]),
                "dens_sink": float(params.dens_sink[i]),
                "source_class_weight": float(params.source_class_weight[i]),
                "static_phase": float(params.static_phase[i]),
                "residue_id": int(params.residue_id[i]),
            }
            for i in range(n)
        ],
        "exclusions": [[int(a), int(b)] for a, b in exclusions],
        "expected": {
            "source": "real v2.6 CustomGBForce (tabulated density kernel), isolated force group, static Context evaluation",
            "energy_total_kJ_per_mol": tabulated_energy,
            "force_total_kJ_per_mol_per_nm": tabulated_forces.tolist(),
        },
        "analytic_comparison": {
            "source": "localcwld_reference.compute_local_cwld (analytic density kernel)",
            "density": analytic.density.tolist(),
            "Q": analytic.Q.tolist(),
            "energy_total_kJ_per_mol": analytic.energy_total,
            "force_total_kJ_per_mol_per_nm": analytic.force_total.tolist(),
            "vs_tabulated": {
                "energy_abs_error": e_abs_err,
                "energy_rel_error": e_rel_err,
                "force_max_abs_error": float(f_abs_err.max()),
                "force_max_abs_error_particle": int(np.unravel_index(np.argmax(f_abs_err), f_abs_err.shape)[0]),
                "force_max_rel_error_where_gt_1e-3": float(f_rel_err[mask].max()) if np.any(mask) else None,
            },
        },
        "dec002_note": (
            "This is one data point for DEC-002, not a full 10,000-point sweep "
            "(see docs/reports/dec002_kernel_comparison.py / "
            "dec002_kernel_comparison_result.json for that). DEC-002 remains "
            "PENDING main-agent sign-off; do not treat either kernel as chosen "
            "based on this fixture alone."
        ),
    }

    path = FIXTURES_DIR / "water_ca_cluster.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {path}")

    # refresh sha256sums.txt
    sums_path = FIXTURES_DIR / "sha256sums.txt"
    lines = []
    for p in sorted(FIXTURES_DIR.iterdir()):
        if p.name == "sha256sums.txt" or not p.is_file():
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{digest}  {p.name}")
    sums_path.write_text("\n".join(lines) + "\n")
    print("sha256sums.txt refreshed.")


if __name__ == "__main__":
    main()
