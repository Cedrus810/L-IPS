"""Real-OpenMM parity check: openmm_localcwld.metadata.build_particle_parameters()
vs. test_lips_vs_pmeV2.6.py's build_phase_cwld_metadata(), on small synthetic
systems (not the full 1CKK system -- that needs PDBFixer solvation, which is
a heavier system-setup step, not just unit testing). Run with the real
openmm_dev interpreter.
"""
import importlib.util
import sys
import pathlib

import numpy as np
import openmm as mm
from openmm import app, unit

REPORTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = REPORTS_DIR.parents[2]

sys.path.insert(0, str(REPO_ROOT / "openmm-localcwld" / "python"))
from openmm_localcwld import metadata as md

spec = importlib.util.spec_from_file_location("v26", str(REPO_ROOT / "test_lips_vs_pmeV2.6.py"))
v26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v26)


def make_system(residue_specs):
    topology = app.Topology()
    chain = topology.addChain()
    system = mm.System()
    nb = mm.NonbondedForce()
    system.addForce(nb)
    index_by_key = {}
    for res_idx, (resname, atoms) in enumerate(residue_specs):
        residue = topology.addResidue(resname, chain)
        for atom_name, element_symbol, qbase in atoms:
            elem = app.Element.getBySymbol(element_symbol)
            atom = topology.addAtom(atom_name, elem, residue)
            idx = system.addParticle(elem.mass)
            nb.addParticle(qbase * unit.elementary_charge, 0.3 * unit.nanometer, 0.5 * unit.kilojoule_per_mole)
            index_by_key[f"{resname}_{res_idx}.{atom_name}"] = idx
    return system, topology, index_by_key


CASES = [
    ("water_response_on", [("HOH", [("O", "O", -0.834), ("H1", "H", 0.417), ("H2", "H", 0.417)])], {}),
    ("water_response_off", [("HOH", [("O", "O", -0.834), ("H1", "H", 0.417), ("H2", "H", 0.417)])], {"enable_water_response": False}),
    ("malformed_water", [("HOH", [("O", "O", -0.834), ("H1", "H", 0.417)])], {}),
    ("ions", [("NA", [("NA", "Na", 1.0)]), ("CL", [("CL", "Cl", -1.0)]), ("CA", [("CA", "Ca", 2.0)])], {"ca_source_weight": 1.5}),
    ("asp", [("ASP", [("N", "N", -0.4), ("CA", "C", 0.0), ("C", "C", 0.6), ("O", "O", -0.6),
                       ("CB", "C", -0.2), ("OD1", "O", -0.7), ("OD2", "O", -0.7)])], {}),
    # NOTE: fast_active_density=False has no v2.6 equivalent (v2.6 hardcodes
    # FAST_ACTIVE_DENSITY=True at module scope with no override), so this
    # case is EXPECTED to mismatch v26's output and is only sanity-checked
    # against itself further down, not asserted equal to v26.
    ("asp_fast_active_off", [("ASP", [("N", "N", -0.4), ("CA", "C", 0.0), ("C", "C", 0.6), ("O", "O", -0.6),
                                       ("CB", "C", -0.2), ("OD1", "O", -0.7), ("OD2", "O", -0.7)])], {}),
    ("lys", [("LYS", [("N", "N", -0.3), ("CA", "C", 0.0), ("C", "C", 0.6), ("O", "O", -0.5),
                       ("CB", "C", -0.1), ("CG", "C", -0.1), ("CD", "C", -0.1), ("CE", "C", -0.1),
                       ("NZ", "N", -0.3)])], {}),
    ("lipid", [("POPC", [("P", "P", 1.0), ("O11", "O", -0.5), ("N", "N", 0.3), ("C1", "C", -0.1), ("C2", "C", -0.1)])], {}),
    ("solute_polarization_off", [("ASP", [("N", "N", -0.4), ("CA", "C", 0.0), ("C", "C", 0.6), ("O", "O", -0.6),
                                            ("CB", "C", -0.2), ("OD1", "O", -0.7), ("OD2", "O", -0.7)])],
     {"enable_solute_polarization": False}),
    ("mixed_water_ion", [
        ("HOH", [("O", "O", -0.834), ("H1", "H", 0.417), ("H2", "H", 0.417)]),
        ("HOH", [("O", "O", -0.834), ("H1", "H", 0.417), ("H2", "H", 0.417)]),
        ("NA", [("NA", "Na", 1.0)]),
        ("ASP", [("N", "N", -0.4), ("CA", "C", 0.0), ("C", "C", 0.6), ("O", "O", -0.6),
                  ("CB", "C", -0.2), ("OD1", "O", -0.7), ("OD2", "O", -0.7)]),
    ], {"a_q2": 0.5}),
]

FIELD_MAP = {
    "qbase": "qbase",
    "charge_mod": "charge_mod_array",
    "dpolar": "dpolar",
    "is_polar": "is_polar",
    "dens_source": "dens_source",
    "dens_sink": "dens_sink",
    "source_class_weight": "source_class_weight",
    "static_phase": "static_phase",
}

overall_ok = True

for case_name, specs, kwargs in CASES:
    fast_active_density = kwargs.pop("fast_active_density_new_param", True)
    if case_name == "asp_fast_active_off":
        fast_active_density = False

    # v2.6 path: build on its own system/topology, normalize ion names first
    system_v26, topo_v26, _ = make_system(specs)
    v26.normalize_ion_resnames(topo_v26)
    dpolar_o = kwargs.get("dpolar_O", -0.15)
    meta_v26 = v26.build_phase_cwld_metadata(
        system_v26,
        topo_v26,
        dpolar_o,
        kwargs.get("a_q2", v26.DEFAULT_A_Q2),
        kwargs.get("enable_water_response", v26.ENABLE_WATER_RESPONSE),
        kwargs.get("enable_solute_polarization", v26.ENABLE_SOLUTE_POLARIZATION),
        kwargs.get("ca_source_weight", v26.CA_SOURCE_WEIGHT),
    )

    # our path: fresh system/topology (unnormalized), our own kwargs incl. fast_active_density
    system_new, topo_new, _ = make_system(specs)
    our_kwargs = dict(kwargs)
    our_kwargs.pop("dpolar_O", None)
    params = md.build_particle_parameters(system_new, topo_new, fast_active_density=fast_active_density, **our_kwargs)

    is_no_v26_equivalent = case_name == "asp_fast_active_off"
    case_ok = True
    for our_field, v26_field in FIELD_MAP.items():
        if is_no_v26_equivalent:
            continue
        a = getattr(params, our_field)
        b = np.asarray(meta_v26[v26_field])
        if not np.allclose(a, b, atol=1e-12, rtol=0):
            case_ok = False
            overall_ok = False
            diff = np.abs(a - b)
            print(f"[{case_name}] MISMATCH field={our_field}: max_abs_diff={diff.max()}")
            print(f"   ours = {a}")
            print(f"   v26  = {b}")
    # residue_id vs mol_ids (v26 stores as float, ours as int64)
    rid_ok = np.allclose(params.residue_id.astype(float), meta_v26["mol_ids"], atol=0, rtol=0)
    if not rid_ok:
        case_ok = False
        overall_ok = False
        print(f"[{case_name}] MISMATCH residue_id vs mol_ids")

    suffix = " (fast_active_density=False, no v2.6 equivalent, self-check only)" if is_no_v26_equivalent else ""
    print(f"[{case_name}] {'OK' if case_ok else 'FAIL'}  (n_particles={params.num_particles}){suffix}")

print()
print("OVERALL:", "ALL CASES MATCH v2.6 EXACTLY" if overall_ok else "SOME CASES MISMATCHED")
sys.exit(0 if overall_ok else 1)
