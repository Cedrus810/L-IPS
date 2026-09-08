"""Real-OpenMM parity check on the ACTUAL 1CKK system (system construction
only -- no Integrator, no Context, no dynamics/production run): compares
openmm_localcwld.metadata.build_particle_parameters() against v2.6's
build_phase_cwld_metadata() field-by-field.
"""
import importlib.util
import os
import sys
import pathlib

import numpy as np

# repo layout: <repo_root>/openmm-localcwld/docs/reports/<this file>
REPORTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = REPORTS_DIR.parents[2]

os.chdir(REPO_ROOT)  # build_1ckk_system() uses a relative '1CKK.pdb' path

sys.path.insert(0, str(REPO_ROOT / "openmm-localcwld" / "python"))
from openmm_localcwld import metadata as md

spec = importlib.util.spec_from_file_location("v26", str(REPO_ROOT / "test_lips_vs_pmeV2.6.py"))
v26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v26)

print("Building 1CKK system (PDBFixer + solvation, no MD/integration)...")
topology, system, positions = v26.build_1ckk_system()
print(f"n_particles = {system.getNumParticles()}")

meta_v26 = v26.build_phase_cwld_metadata(system, topology)
params = md.build_particle_parameters(system, topology)

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

ok = True
for our_field, v26_field in FIELD_MAP.items():
    a = getattr(params, our_field)
    b = np.asarray(meta_v26[v26_field])
    max_abs = np.max(np.abs(a - b))
    n_mismatch = int(np.sum(np.abs(a - b) > 0))
    status = "OK" if max_abs == 0.0 else "MISMATCH"
    if max_abs != 0.0:
        ok = False
        bad_idx = np.where(np.abs(a - b) > 0)[0][:10]
        print(f"[{our_field}] {status} max_abs_diff={max_abs} n_mismatch={n_mismatch} first_bad_idx={bad_idx.tolist()}")
    else:
        print(f"[{our_field}] {status} (exact match, {len(a)} particles)")

rid_ok = np.array_equal(params.residue_id.astype(float), meta_v26["mol_ids"])
print(f"[residue_id vs mol_ids] {'OK' if rid_ok else 'MISMATCH'}")
ok = ok and rid_ok

print()
print("OVERALL 1CKK PARITY:", "EXACT MATCH" if ok else "MISMATCH FOUND")
sys.exit(0 if ok else 1)
