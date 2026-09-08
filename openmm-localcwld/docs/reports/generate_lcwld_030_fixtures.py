"""Generate/verify the LCWLD-030 fixtures that don't require a real chemical
system (three_particle_chain, same_residue, excluded_pair, pbc_cross_boundary,
zmm_orders), using localcwld_reference.py as the sole source of truth, with
finite-difference cross-checks computed independently in this same script
(not just re-deriving the same formula path). Also regression-checks
localcwld_reference against the existing hand-derived
two_particle_directional.json fixture.

Does NOT touch OpenMM at all -- pure numpy. Run with any interpreter that has
numpy (this repo's sandbox system python lacks numpy, so this was run with
the openmm_dev conda environment's python, which happens to have numpy
installed as an OpenMM dependency; no `import openmm` occurs anywhere in this
script or in localcwld_reference).
"""
import hashlib
import json
import pathlib
import sys

import numpy as np

REPORTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = REPORTS_DIR.parents[2]
LCWLD_ROOT = REPORTS_DIR.parents[1]
FIXTURES_DIR = LCWLD_ROOT / "tests" / "fixtures"

sys.path.insert(0, str(LCWLD_ROOT / "python"))
from localcwld_reference import Globals, ParticleArrays, compute_local_cwld, zmm_uclosure, zmm_duclosure_dr

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print(f"FAIL: {msg}")


def make_particles(specs):
    """specs: list of dicts with the nine fields (residue_id as int)."""
    fields = ["qbase", "charge_mod", "dpolar", "is_polar", "dens_source", "dens_sink", "source_class_weight", "static_phase"]
    arrays = {f: np.array([s[f] for s in specs], dtype=float) for f in fields}
    arrays["residue_id"] = np.array([s["residue_id"] for s in specs], dtype=np.int64)
    return ParticleArrays(**arrays)


def particle_dict(p, idx, position):
    return {
        "index": idx,
        "position_nm": list(position),
        "qbase": float(p.qbase[idx]),
        "charge_mod": float(p.charge_mod[idx]),
        "dpolar": float(p.dpolar[idx]),
        "is_polar": float(p.is_polar[idx]),
        "dens_source": float(p.dens_source[idx]),
        "dens_sink": float(p.dens_sink[idx]),
        "source_class_weight": float(p.source_class_weight[idx]),
        "static_phase": float(p.static_phase[idx]),
        "residue_id": int(p.residue_id[idx]),
    }


def globals_dict(g):
    return {
        "r_env_nm": g.r_env,
        "rc_nm": g.rc,
        "zmm_order": g.zmm_order,
        "rho0": g.rho0,
        "k_polar": g.k_polar,
        "q_delta_clamp_e": g.q_delta_clamp,
        "ONE_4PI_EPS0": g.one_4pi_eps0,
        "use_q_penalty": g.use_q_penalty,
        "q_penalty_strength": g.q_penalty_strength,
    }


def finite_diff_forces(positions, box, particles, exclusions, g, h=1e-6):
    """Central finite difference of total energy w.r.t. each particle's x/y/z,
    independent cross-check of compute_local_cwld's analytic forces."""
    n = positions.shape[0]
    fd = np.zeros((n, 3))
    for i in range(n):
        for d in range(3):
            pos_plus = positions.copy()
            pos_plus[i, d] += h
            pos_minus = positions.copy()
            pos_minus[i, d] -= h
            e_plus = compute_local_cwld(pos_plus, box, particles, exclusions, g, include_forces=False).energy_total
            e_minus = compute_local_cwld(pos_minus, box, particles, exclusions, g, include_forces=False).energy_total
            fd[i, d] = -(e_plus - e_minus) / (2 * h)
    return fd


def write_fixture(name, payload):
    path = FIXTURES_DIR / f"{name}.json"
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# 0. Regression check against the existing hand-derived two_particle fixture
# ---------------------------------------------------------------------------

def check_two_particle_regression():
    fixture = json.loads((FIXTURES_DIR / "two_particle_directional.json").read_text())
    g = Globals(
        r_env=fixture["globals"]["r_env_nm"],
        rc=fixture["globals"]["rc_nm"],
        zmm_order=fixture["globals"]["zmm_order"],
        rho0=fixture["globals"]["rho0"],
        k_polar=fixture["globals"]["k_polar"],
        q_delta_clamp=fixture["globals"]["q_delta_clamp_e"],
        one_4pi_eps0=fixture["globals"]["ONE_4PI_EPS0"],
        use_q_penalty=fixture["globals"]["use_q_penalty"],
        q_penalty_strength=fixture["globals"]["q_penalty_strength"],
    )
    specs = fixture["particles"]
    particles = make_particles(specs)
    positions = np.array([p["position_nm"] for p in specs])
    box = np.array(fixture["box_vectors_nm"])
    result = compute_local_cwld(positions, box, particles, fixture["exclusions"], g)

    exp = fixture["expected"]
    check(np.allclose(result.density, exp["density"], atol=1e-12), "two_particle regression: density")
    check(np.allclose(result.Q, exp["Q"], atol=1e-12), "two_particle regression: Q")
    check(np.allclose(result.dQ_ddensity, exp["dQ_ddensity"], atol=1e-12), "two_particle regression: dQ_ddensity")
    check(np.allclose(result.lambda_, exp["lambda"], atol=1e-9), "two_particle regression: lambda")
    check(abs(result.energy_total - exp["energy_total_kJ_per_mol"]) < 1e-9, "two_particle regression: energy")
    check(
        np.allclose(result.force_direct, exp["force_direct_kJ_per_mol_per_nm"], atol=1e-9),
        "two_particle regression: force_direct",
    )
    check(
        np.allclose(result.force_chain, exp["force_chain_kJ_per_mol_per_nm"], atol=1e-6),
        "two_particle regression: force_chain",
    )
    print("two_particle_directional.json regression check done.")


# ---------------------------------------------------------------------------
# 1. three_particle_chain
# ---------------------------------------------------------------------------

def build_three_particle_chain():
    g = Globals(r_env=0.35, rc=1.2, zmm_order=2, rho0=13.5, k_polar=0.8, q_delta_clamp=0.20)
    specs = [
        dict(qbase=-0.8, charge_mod=1.0, dpolar=-0.15, is_polar=1.0, dens_source=0.0, dens_sink=1.0,
             source_class_weight=0.0, static_phase=1.0, residue_id=0),
        dict(qbase=2.0, charge_mod=1.0, dpolar=0.0, is_polar=0.0, dens_source=1.0, dens_sink=0.0,
             source_class_weight=2.0, static_phase=0.0, residue_id=1),
        dict(qbase=-1.0, charge_mod=1.0, dpolar=0.0, is_polar=0.0, dens_source=0.0, dens_sink=0.0,
             source_class_weight=0.0, static_phase=0.0, residue_id=2),
    ]
    particles = make_particles(specs)
    positions = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.5, 0.0, 0.0]])
    box = None
    exclusions = []

    result = compute_local_cwld(positions, box, particles, exclusions, g)
    fd = finite_diff_forces(positions, box, particles, exclusions, g)

    for h in (1e-4, 1e-5, 1e-6):
        fd_h = finite_diff_forces(positions, box, particles, exclusions, g, h=h)
        max_diff = np.max(np.abs(fd_h - result.force_total))
        print(f"[three_particle_chain] h={h} max|fd-analytic|={max_diff:.3e}")
    check(np.max(np.abs(fd - result.force_total)) < 1e-4, "three_particle_chain: finite-difference force check (h=1e-6)")
    check(result.density[1] == 0.0 and result.density[2] == 0.0, "three_particle_chain: only particle 0 is a sink")
    check(abs(np.sum(result.force_total, axis=0)).max() < 1e-8, "three_particle_chain: net force ~0")

    payload = {
        "fixture_name": "three_particle_chain",
        "fixture_schema_version": 1,
        "formula_version": "v2.6-compatible-1",
        "description": (
            "3 particles, 3 distinct residues, no exclusions. Particle 0 is the "
            "only response sink+polar site (water-O-like); particle 1 is the only "
            "density source (Ca2+-like) and is within r_env of particle 0; particle 2 "
            "is a fixed-charge spectator ion (Cl--like) inside rc of both but not a "
            "density source or sink. Tests that lambda correctly accumulates over "
            "multiple pair partners (particle 0 and particle 1 each interact "
            "directly with two neighbors) while the chain-force pass only fires on "
            "the single 0-1 density edge."
        ),
        "generator": {
            "method": "localcwld_reference.compute_local_cwld + independent central finite difference in this script",
            "script": "openmm-localcwld/docs/reports/generate_lcwld_030_fixtures.py",
        },
        "globals": globals_dict(g),
        "box_vectors_nm": None,
        "particles": [particle_dict(particles, i, positions[i]) for i in range(3)],
        "exclusions": [],
        "expected": {
            "density": result.density.tolist(),
            "Q": result.Q.tolist(),
            "dQ_ddensity": result.dQ_ddensity.tolist(),
            "lambda": result.lambda_.tolist(),
            "energy_direct_kJ_per_mol": result.energy_direct,
            "energy_penalty_kJ_per_mol": result.energy_penalty,
            "energy_total_kJ_per_mol": result.energy_total,
            "force_direct_kJ_per_mol_per_nm": result.force_direct.tolist(),
            "force_chain_kJ_per_mol_per_nm": result.force_chain.tolist(),
            "force_total_kJ_per_mol_per_nm": result.force_total.tolist(),
            "finite_difference_force_total_h1e-6": fd.tolist(),
        },
    }
    write_fixture("three_particle_chain", payload)


# ---------------------------------------------------------------------------
# 2. same_residue
# ---------------------------------------------------------------------------

def build_same_residue():
    g = Globals(r_env=0.35, rc=1.2, zmm_order=2, rho0=13.5, k_polar=0.8, q_delta_clamp=0.20)
    specs = [
        dict(qbase=-0.8, charge_mod=1.0, dpolar=-0.15, is_polar=1.0, dens_source=0.0, dens_sink=1.0,
             source_class_weight=0.0, static_phase=1.0, residue_id=0),
        dict(qbase=2.0, charge_mod=1.0, dpolar=0.0, is_polar=0.0, dens_source=1.0, dens_sink=0.0,
             source_class_weight=2.0, static_phase=0.0, residue_id=0),  # SAME residue_id as particle 0
    ]
    particles = make_particles(specs)
    positions = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    box = None
    exclusions = []

    result = compute_local_cwld(positions, box, particles, exclusions, g)
    fd = finite_diff_forces(positions, box, particles, exclusions, g)

    check(result.density[0] == 0.0, "same_residue: density must be 0 across a same-residue edge")
    check(result.Q[0] == particles.qbase[0], "same_residue: Q0 must equal qbase0 (no polarization)")
    check(result.energy_total != 0.0, "same_residue: Local pair energy must NOT be zeroed by same-residue alone")
    check(np.allclose(result.force_chain, 0.0), "same_residue: chain force must be exactly 0 (no density edge)")
    check(np.max(np.abs(fd - result.force_total)) < 1e-4, "same_residue: finite-difference force check")

    payload = {
        "fixture_name": "same_residue",
        "fixture_schema_version": 1,
        "formula_version": "v2.6-compatible-1",
        "description": (
            "Same two particles/geometry as two_particle_directional, but both "
            "given residue_id=0 and NOT excluded. Per plan section 49: same-residue "
            "only zeroes the DENSITY edge, not the Local pair energy/force -- this "
            "fixture is the regression test for that exact distinction (do not "
            "confuse with excluded_pair, where everything is zero)."
        ),
        "generator": {
            "method": "localcwld_reference.compute_local_cwld + independent central finite difference in this script",
            "script": "openmm-localcwld/docs/reports/generate_lcwld_030_fixtures.py",
        },
        "globals": globals_dict(g),
        "box_vectors_nm": None,
        "particles": [particle_dict(particles, i, positions[i]) for i in range(2)],
        "exclusions": [],
        "expected": {
            "density": result.density.tolist(),
            "Q": result.Q.tolist(),
            "dQ_ddensity": result.dQ_ddensity.tolist(),
            "lambda": result.lambda_.tolist(),
            "energy_direct_kJ_per_mol": result.energy_direct,
            "energy_penalty_kJ_per_mol": result.energy_penalty,
            "energy_total_kJ_per_mol": result.energy_total,
            "force_direct_kJ_per_mol_per_nm": result.force_direct.tolist(),
            "force_chain_kJ_per_mol_per_nm": result.force_chain.tolist(),
            "force_total_kJ_per_mol_per_nm": result.force_total.tolist(),
            "finite_difference_force_total_h1e-6": fd.tolist(),
        },
        "invariants": {
            "density_zero_across_same_residue_edge": True,
            "chain_force_exactly_zero": True,
            "local_pair_energy_nonzero": True,
        },
    }
    write_fixture("same_residue", payload)


# ---------------------------------------------------------------------------
# 3. excluded_pair
# ---------------------------------------------------------------------------

def build_excluded_pair():
    g = Globals(r_env=0.35, rc=1.2, zmm_order=2, rho0=13.5, k_polar=0.8, q_delta_clamp=0.20)
    specs = [
        dict(qbase=-0.8, charge_mod=1.0, dpolar=-0.15, is_polar=1.0, dens_source=0.0, dens_sink=1.0,
             source_class_weight=0.0, static_phase=1.0, residue_id=0),
        dict(qbase=2.0, charge_mod=1.0, dpolar=0.0, is_polar=0.0, dens_source=1.0, dens_sink=0.0,
             source_class_weight=2.0, static_phase=0.0, residue_id=1),
    ]
    particles = make_particles(specs)
    positions = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    box = None
    exclusions = [(0, 1)]

    result = compute_local_cwld(positions, box, particles, exclusions, g)

    check(np.allclose(result.density, 0.0), "excluded_pair: density must be exactly 0")
    check(np.array_equal(result.Q, particles.qbase), "excluded_pair: Q must equal qbase for both particles")
    check(result.energy_total == 0.0, "excluded_pair: energy must be exactly 0")
    check(np.allclose(result.force_direct, 0.0), "excluded_pair: force_direct must be exactly 0")
    check(np.allclose(result.force_chain, 0.0), "excluded_pair: force_chain must be exactly 0")

    payload = {
        "fixture_name": "excluded_pair",
        "fixture_schema_version": 1,
        "formula_version": "v2.6-compatible-1",
        "description": (
            "Same geometry as same_residue but different residue_id (0 and 1) plus "
            "an explicit addExclusion(0,1). Density, Local pair energy, direct "
            "force and chain force must ALL be exactly zero -- contrast with "
            "same_residue, where only the density edge is zeroed."
        ),
        "generator": {
            "method": "localcwld_reference.compute_local_cwld",
            "script": "openmm-localcwld/docs/reports/generate_lcwld_030_fixtures.py",
        },
        "globals": globals_dict(g),
        "box_vectors_nm": None,
        "particles": [particle_dict(particles, i, positions[i]) for i in range(2)],
        "exclusions": [[0, 1]],
        "expected": {
            "density": result.density.tolist(),
            "Q": result.Q.tolist(),
            "dQ_ddensity": result.dQ_ddensity.tolist(),
            "lambda": result.lambda_.tolist(),
            "energy_direct_kJ_per_mol": result.energy_direct,
            "energy_penalty_kJ_per_mol": result.energy_penalty,
            "energy_total_kJ_per_mol": result.energy_total,
            "force_direct_kJ_per_mol_per_nm": result.force_direct.tolist(),
            "force_chain_kJ_per_mol_per_nm": result.force_chain.tolist(),
            "force_total_kJ_per_mol_per_nm": result.force_total.tolist(),
        },
        "invariants": {"everything_exactly_zero": True},
    }
    write_fixture("excluded_pair", payload)


# ---------------------------------------------------------------------------
# 4. pbc_cross_boundary
# ---------------------------------------------------------------------------

def build_pbc_cross_boundary():
    # Deliberately small 1 nm cubic box with a shrunk rc (0.4 nm, still >= r_env)
    # so the cutoff stays well under half the box length while still letting us
    # exercise a genuine cross-boundary minimum-image pair. NOT representative
    # of the production r_env=0.35/rc=1.2 defaults -- this fixture is about PBC
    # correctness, not physical realism.
    g = Globals(r_env=0.35, rc=0.4, zmm_order=2, rho0=13.5, k_polar=0.8, q_delta_clamp=0.20)
    specs = [
        dict(qbase=-0.8, charge_mod=1.0, dpolar=-0.15, is_polar=1.0, dens_source=0.0, dens_sink=1.0,
             source_class_weight=0.0, static_phase=1.0, residue_id=0),
        dict(qbase=2.0, charge_mod=1.0, dpolar=0.0, is_polar=0.0, dens_source=1.0, dens_sink=0.0,
             source_class_weight=2.0, static_phase=0.0, residue_id=1),
    ]
    particles = make_particles(specs)
    box = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # raw separation along x is 0.9 nm; minimum image must wrap to 0.1 nm.
    positions = np.array([[0.95, 0.5, 0.5], [0.05, 0.5, 0.5]])
    exclusions = []

    result = compute_local_cwld(positions, box, particles, exclusions, g)
    fd = finite_diff_forces(positions, box, particles, exclusions, g)

    # sanity: recompute what the *unwrapped* (wrong) r=0.9 would have given, to
    # make sure the fixture is actually exercising the wrap and not accidentally
    # passing because r_env/rc make the density term zero either way.
    r_wrapped = 0.1
    check(result.density[0] > 0.0, "pbc_cross_boundary: minimum-image density edge must be nonzero (r=0.1 < r_env)")

    # translation-by-one-full-lattice-vector invariance (plan section 31.2/47)
    shifted_positions = positions + box[0]  # shift both particles by the 'a' lattice vector
    shifted_result = compute_local_cwld(shifted_positions, box, particles, exclusions, g)
    check(np.allclose(result.density, shifted_result.density, atol=0), "pbc_cross_boundary: lattice-translation invariance (density)")
    check(abs(result.energy_total - shifted_result.energy_total) == 0.0, "pbc_cross_boundary: lattice-translation invariance (energy)")
    check(np.allclose(result.force_total, shifted_result.force_total, atol=0), "pbc_cross_boundary: lattice-translation invariance (force)")
    check(np.max(np.abs(fd - result.force_total)) < 1e-4, "pbc_cross_boundary: finite-difference force check")

    payload = {
        "fixture_name": "pbc_cross_boundary",
        "fixture_schema_version": 1,
        "formula_version": "v2.6-compatible-1",
        "description": (
            "1 nm cubic orthorhombic box (deliberately small; rc shrunk to 0.4 nm "
            "for this fixture only, still >= r_env, so the cutoff stays under half "
            "the box length). Two particles placed near opposite faces so their raw "
            "(unwrapped) separation is 0.9 nm but the minimum-image separation is "
            "0.1 nm. Also checks that shifting BOTH particles by one full lattice "
            "vector 'a' leaves density/energy/force exactly unchanged."
        ),
        "generator": {
            "method": "localcwld_reference.compute_local_cwld + independent central finite difference in this script",
            "script": "openmm-localcwld/docs/reports/generate_lcwld_030_fixtures.py",
        },
        "globals": globals_dict(g),
        "box_vectors_nm": box.tolist(),
        "particles": [particle_dict(particles, i, positions[i]) for i in range(2)],
        "exclusions": [],
        "expected_minimum_image_separation_nm": r_wrapped,
        "expected": {
            "density": result.density.tolist(),
            "Q": result.Q.tolist(),
            "dQ_ddensity": result.dQ_ddensity.tolist(),
            "lambda": result.lambda_.tolist(),
            "energy_direct_kJ_per_mol": result.energy_direct,
            "energy_penalty_kJ_per_mol": result.energy_penalty,
            "energy_total_kJ_per_mol": result.energy_total,
            "force_direct_kJ_per_mol_per_nm": result.force_direct.tolist(),
            "force_chain_kJ_per_mol_per_nm": result.force_chain.tolist(),
            "force_total_kJ_per_mol_per_nm": result.force_total.tolist(),
            "finite_difference_force_total_h1e-6": fd.tolist(),
        },
        "invariants": {
            "lattice_translation_by_a_leaves_energy_force_density_unchanged": True,
        },
        "note": (
            "This fixture only covers an orthorhombic box shifted along a single "
            "axis-aligned lattice vector. A genuine triclinic (tilted) box test is "
            "still TODO -- see plan section 47; do not treat this fixture as "
            "covering triclinic support."
        ),
    }
    write_fixture("pbc_cross_boundary", payload)


# ---------------------------------------------------------------------------
# 5. zmm_orders
# ---------------------------------------------------------------------------

def build_zmm_orders():
    rc = 1.2
    r_values = [0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.15, 1.199]
    orders_payload = {}
    for order in (1, 2, 3):
        rows = []
        for r in r_values:
            U = zmm_uclosure(order, r, rc)
            dUdr = zmm_duclosure_dr(order, r, rc)
            fd_vals = {}
            for h in (1e-4, 1e-5, 1e-6):
                fd = (zmm_uclosure(order, r + h, rc) - zmm_uclosure(order, r - h, rc)) / (2 * h)
                fd_vals[f"h_{h:g}"] = fd
            check(abs(fd_vals["h_1e-06"] - dUdr) < 1e-4, f"zmm_orders: order={order} r={r} FD dU/dr mismatch")
            rows.append({"r_nm": r, "U": U, "dU_dr": dUdr, "finite_difference_dU_dr": fd_vals})

        # value and derivative should vanish smoothly at r -> rc
        r_near_rc = rc * (1 - 1e-6)
        U_near_rc = zmm_uclosure(order, r_near_rc, rc)
        dU_near_rc = zmm_duclosure_dr(order, r_near_rc, rc)
        check(abs(U_near_rc) < 1e-6, f"zmm_orders: order={order} U(r->rc) should vanish, got {U_near_rc}")
        check(abs(dU_near_rc) < 1e-4, f"zmm_orders: order={order} dU/dr(r->rc) should vanish, got {dU_near_rc}")

        orders_payload[str(order)] = {
            "rows": rows,
            "U_at_r_near_rc": U_near_rc,
            "dU_dr_at_r_near_rc": dU_near_rc,
        }

    payload = {
        "fixture_name": "zmm_orders",
        "fixture_schema_version": 1,
        "formula_version": "v2.6-compatible-1",
        "description": (
            "U_ell(r) and dU_ell/dr for ell=1,2,3 at rc=1.2nm, evaluated at a "
            "spread of r values plus a point very close to rc, with independent "
            "central finite-difference cross-checks at three step sizes. Confirms "
            "U and dU/dr both vanish smoothly as r -> rc for every order (the "
            "'zero multipole' cancellation construction), and that no order is "
            "untested (plan section 32 explicitly warns against only testing the "
            "default ell=2)."
        ),
        "generator": {
            "method": "localcwld_reference.zmm_uclosure/zmm_duclosure_dr + independent central finite difference",
            "script": "openmm-localcwld/docs/reports/generate_lcwld_030_fixtures.py",
        },
        "rc_nm": rc,
        "orders": orders_payload,
    }
    write_fixture("zmm_orders", payload)


def main():
    check_two_particle_regression()
    build_three_particle_chain()
    build_same_residue()
    build_excluded_pair()
    build_pbc_cross_boundary()
    build_zmm_orders()

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)

    # refresh sha256sums.txt for everything in tests/fixtures/
    sums_path = FIXTURES_DIR / "sha256sums.txt"
    lines = []
    for p in sorted(FIXTURES_DIR.iterdir()):
        if p.name == "sha256sums.txt" or not p.is_file():
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{digest}  {p.name}")
    sums_path.write_text("\n".join(lines) + "\n")

    print("\nALL CHECKS PASSED. Fixtures written and sha256sums.txt refreshed.")


if __name__ == "__main__":
    main()
