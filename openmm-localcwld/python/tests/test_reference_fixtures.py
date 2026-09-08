"""Regression tests: localcwld_reference.compute_local_cwld() must reproduce
every golden fixture in tests/fixtures/ exactly (float64, essentially
machine precision). This is the permanent, CI-able form of what
docs/reports/generate_lcwld_030_fixtures.py did once to generate them.

Does NOT need OpenMM -- localcwld_reference has no OpenMM dependency. Needs
numpy, which this sandbox's system python lacks; run with any interpreter
that has numpy (e.g. the openmm_dev conda env, which has it as an OpenMM
dependency even though these tests never import openmm).
"""
import json
import pathlib

import numpy as np
import pytest

FIXTURES_DIR = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from localcwld_reference import Globals, ParticleArrays, compute_local_cwld


def _load_globals(g):
    return Globals(
        r_env=g["r_env_nm"], rc=g["rc_nm"], zmm_order=g["zmm_order"], rho0=g["rho0"],
        k_polar=g["k_polar"], q_delta_clamp=g["q_delta_clamp_e"], one_4pi_eps0=g["ONE_4PI_EPS0"],
        use_q_penalty=g["use_q_penalty"], q_penalty_strength=g["q_penalty_strength"],
    )


def _load_particles(specs):
    fields = ["qbase", "charge_mod", "dpolar", "is_polar", "dens_source", "dens_sink", "source_class_weight", "static_phase"]
    arrays = {f: np.array([s[f] for s in specs], dtype=float) for f in fields}
    arrays["residue_id"] = np.array([s["residue_id"] for s in specs], dtype=np.int64)
    return ParticleArrays(**arrays)


@pytest.mark.parametrize("fixture_name", [
    "two_particle_directional",
    "three_particle_chain",
    "same_residue",
    "excluded_pair",
    "pbc_cross_boundary",
])
def test_fixture_matches_reference(fixture_name):
    fixture = json.loads((FIXTURES_DIR / f"{fixture_name}.json").read_text())
    g = _load_globals(fixture["globals"])
    particles = _load_particles(fixture["particles"])
    positions = np.array([p["position_nm"] for p in fixture["particles"]])
    box = np.array(fixture["box_vectors_nm"]) if fixture.get("box_vectors_nm") else None
    exclusions = fixture["exclusions"]

    result = compute_local_cwld(positions, box, particles, exclusions, g)
    exp = fixture["expected"]

    assert np.allclose(result.density, exp["density"], atol=1e-10), f"{fixture_name}: density"
    assert np.allclose(result.Q, exp["Q"], atol=1e-10), f"{fixture_name}: Q"
    assert np.allclose(result.dQ_ddensity, exp["dQ_ddensity"], atol=1e-9), f"{fixture_name}: dQ_ddensity"
    assert abs(result.energy_total - exp["energy_total_kJ_per_mol"]) < 1e-8, f"{fixture_name}: energy"
    assert np.allclose(result.force_direct, exp["force_direct_kJ_per_mol_per_nm"], atol=1e-7), f"{fixture_name}: force_direct"
    assert np.allclose(result.force_chain, exp["force_chain_kJ_per_mol_per_nm"], atol=1e-5), f"{fixture_name}: force_chain"


def test_zmm_orders_fixture():
    fixture = json.loads((FIXTURES_DIR / "zmm_orders.json").read_text())
    from localcwld_reference import reference as ref_mod

    rc = fixture["rc_nm"]
    for order_str, payload in fixture["orders"].items():
        order = int(order_str)
        for row in payload["rows"]:
            r = row["r_nm"]
            assert abs(ref_mod.zmm_uclosure(order, r, rc) - row["U"]) < 1e-12
            assert abs(ref_mod.zmm_duclosure_dr(order, r, rc) - row["dU_dr"]) < 1e-9


def test_water_ca_cluster_fixture_analytic_vs_tabulated():
    """This fixture's ground truth ('expected') is the REAL v2.6 CustomGBForce,
    not localcwld_reference -- so this test re-derives the analytic side and
    checks it against the recorded comparison, as a regression guard on the
    DEC-002 evidence rather than a golden-value check like the other fixtures."""
    fixture = json.loads((FIXTURES_DIR / "water_ca_cluster.json").read_text())
    g = _load_globals(fixture["globals"])
    particles = _load_particles(fixture["particles"])
    positions = np.array([p["position_nm"] for p in fixture["particles"]])
    box = np.array(fixture["box_vectors_nm"])
    exclusions = fixture["exclusions"]

    result = compute_local_cwld(positions, box, particles, exclusions, g)
    ac = fixture["analytic_comparison"]

    assert np.allclose(result.density, ac["density"], atol=1e-10)
    assert np.allclose(result.Q, ac["Q"], atol=1e-10)
    assert abs(result.energy_total - ac["energy_total_kJ_per_mol"]) < 1e-8
    assert np.allclose(result.force_total, ac["force_total_kJ_per_mol_per_nm"], atol=1e-6)

    # DEC-002 regression guard: analytic-vs-tabulated error should stay in the
    # ballpark recorded when this fixture was generated; a large jump here
    # means either the analytic kernel or the tabulated cross-check changed
    # and DEC-002 needs to be revisited, not silently re-baselined.
    vs = ac["vs_tabulated"]
    assert vs["energy_rel_error"] < 1e-6
    assert vs["force_max_abs_error"] < 1e-4
