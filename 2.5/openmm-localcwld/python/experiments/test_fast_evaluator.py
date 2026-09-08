"""End-to-end fast CPU regression; never a substitute for CUDA validation."""

from dataclasses import fields, replace
import json
from pathlib import Path

import numpy as np
import pytest

from localcwld_fast import LocalCWLDFastEvaluator
from localcwld_reference import Globals, ParticleArrays, compute_local_cwld


# The fast backend is float32 end to end (DEC-004) while localcwld_reference
# stays float64, so these are single-precision agreement bands, not the old
# near-bit-exact ones. Measured worst case on this suite is ~2e-5 relative on
# forces and ~1e-7 on energy; the bands below keep roughly an order of
# magnitude of headroom. Tightening them past float32 eps (1.2e-7) would only
# be testing that float32 is float64.
_RTOL, _ATOL = 2e-4, 2e-3


def _assert_result(actual, expected):
    for name in ("density", "Q", "dQ_ddensity"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=_RTOL, atol=1e-6, err_msg=name)
    np.testing.assert_allclose(actual.lambda_, expected.lambda_, rtol=_RTOL, atol=_ATOL)
    for name in ("energy_direct", "energy_penalty", "energy_total"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=_RTOL, atol=_ATOL, err_msg=name)
    for name in ("force_direct", "force_chain", "force_total"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=_RTOL, atol=_ATOL, err_msg=name)


@pytest.mark.parametrize("energy,forces", [(False, False), (False, True), (True, False), (True, True)])
def test_fast_all_stages_and_flags(case, energy, forces):
    evaluator = LocalCWLDFastEvaluator(case.particles, case.exclusions, case.globals)
    actual = evaluator.compute(case.positions, case.box, include_energy=energy, include_forces=forces)
    expected = compute_local_cwld(case.positions, case.box, case.particles, case.exclusions, case.globals,
                                  include_energy=energy, include_forces=forces)
    _assert_result(actual, expected)


@pytest.mark.parametrize("name", ["two_particle_directional", "three_particle_chain", "same_residue",
                                  "excluded_pair", "pbc_cross_boundary", "water_ca_cluster"])
def test_fast_existing_fixtures(name):
    path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / f"{name}.json"
    fixture = json.loads(path.read_text())
    specs, g = fixture["particles"], fixture["globals"]
    arrays = {f.name: np.array([p[f.name] for p in specs], dtype=np.int64 if f.name == "residue_id" else float)
              for f in fields(ParticleArrays)}
    p = ParticleArrays(**arrays)
    globals_ = Globals(r_env=g["r_env_nm"], rc=g["rc_nm"], zmm_order=g["zmm_order"], rho0=g["rho0"],
                       k_polar=g["k_polar"], q_delta_clamp=g["q_delta_clamp_e"],
                       one_4pi_eps0=g["ONE_4PI_EPS0"], use_q_penalty=g["use_q_penalty"],
                       q_penalty_strength=g["q_penalty_strength"])
    pos = np.array([p["position_nm"] for p in specs])
    box = np.array(fixture["box_vectors_nm"]) if fixture.get("box_vectors_nm") else None
    expected = compute_local_cwld(pos, box, p, fixture["exclusions"], globals_)
    actual = LocalCWLDFastEvaluator(p, fixture["exclusions"], globals_).compute(pos, box)
    _assert_result(actual, expected)
    # water_ca uses analytic_comparison: this does not silently choose an
    # analytic replacement for its different, tabulated CustomGB golden.
    golden = fixture["analytic_comparison"] if name == "water_ca_cluster" else fixture["expected"]
    # Goldens stay float64 and are not regenerated for this change; the fast
    # backend is now checked against them at single precision.
    np.testing.assert_allclose(actual.energy_total, golden["energy_total_kJ_per_mol"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(actual.force_total, golden["force_total_kJ_per_mol_per_nm"], rtol=_RTOL, atol=_ATOL)


def test_total_force_finite_difference_against_reference(case):
    """Check F = -dE/dx on the float64 reference, then the fast backend on it.

    The finite difference deliberately runs against localcwld_reference, not
    against the float32 fast backend. A central difference with h=1e-5 nm
    subtracts two nearly equal energies of order 200 kJ/mol; in float32 the
    surviving digits are noise, measured at |F - FD| ~ 2 kJ/mol/nm against
    forces of ~100 (a 2% floor), versus 4e-7 in float64. Running the gradient
    check in float32 would test rounding, not the force expression.

    The fast backend's own correctness is then the second assertion: it must
    reproduce the reference forces to single precision.
    """
    h = 1e-5
    reference = compute_local_cwld(case.positions, case.box, case.particles,
                                   case.exclusions, case.globals)
    fd = np.zeros_like(case.positions)
    for i in range(case.particles.num_particles):
        for axis in range(3):
            plus, minus = case.positions.copy(), case.positions.copy()
            plus[i, axis] += h
            minus[i, axis] -= h
            ep = compute_local_cwld(plus, case.box, case.particles, case.exclusions,
                                    case.globals, include_forces=False).energy_total
            em = compute_local_cwld(minus, case.box, case.particles, case.exclusions,
                                    case.globals, include_forces=False).energy_total
            fd[i, axis] = -(ep - em) / (2 * h)
    np.testing.assert_allclose(reference.force_total, fd, rtol=1e-6, atol=1e-5)

    evaluator = LocalCWLDFastEvaluator(case.particles, case.exclusions, case.globals)
    _assert_result(evaluator.compute(case.positions, case.box), reference)


def _random_particles(n, rng):
    return ParticleArrays(qbase=rng.uniform(-1, 1, n), charge_mod=rng.uniform(.5, 2, n),
                         dpolar=rng.uniform(-.15, .15, n), is_polar=rng.integers(0, 2, n).astype(float),
                         dens_source=rng.integers(0, 2, n).astype(float), dens_sink=rng.integers(0, 2, n).astype(float),
                         source_class_weight=rng.uniform(.3, 2, n), static_phase=np.ones(n), residue_id=np.arange(n)//3)


@pytest.mark.parametrize("box_kind", ["none", "orthorhombic", "triclinic"])
@pytest.mark.parametrize("seed", [7, 19, 41])
def test_fast_sparse_neighbors_and_reordering(box_kind, seed):
    rng = np.random.default_rng(seed)
    p = _random_particles(64, rng)
    box = {"none": None, "orthorhombic": np.diag([2.8, 3., 3.2]),
           "triclinic": np.array([[2.8, 0., 0.], [1.2, 3., 0.], [-1.3, 1.4, 3.2]])}[box_kind]
    pos = rng.uniform(0, 1, (64, 3)) @ (box if box is not None else np.eye(3)*3)
    if box is not None:
        pos += rng.integers(-2, 3, (64, 3)) @ box
    g = Globals(r_env=.7, rc=1.2, use_q_penalty=True)
    exclusions = [(4, 1), (8, 23)]
    expected = compute_local_cwld(pos, box, p, exclusions, g)
    evaluator = LocalCWLDFastEvaluator(p, exclusions, g)
    actual = evaluator.compute(pos, box)
    _assert_result(actual, expected)
    assert np.max(np.abs(actual.force_chain)) > 0, "must exercise density edges"
    # Newton's third law is exact per pair, but the float32 scatter-add over
    # 64 particles leaves a rounding residual rather than a hard zero.
    np.testing.assert_allclose(actual.force_total.sum(axis=0), 0., rtol=0., atol=_ATOL)
    perm = rng.permutation(64)
    inverse = np.argsort(perm)
    pp = ParticleArrays(**{f.name: getattr(p, f.name)[perm] for f in fields(p)})
    ee = [(int(inverse[i]), int(inverse[j])) for i, j in exclusions]
    reordered = LocalCWLDFastEvaluator(pp, ee, g).compute(pos[perm], box)
    # Permuting particles changes the float32 summation order, so this is
    # order-independence to single precision, not bit reproducibility.
    np.testing.assert_allclose(reordered.force_total[inverse], actual.force_total,
                               rtol=_RTOL, atol=_ATOL)


def test_fast_snapshot_parameters_and_rebuilds_geometry(case):
    evaluator = LocalCWLDFastEvaluator(case.particles, case.exclusions, case.globals)
    before = evaluator.compute(case.positions, case.box)
    positions_copy = case.positions.copy()
    case.particles.qbase[:] += 1.0
    _assert_result(evaluator.compute(case.positions, case.box), before)
    np.testing.assert_array_equal(case.positions, positions_copy)
    # A fresh evaluator is the explicit parameter update operation.
    updated = LocalCWLDFastEvaluator(case.particles, case.exclusions, case.globals)
    moved = case.positions.copy()
    moved[0] += [.6, .2, -.1]
    new_box = case.box * 1.1
    actual = updated.compute(moved, new_box)
    expected = compute_local_cwld(moved, new_box, case.particles, case.exclusions, case.globals)
    _assert_result(actual, expected)


def _ulp32(value, steps):
    """Step `steps` float32 ULPs away from `value` (negative steps go down)."""
    out = np.float32(value)
    target = np.float32(-np.inf if steps < 0 else np.inf)
    for _ in range(abs(steps)):
        out = np.nextafter(out, target, dtype=np.float32)
    return float(out)


# Probe the cutoffs a few float32 ULPs out rather than 1e-9 nm. At 0.35 nm one
# float32 ULP is ~3e-8, so the old float64-sized offsets all collapse onto the
# same float32 number and stop testing anything. The property under test --
# the cutoff is strict and the neighbor-search margin does not widen it -- is
# unchanged; only the resolution it is probed at follows the working precision.
@pytest.mark.parametrize("r", [_ulp32(0.35, -4), 0.35, _ulp32(0.35, 4),
                               _ulp32(1.2, -4), 1.2, _ulp32(1.2, 4)])
def test_fast_strict_boundaries(r):
    p = _random_particles(2, np.random.default_rng(4))
    p = replace(p, residue_id=np.arange(2), is_polar=np.ones(2), dens_sink=np.ones(2), dens_source=np.ones(2))
    pos = np.array([[0., 0., 0.], [r, 0., 0.]])
    g = Globals(r_env=.35, rc=1.2)
    _assert_result(LocalCWLDFastEvaluator(p, [], g).compute(pos), compute_local_cwld(pos, None, p, [], g))


@pytest.mark.parametrize("radius", [.35, 1.2])
def test_fast_cutoff_roundoff_in_arbitrary_directions(radius):
    """Off-axis cutoff behaviour, split into the resolvable and the ambiguous.

    Rotating a separation into a general direction costs about sqrt(3) float32
    ULPs of length, so an offset has to clear roughly 1e-6 nm before the two
    implementations can be required to agree on whether the pair is inside the
    cutoff. Clearly inside and clearly outside are asserted normally. Exactly
    on the cutoff is genuinely undecidable in float32 -- and it matters,
    because this closure's pair force does not vanish at rc (a flip is worth
    tens of kJ/mol/nm, not a rounding crumb) -- so it is asserted to land on
    one of the two admissible answers instead of being quietly loosened away.
    The GPU kernel will have exactly the same property.
    """
    rng = np.random.default_rng(9)
    p = _random_particles(2, rng)
    p = replace(p, residue_id=np.arange(2), is_polar=np.ones(2), dens_sink=np.ones(2), dens_source=np.ones(2))
    g = Globals(r_env=.35, rc=1.2)
    evaluator = LocalCWLDFastEvaluator(p, [], g)

    resolvable = 16 * (np.nextafter(np.float32(radius), np.float32(np.inf)) - np.float32(radius))
    for distance in (radius - resolvable, radius + resolvable):
        for _ in range(30):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            pos = np.array([[0., 0., 0.], distance * direction])
            _assert_result(evaluator.compute(pos), compute_local_cwld(pos, None, p, [], g))

    for _ in range(30):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        pos = np.array([[0., 0., 0.], radius * direction])
        actual = evaluator.compute(pos)
        inside = compute_local_cwld(np.array([[0., 0., 0.], (radius - resolvable) * direction]),
                                    None, p, [], g)
        outside = compute_local_cwld(np.array([[0., 0., 0.], (radius + resolvable) * direction]),
                                     None, p, [], g)
        assert (np.allclose(actual.force_total, inside.force_total, rtol=_RTOL, atol=_ATOL)
                or np.allclose(actual.force_total, outside.force_total, rtol=_RTOL, atol=_ATOL)), (
            f"at r == {radius} the float32 result matched neither the included nor "
            f"the excluded branch: {actual.force_total}")


@pytest.mark.parametrize("tilt", [0., .5])
def test_fast_periodic_negative_epsilon_wrap(tilt):
    p = _random_particles(2, np.random.default_rng(3))
    g = Globals(r_env=.35, rc=1.2)
    box = np.array([[3., 0., 0.], [tilt, 3., 0.], [0., tilt, 3.]])
    pos = np.array([[-1e-18, 0., 0.], [.3, 0., 0.]])
    _assert_result(LocalCWLDFastEvaluator(p, [], g).compute(pos, box),
                   compute_local_cwld(pos, box, p, [], g))


@pytest.mark.parametrize("n", [0, 1])
def test_fast_empty_and_single_particle(n):
    p = _random_particles(n, np.random.default_rng(3))
    pos = np.zeros((n, 3))
    g = Globals(r_env=.35, rc=1.2)
    _assert_result(LocalCWLDFastEvaluator(p, [], g).compute(pos), compute_local_cwld(pos, None, p, [], g))


def test_fast_fixed_charges_still_have_pair_correction():
    p = _random_particles(2, np.random.default_rng(3))
    p = replace(p, is_polar=np.zeros(2), dens_sink=np.zeros(2))
    g = Globals(r_env=.35, rc=1.2)
    pos = np.array([[0., 0., 0.], [.25, .1, .2]])
    actual = LocalCWLDFastEvaluator(p, [], g).compute(pos)
    expected = compute_local_cwld(pos, None, p, [], g)
    _assert_result(actual, expected)
    assert actual.energy_total != 0 and np.max(np.abs(actual.force_direct)) > 0
    np.testing.assert_array_equal(actual.force_chain, 0.)


def test_fast_rejects_invalid_input():
    p = _random_particles(2, np.random.default_rng(3))
    g = Globals(r_env=.35, rc=1.2)
    with pytest.raises(ValueError):
        LocalCWLDFastEvaluator(replace(p, qbase=np.zeros((2, 1))), [], g)
    with pytest.raises(ValueError):
        LocalCWLDFastEvaluator(p, [(0, 2)], g)
    with pytest.raises(ValueError):
        LocalCWLDFastEvaluator(p, [], replace(g, one_4pi_eps0=float("nan")))
    evaluator = LocalCWLDFastEvaluator(p, [], g)
    with pytest.raises(ValueError):
        evaluator.compute(np.full((2, 3), np.nan))
    with pytest.raises(ValueError):
        evaluator.compute(np.zeros((2, 3)))
    # Coincident particles are harmless if their pair is explicitly excluded.
    LocalCWLDFastEvaluator(p, [(0, 1)], g).compute(np.zeros((2, 3)))
