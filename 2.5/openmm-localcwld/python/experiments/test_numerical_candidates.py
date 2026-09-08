"""CPU algebra experiments, NOT a GPU benchmark or tabulated-kernel validation.

Candidate expressions live only here. The production float64 reference and all
golden files remain untouched. Explicit tolerances below apply to these small
float64 equivalence/finite-difference experiments, not to GPU acceptance.
"""

from dataclasses import fields
from decimal import Decimal, localcontext

import numpy as np
import pytest

from localcwld_reference import Globals, ParticleArrays, compute_local_cwld
from localcwld_reference.reference import minimum_image, zmm_duclosure_dr, zmm_uclosure


def _compute(case, **flags):
    return compute_local_cwld(
        case.positions, case.box, case.particles, case.exclusions, case.globals, **flags,
    )


def _squared_density_and_chain(positions, box, particles, exclusions, g, adjoint, dq):
    """Analytic K only; Q/dQ/lambda come from reference to isolate A/D algebra.

    This intentionally shares the reference PBC helper. It does not independently
    validate minimum-image geometry or an end-to-end optimized implementation.
    """
    n = particles.num_particles
    excluded = {tuple(sorted(pair)) for pair in exclusions}
    source = particles.dens_source * particles.source_class_weight * particles.charge_mod
    response = adjoint * dq * particles.dens_sink
    env2 = g.r_env * g.r_env
    density = np.zeros(n)
    chain = np.zeros((n, 3))
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in excluded or particles.residue_id[i] == particles.residue_id[j]:
                continue
            delta = minimum_image(positions[j] - positions[i], box)
            r2 = float(delta @ delta)
            if r2 >= env2:
                continue
            t = 1.0 - r2 / env2
            density[i] += particles.dens_sink[i] * source[j] * t * t
            density[j] += particles.dens_sink[j] * source[i] * t * t
            force = (-4.0 / env2 * t * (response[i] * source[j] + response[j] * source[i])) * delta
            chain[i] += force
            chain[j] -= force
    return density, chain


def test_squared_density_and_chain_match_reference(case, record_property):
    reference = _compute(case)
    assert np.max(np.abs(reference.force_chain)) > 1e-3, "vacuous chain-force test"
    density, chain = _squared_density_and_chain(
        case.positions, case.box, case.particles, case.exclusions, case.globals,
        reference.lambda_, reference.dQ_ddensity,
    )
    record_property("density_max_abs", float(np.max(np.abs(density - reference.density))))
    record_property("chain_max_abs", float(np.max(np.abs(chain - reference.force_chain))))
    np.testing.assert_allclose(density, reference.density, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(chain, reference.force_chain, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("offset", [-1e-6, 0.0, 1e-6], ids=["inside", "boundary", "outside"])
def test_squared_kernel_environment_boundary(offset):
    g = Globals(r_env=0.35, rc=1.2)
    p = ParticleArrays(
        qbase=np.array([-0.834, 2.0]), charge_mod=np.ones(2),
        dpolar=np.array([-0.15, 0.0]), is_polar=np.array([1.0, 0.0]),
        dens_source=np.array([0.0, 1.0]), dens_sink=np.array([1.0, 0.0]),
        source_class_weight=np.array([0.0, 2.0]), static_phase=np.array([1.0, 0.0]),
        residue_id=np.arange(2),
    )
    positions = np.array([[0.0, 0.0, 0.0], [g.r_env + offset, 0.0, 0.0]])
    ref = compute_local_cwld(positions, None, p, [], g)
    density, chain = _squared_density_and_chain(
        positions, None, p, [], g, ref.lambda_, ref.dQ_ddensity,
    )
    np.testing.assert_allclose(density, ref.density, rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(chain, ref.force_chain, rtol=1e-10, atol=1e-12)
    if offset >= 0:
        np.testing.assert_array_equal(density, 0.0)
        np.testing.assert_array_equal(chain, 0.0)


def test_precomputed_response_coefficients(case):
    ref = _compute(case)
    p, g = case.particles, case.globals
    amplitude = p.static_phase * p.is_polar * p.dpolar
    t = np.tanh((g.k_polar / g.rho0) * ref.density)
    outer = np.tanh(amplitude * t / g.q_delta_clamp)
    q = p.qbase + g.q_delta_clamp * outer
    dq = amplitude * (g.k_polar / g.rho0) * (1 - outer * outer) * (1 - t * t)
    np.testing.assert_allclose(q, ref.Q, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dq, ref.dQ_ddensity, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(q[amplitude == 0], p.qbase[amplitude == 0])
    np.testing.assert_array_equal(dq[amplitude == 0], 0.0)


@pytest.mark.parametrize("energy,forces", [(False, False), (False, True), (True, False), (True, True)])
def test_energy_force_flags(case, energy, forces):
    ref = _compute(case)
    result = _compute(case, include_energy=energy, include_forces=forces)
    assert result.energy_total == (ref.energy_total if energy else 0.0)
    np.testing.assert_array_equal(result.force_total, ref.force_total if forces else np.zeros_like(case.positions))


def test_triclinic_translation_permutation_and_net_force(case):
    ref = _compute(case)
    p = case.particles
    rng = np.random.default_rng(911)
    shifted = case.positions + rng.integers(-2, 3, (p.num_particles, 3)) @ case.box
    shifted += np.array([0.17, -0.09, 0.23])
    perm = rng.permutation(p.num_particles)
    inverse = np.argsort(perm)
    reordered = ParticleArrays(**{f.name: getattr(p, f.name)[perm] for f in fields(p)})
    exclusions = [(int(inverse[i]), int(inverse[j])) for i, j in case.exclusions]
    result = compute_local_cwld(shifted[perm], case.box, reordered, exclusions, case.globals)
    np.testing.assert_allclose(result.energy_total, ref.energy_total, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(result.force_total[inverse], ref.force_total, rtol=1e-11, atol=1e-9)
    np.testing.assert_allclose(ref.force_total.sum(axis=0), 0.0, rtol=0.0, atol=1e-10)


def test_total_force_finite_difference_converges(case, record_property):
    ref = _compute(case)
    errors = []
    for h in (1e-4, 1e-5, 1e-6):
        fd = np.zeros_like(case.positions)
        for i in range(case.particles.num_particles):
            for axis in range(3):
                plus, minus = case.positions.copy(), case.positions.copy()
                plus[i, axis] += h
                minus[i, axis] -= h
                ep = compute_local_cwld(plus, case.box, case.particles, case.exclusions, case.globals, include_forces=False)
                em = compute_local_cwld(minus, case.box, case.particles, case.exclusions, case.globals, include_forces=False)
                fd[i, axis] = -(ep.energy_total - em.energy_total) / (2 * h)
        error = float(np.max(np.abs(fd - ref.force_total)))
        errors.append(error)
        record_property(f"fd_max_abs_h{h:g}", error)
    print(f"ell={case.globals.zmm_order} penalty={case.globals.use_q_penalty} FD max_abs={errors}")
    assert errors[1] < 0.1 * errors[0], errors
    assert errors[2] < errors[1], errors
    assert errors[-1] < 1e-5, errors  # kJ/mol/nm, this synthetic float64 FD experiment only


def _pair_terms(order, r, rc, qi, qj, di, dj, dtype, stable):
    """Return E/C and (dE/dr)/C, keeping every operation in the chosen dtype."""
    t = dtype
    r, rc, qi, qj, di, dj = (t(v) for v in (r, rc, qi, qj, di, dj))
    coefficients = {
        1: (-1.5, 0.5, 0.0, 0.0),
        2: (-15 / 8, 5 / 4, -3 / 8, 0.0),
        3: (-35 / 16, 35 / 16, -21 / 16, 5 / 16),
    }
    c0, c2, c4, c6 = map(t, coefficients[order])
    invrc, invr = t(1) / rc, t(1) / r
    x = r * invrc
    x2 = x * x
    polynomial = (c0 + x2 * (c2 + x2 * (c4 + x2 * c6))) * invrc
    derivative = x * (t(2) * c2 + x2 * (t(4) * c4 + x2 * t(6) * c6)) * invrc * invrc
    qprod = qi * qj
    Qprod = (qi + di) * (qj + dj)
    if stable:
        difference = qi * dj + qj * di + di * dj
        energy = difference * invr + Qprod * polynomial + qprod * invrc
        force = -difference * invr * invr + Qprod * derivative
    else:
        energy = Qprod * (invr + polynomial) - qprod * (invr - invrc)
        force = Qprod * (-invr * invr + derivative) + qprod * invr * invr
    return energy, force


@pytest.mark.parametrize("delta", [0.0, 1e-8, 0.01])
def test_rearranged_pair_matches_float64_reference(order, delta):
    qi, qj, rc = -0.834, 2.0, 1.2
    for r in np.geomspace(0.08, rc * (1 - 1e-8), 64):
        energy, force = _pair_terms(order, r, rc, qi, qj, delta, -2 * delta, np.float64, True)
        Qprod = (qi + delta) * (qj - 2 * delta)
        expected_energy = Qprod * zmm_uclosure(order, r, rc) - qi * qj * (1 / r - 1 / rc)
        expected_force = Qprod * zmm_duclosure_dr(order, r, rc) + qi * qj / (r * r)
        np.testing.assert_allclose(energy, expected_energy, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(force, expected_force, rtol=1e-11, atol=1e-12)


def _decimal_pair_derivative(order, values):
    """Independent 70-digit formula, at exactly the same float32 inputs."""
    with localcontext() as ctx:
        ctx.prec = 70
        r, rc, qi, qj, di, dj = (Decimal.from_float(float(np.float32(v))) for v in values)
        if order == 1:
            du = -1 / r**2 + r / rc**3
        elif order == 2:
            du = -1 / r**2 + 5 * r / (2 * rc**3) - 3 * r**3 / (2 * rc**5)
        else:
            du = -1 / r**2 + 35 * r / (8 * rc**3) - 21 * r**3 / (4 * rc**5) + 15 * r**5 / (8 * rc**7)
        return float((qi + di) * (qj + dj) * du + qi * qj / r**2)


@pytest.mark.parametrize("delta", [0.0, 1e-8, 1e-3], ids=["fixed", "tiny-response", "response"])
def test_float32_pair_cancellation_experiment(order, delta, record_property):
    """Aggregate error over a declared domain, not 'better at every point'.

    NumPy float32 does not emulate GPU FMA, reductions, or coordinate storage.
    Both expressions use the same Horner polynomial to isolate cancellation.
    """
    original_errors, candidate_errors = [], []
    for r in np.geomspace(0.08, 0.6, 128):
        values = (r, 1.2, -0.834, 2.0, delta, -2 * delta)
        truth = _decimal_pair_derivative(order, values)
        for stable, errors in [(False, original_errors), (True, candidate_errors)]:
            _, force = _pair_terms(order, *values, np.float32, stable)
            assert isinstance(force, np.float32), "accidental float64 promotion"
            errors.append(float(force) - truth)
    original_rmse = float(np.sqrt(np.mean(np.square(original_errors))))
    candidate_rmse = float(np.sqrt(np.mean(np.square(candidate_errors))))
    record_property("original_rmse_dEdr_over_C", original_rmse)
    record_property("candidate_rmse_dEdr_over_C", candidate_rmse)
    record_property("original_max_abs", float(np.max(np.abs(original_errors))))
    record_property("candidate_max_abs", float(np.max(np.abs(candidate_errors))))
    print(f"float32 ell={order} delta={delta:g}: RMSE original={original_rmse:.6g} candidate={candidate_rmse:.6g}")
    assert np.isfinite(candidate_rmse)
    assert candidate_rmse < original_rmse, (original_rmse, candidate_rmse)
