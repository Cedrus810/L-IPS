"""localcwld_reference: standalone, unoptimized float64 reference implementation
of the LocalCWLDForce math (plan sections 18 and 31).

Deliberately has NO dependency on OpenMM. It consumes plain arrays (the same
nine per-particle fields `openmm_localcwld.metadata.ParticleParameters`
produces, but as raw numpy arrays so this module can be exercised purely from
JSON fixtures) plus positions/box/exclusions/globals, and returns every
intermediate quantity the plan requires for auditing: density, Q, dQ/ddensity,
lambda (chargeGradient), direct/chain energy and forces.

This is a reproduction target, not a design: every formula here is copied
from PLAN_LocalCWLDForce_OpenMM_Plugin.md sections 18/31/32 without
"cleverness". Do not vectorize away the explicit double loops -- auditability
here matters more than speed (see plan section 31, "禁止优化、线程化"). Any
GPU/Reference-kernel implementation of LocalCWLDForce itself must match this
module's numbers, not the other way around.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

ONE_4PI_EPS0_DEFAULT = 138.935458


# ---------------------------------------------------------------------------
# ZMM closure (plan section 18.3 / 32)
# ---------------------------------------------------------------------------

def zmm_uclosure(order: int, r: float, rc: float) -> float:
    if order == 1:
        return 1.0 / r - 1.5 / rc + r ** 2 / (2.0 * rc ** 3)
    if order == 2:
        return (
            1.0 / r
            - 15.0 / (8.0 * rc)
            + 5.0 * r ** 2 / (4.0 * rc ** 3)
            - 3.0 * r ** 4 / (8.0 * rc ** 5)
        )
    if order == 3:
        return (
            1.0 / r
            - 35.0 / (16.0 * rc)
            + 35.0 * r ** 2 / (16.0 * rc ** 3)
            - 21.0 * r ** 4 / (16.0 * rc ** 5)
            + 5.0 * r ** 6 / (16.0 * rc ** 7)
        )
    raise ValueError(f"unsupported ZMM closure order={order} (must be 1, 2, or 3)")


def zmm_duclosure_dr(order: int, r: float, rc: float) -> float:
    if order == 1:
        return -1.0 / r ** 2 + r / rc ** 3
    if order == 2:
        return -1.0 / r ** 2 + 5.0 * r / (2.0 * rc ** 3) - 3.0 * r ** 3 / (2.0 * rc ** 5)
    if order == 3:
        return (
            -1.0 / r ** 2
            + 35.0 * r / (8.0 * rc ** 3)
            - 21.0 * r ** 3 / (4.0 * rc ** 5)
            + 15.0 * r ** 5 / (8.0 * rc ** 7)
        )
    raise ValueError(f"unsupported ZMM closure order={order} (must be 1, 2, or 3)")


# ---------------------------------------------------------------------------
# PBC minimum image (reduced triclinic form, matching OpenMM's convention:
# reduce along c, then b, then a -- see plan section 31.2/47).
# ---------------------------------------------------------------------------

def minimum_image(delta: np.ndarray, box_vectors: Optional[np.ndarray]) -> np.ndarray:
    if box_vectors is None:
        return delta
    a, b, c = box_vectors[0], box_vectors[1], box_vectors[2]
    delta = delta - c * round(delta[2] / c[2])
    delta = delta - b * round(delta[1] / b[1])
    delta = delta - a * round(delta[0] / a[0])
    return delta


@dataclass(frozen=True)
class Globals:
    r_env: float
    rc: float
    zmm_order: int = 2
    rho0: float = 13.5
    k_polar: float = 0.8
    q_delta_clamp: float = 0.20
    one_4pi_eps0: float = ONE_4PI_EPS0_DEFAULT
    use_q_penalty: bool = False
    q_penalty_strength: float = 180.0

    def __post_init__(self):
        if not (math.isfinite(self.r_env) and self.r_env > 0):
            raise ValueError(f"r_env must be finite and > 0, got {self.r_env}")
        if not (math.isfinite(self.rc) and self.rc > 0):
            raise ValueError(f"rc must be finite and > 0, got {self.rc}")
        if self.r_env > self.rc:
            raise ValueError(f"r_env ({self.r_env}) must be <= rc ({self.rc})")
        if self.zmm_order not in (1, 2, 3):
            raise ValueError(f"zmm_order must be 1, 2, or 3, got {self.zmm_order}")
        if not (math.isfinite(self.rho0) and self.rho0 > 0):
            raise ValueError(f"rho0 must be finite and > 0, got {self.rho0}")
        if not (math.isfinite(self.q_delta_clamp) and self.q_delta_clamp > 0):
            raise ValueError(f"q_delta_clamp must be finite and > 0, got {self.q_delta_clamp}")


@dataclass(frozen=True)
class ParticleArrays:
    qbase: np.ndarray
    charge_mod: np.ndarray
    dpolar: np.ndarray
    is_polar: np.ndarray
    dens_source: np.ndarray
    dens_sink: np.ndarray
    source_class_weight: np.ndarray
    static_phase: np.ndarray
    residue_id: np.ndarray

    @property
    def num_particles(self) -> int:
        return len(self.qbase)


@dataclass
class LocalCWLDResult:
    density: np.ndarray
    Q: np.ndarray
    dQ_ddensity: np.ndarray
    lambda_: np.ndarray  # dE/dQ_i ("chargeGradient" per plan section 48)
    energy_direct: float
    energy_penalty: float
    force_direct: np.ndarray
    force_chain: np.ndarray

    @property
    def energy_total(self) -> float:
        return self.energy_direct + self.energy_penalty

    @property
    def force_total(self) -> np.ndarray:
        return self.force_direct + self.force_chain


def _normalize_exclusions(exclusions):
    out = set()
    for i, j in exclusions:
        if i == j:
            raise ValueError(f"exclusion cannot reference the same particle ({i})")
        out.add((min(i, j), max(i, j)))
    return out


def compute_local_cwld(
    positions: np.ndarray,
    box_vectors: Optional[np.ndarray],
    particles: ParticleArrays,
    exclusions,
    globals_: Globals,
    *,
    include_forces: bool = True,
    include_energy: bool = True,
) -> LocalCWLDResult:
    """Naive O(N^2) reference implementation, following plan section 31's
    pass structure exactly: A) density, B) Q/dQ, C) pair energy/direct
    force/lambda, penalty, D) chain force. See plan section 48 for the
    includeEnergy/includeForces truth table this follows.
    """
    n = particles.num_particles
    if positions.shape != (n, 3):
        raise ValueError(f"positions shape {positions.shape} does not match num_particles {n}")
    excl = _normalize_exclusions(exclusions)

    def excluded(i, j):
        return (min(i, j), max(i, j)) in excl

    def geometry(i, j):
        delta = minimum_image(positions[j] - positions[i], box_vectors)
        r2 = float(np.dot(delta, delta))
        if r2 <= 0.0:
            raise ValueError(f"LocalCWLDForce reference: zero pair distance for ({i},{j})")
        r = math.sqrt(r2)
        unit = delta / r  # points from i to j
        return delta, unit, r, r2

    r_env = globals_.r_env
    rc = globals_.rc
    rho0 = globals_.rho0
    k_polar = globals_.k_polar
    q_delta_clamp = globals_.q_delta_clamp
    C = globals_.one_4pi_eps0
    order = globals_.zmm_order

    # --- Pass A: density (plan 18.1 / 31.4) ---
    density = np.zeros(n)
    for i in range(n):
        if particles.dens_sink[i] == 0.0:
            continue
        total = 0.0
        for j in range(n):
            if i == j:
                continue
            if particles.dens_source[j] == 0.0:
                continue
            if particles.residue_id[i] == particles.residue_id[j]:
                continue
            if excluded(i, j):
                continue
            _, _, r, _ = geometry(i, j)
            if r >= r_env:
                continue
            x = r / r_env
            kernel = (1.0 - x * x) ** 2
            source_amplitude = (
                particles.dens_source[j] * particles.source_class_weight[j] * particles.charge_mod[j]
            )
            total += source_amplitude * kernel
        density[i] = particles.dens_sink[i] * total

    if not np.all(np.isfinite(density)):
        raise FloatingPointError("density contains non-finite values")

    # --- Pass B: Q and dQ/ddensity (plan 18.2 / 31.5) ---
    Q = np.zeros(n)
    dQ_ddensity = np.zeros(n)
    for i in range(n):
        A = particles.static_phase[i] * particles.is_polar[i] * particles.dpolar[i]
        t = math.tanh(k_polar * density[i] / rho0)
        y = A * t / q_delta_clamp
        outer = math.tanh(y)
        Q[i] = particles.qbase[i] + q_delta_clamp * outer
        dQ_ddensity[i] = A * (k_polar / rho0) * (1.0 - outer * outer) * (1.0 - t * t)

    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(dQ_ddensity)):
        raise FloatingPointError("Q or dQ/ddensity contains non-finite values")

    # --- Pass C: pair energy, direct force, lambda (plan 18.4/18.5 / 31.6) ---
    energy_direct = 0.0
    lambda_ = np.zeros(n)
    force_direct = np.zeros((n, 3))
    for i in range(n):
        for j in range(i + 1, n):
            if excluded(i, j):
                continue
            _, unit, r, r2 = geometry(i, j)
            if r >= rc:
                continue
            U = zmm_uclosure(order, r, rc)
            dUdr = zmm_duclosure_dr(order, r, rc)
            shifted_rf = 1.0 / r - 1.0 / rc

            pair_energy = C * (Q[i] * Q[j] * U - particles.qbase[i] * particles.qbase[j] * shifted_rf)
            if include_energy:
                energy_direct += pair_energy

            # lambda must be computed whenever forces are requested, regardless
            # of include_energy (plan section 48 truth table).
            if include_forces or include_energy:
                lambda_[i] += C * Q[j] * U
                lambda_[j] += C * Q[i] * U

            if include_forces:
                dEdr = C * (Q[i] * Q[j] * dUdr + particles.qbase[i] * particles.qbase[j] / r2)
                force_on_i = dEdr * unit
                force_direct[i] += force_on_i
                force_direct[j] -= force_on_i

    # --- Penalty (plan 18.6 / 31.7) ---
    energy_penalty = 0.0
    if globals_.use_q_penalty:
        for i in range(n):
            delta_q = Q[i] - particles.qbase[i]
            if include_energy:
                energy_penalty += 0.5 * globals_.q_penalty_strength * delta_q * delta_q
            if include_forces or include_energy:
                lambda_[i] += globals_.q_penalty_strength * delta_q

    # --- Pass D: density chain force (plan 18.5 / 31.8) ---
    force_chain = np.zeros((n, 3))
    if include_forces:
        for i in range(n):
            for j in range(i + 1, n):
                if excluded(i, j):
                    continue
                if particles.residue_id[i] == particles.residue_id[j]:
                    continue
                _, unit, r, r2 = geometry(i, j)
                if r >= r_env:
                    continue
                x2 = r2 / (r_env * r_env)
                dkernel_dr = -4.0 * r / (r_env * r_env) * (1.0 - x2)

                Bi = particles.dens_source[i] * particles.source_class_weight[i] * particles.charge_mod[i]
                Bj = particles.dens_source[j] * particles.source_class_weight[j] * particles.charge_mod[j]
                gi = lambda_[i] * dQ_ddensity[i]
                gj = lambda_[j] * dQ_ddensity[j]

                dEdr_chain = (gi * particles.dens_sink[i] * Bj + gj * particles.dens_sink[j] * Bi) * dkernel_dr
                force_on_i = dEdr_chain * unit
                force_chain[i] += force_on_i
                force_chain[j] -= force_on_i

    return LocalCWLDResult(
        density=density,
        Q=Q,
        dQ_ddensity=dQ_ddensity,
        lambda_=lambda_,
        energy_direct=energy_direct if include_energy else 0.0,
        energy_penalty=energy_penalty if include_energy else 0.0,
        force_direct=force_direct,
        force_chain=force_chain,
    )
