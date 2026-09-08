"""Sparse, vectorized single-precision implementation for offline CWLD work.

Everything here is float32, end to end, because that is what the production
simulation actually does: the CUDA platform evaluates forces in single
precision, so a float64 offline backend would be measuring a kernel nobody
runs. See docs/decisions/DEC-004-precision.md.

The auditable localcwld_reference implementation remains float64 and unchanged;
it is the oracle this backend is checked against, not a parallel production
path. This module implements the ANALYTIC kernel, not the unresolved v2.6
tabulated compatibility mode. Never install it as a per-step Python callback in
a GPU simulation.
"""

from dataclasses import fields
import math

import numpy as np
from scipy.spatial import cKDTree

from localcwld_reference import Globals, ParticleArrays
from localcwld_reference.reference import LocalCWLDResult


class LocalCWLDFastEvaluator:
    """Prepare fixed parameters once, then evaluate changing positions/boxes.

    All arithmetic and all returned arrays are float32 -- parameters, geometry,
    density, Q, energies and forces alike. There is no precision switch: this
    backend exists to model what the single-precision GPU kernel will compute,
    and a float64 mode would quietly hide the rounding the real run has.

    Parameters are copied and made read-only; create a new evaluator to update
    them. Positions and neighbor lists are NOT cached: every compute() rebuilds
    neighbors, density, Q, adjoints and forces. Memory scales with candidate
    neighbors, not a dense N-by-N distance matrix (dense clusters can still
    have O(N^2) neighbors).

    Supports no box, orthorhombic boxes and OpenMM reduced triclinic boxes.
    Units and returned diagnostics match localcwld_reference, but agreement
    with it is at float32 level (~1e-6 relative), not bit-for-bit.
    Cutoff conditions are strict; approximate neighbor search is disabled.
    """

    #: Working precision for every array this class produces.
    DTYPE = np.float32

    def __init__(self, particles: ParticleArrays, exclusions, globals_: Globals):
        qbase = np.asarray(particles.qbase)
        if qbase.ndim != 1:
            raise ValueError("qbase must be a one-dimensional array")
        self._n = len(qbase)
        arrays = {}
        for field in fields(ParticleArrays):
            value = np.asarray(getattr(particles, field.name))
            if value.shape != (self._n,):
                raise ValueError(f"{field.name} must have shape ({self._n},)")
            if field.name == "residue_id":
                if not np.issubdtype(value.dtype, np.integer):
                    raise ValueError("residue_id must contain integers")
                value = np.array(value, dtype=np.int64, copy=True)
            else:
                # Reject non-finite input before the cast, then again after it:
                # a value that is finite as a double can overflow to inf in
                # float32, and a silent inf is worse than a rejected parameter.
                if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
                    raise ValueError(f"{field.name} contains non-finite values")
                value = np.array(value, dtype=self.DTYPE, copy=True)
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"{field.name} overflows float32")
            value.flags.writeable = False
            arrays[field.name] = value
        self._p = ParticleArrays(**arrays)
        self._g = Globals(**{f.name: getattr(globals_, f.name) for f in fields(Globals)})
        for name in ("k_polar", "one_4pi_eps0", "q_penalty_strength"):
            if not math.isfinite(getattr(self._g, name)):
                raise ValueError(f"{name} must be finite")

        excluded = set()
        for pair in exclusions:
            if len(pair) != 2:
                raise ValueError("each exclusion must contain two indices")
            i, j = pair
            if not all(isinstance(v, (int, np.integer)) for v in pair):
                raise ValueError("exclusion indices must be integers")
            if not (0 <= i < self._n and 0 <= j < self._n) or i == j:
                raise ValueError(f"invalid exclusion ({i}, {j})")
            i, j = sorted((int(i), int(j)))
            excluded.add(i * self._n + j)
        self._exclusions = np.array(sorted(excluded), dtype=np.int64)
        self._source = self._p.dens_source * self._p.source_class_weight * self._p.charge_mod
        self._amplitude = self._p.static_phase * self._p.is_polar * self._p.dpolar
        self._active = np.flatnonzero(self._amplitude != 0.0)
        self._env2 = self._g.r_env * self._g.r_env
        self._rc2 = self._g.rc * self._g.rc
        self._invrc = 1.0 / self._g.rc
        self._k_over_rho = self._g.k_polar / self._g.rho0
        self._coefficients = {
            1: (-1.5, 0.5, 0.0, 0.0),
            2: (-15 / 8, 5 / 4, -3 / 8, 0.0),
            3: (-35 / 16, 35 / 16, -21 / 16, 5 / 16),
        }[self._g.zmm_order]

    @classmethod
    def _box(cls, box_vectors):
        if box_vectors is None:
            return None
        box = np.asarray(box_vectors, dtype=cls.DTYPE)
        if box.shape != (3, 3) or not np.all(np.isfinite(box)):
            raise ValueError("box must be a finite (3, 3) array")
        if np.any(np.diag(box) <= 0) or np.any(box[np.triu_indices(3, 1)] != 0):
            raise ValueError("box must use OpenMM reduced triclinic orientation")
        if (abs(box[1, 0]) > box[0, 0] / 2 or abs(box[2, 0]) > box[0, 0] / 2
                or abs(box[2, 1]) > box[1, 1] / 2):
            raise ValueError("box must be reduced triclinic")
        return box

    def _neighbors(self, positions, box):
        if box is None:
            tree_positions = positions
            radius = self._g.rc
            boxsize = None
        elif np.count_nonzero(box - np.diag(np.diag(box))) == 0:
            boxsize = np.diag(box)
            tree_positions = np.mod(positions, boxsize)
            radius = self._g.rc
        else:
            # If |d_cart| < rc, then |d_fractional| <= rc/sigma_min(box).
            # The torus nearest fractional displacement cannot be longer than
            # that image. This conservative sphere therefore includes every
            # relevant Cartesian pair, even though the metrics differ.
            fractional = positions @ np.linalg.inv(box)
            tree_positions = np.mod(fractional, 1.0)
            radius = self._g.rc / np.linalg.svd(box, compute_uv=False)[-1]
            boxsize = 1.0
        if boxsize is not None:
            # remainder(-epsilon, L) can round to L, which cKDTree rejects.
            # Map that endpoint to its equivalent periodic image at zero.
            tree_positions = np.where(tree_positions >= boxsize, 0.0, tree_positions)
        # Search conservatively around roundoff; the exact Cartesian cutoff
        # below is authoritative. This is not a physical cutoff enlargement.
        # The margin is sized in float32 eps, which is what the coordinates
        # actually carry -- cKDTree widens to float64 internally, but that only
        # makes the candidate sphere more generous, never less.
        margin = (64 * np.finfo(self.DTYPE).eps
                  * max(1.0, float(radius), float(np.max(np.abs(positions), initial=0))))
        tree = cKDTree(tree_positions, boxsize=boxsize)
        pairs = tree.query_pairs(radius + margin, eps=0.0, output_type="ndarray")
        if self._exclusions.size and pairs.size:
            keys = pairs[:, 0] * self._n + pairs[:, 1]
            pairs = pairs[~np.isin(keys, self._exclusions)]
        i, j = pairs[:, 0], pairs[:, 1]
        delta = positions[j] - positions[i]
        if box is not None:
            for axis in (2, 1, 0):
                delta -= np.rint(delta[:, axis] / box[axis, axis])[:, None] * box[axis]
        r2 = np.einsum("ij,ij->i", delta, delta)
        # The float64 version of this backend re-evaluated a roundoff-sized
        # shell around r_env / rc with a scalar dot, so that pair membership
        # matched the scalar Reference bit for bit. In float32 that trick is
        # pointless: membership within ~1e-7 relative of a cutoff is genuinely
        # ambiguous at this precision, exactly as it is on the GPU, and the
        # shell is now wide enough that the Python loop over it would be a real
        # cost. Single precision means a pair right on the cutoff may be
        # included here and excluded by the Reference; that is expected.
        if np.any(r2 <= 0):
            raise ValueError("zero pair distance for non-excluded particles")
        distances = np.sqrt(r2)
        keep = distances < self._g.rc
        return i[keep], j[keep], delta[keep], r2[keep], distances[keep]

    def _sum(self, indices, weights):
        # NumPy ships no float32 bincount: it accumulates in float64 and we
        # narrow the result. That is a library limitation, not a precision
        # choice -- every value that leaves this method is float32. The GPU
        # port will use a native float32 scatter-add instead, so per-particle
        # sums there can differ from these by float32 rounding.
        return np.bincount(indices, weights=weights, minlength=self._n).astype(self.DTYPE)

    def _forces(self, i, j, pair_forces):
        return np.column_stack([
            self._sum(i, pair_forces[:, axis]) - self._sum(j, pair_forces[:, axis])
            for axis in range(3)
        ])

    def compute(self, positions, box_vectors=None, *, include_forces=True, include_energy=True):
        """Evaluate a snapshot, including current neighbor search in this call.

        Diagnostics remain available for every flag combination, matching the
        Reference API; forces/energies themselves are zero when not requested.
        No input is mutated and there is no hidden parameter or neighbor reuse.
        """
        positions = np.asarray(positions, dtype=self.DTYPE)
        if positions.shape != (self._n, 3) or not np.all(np.isfinite(positions)):
            raise ValueError(f"positions must be a finite ({self._n}, 3) array")
        box = self._box(box_vectors)
        i, j, delta, r2, distances = self._neighbors(positions, box)
        p, g = self._p, self._g

        edge = ((distances < g.r_env) & (p.residue_id[i] != p.residue_id[j])
                & (((p.dens_sink[i] != 0) & (self._source[j] != 0))
                   | ((p.dens_sink[j] != 0) & (self._source[i] != 0))))
        ei, ej = i[edge], j[edge]
        t = 1.0 - r2[edge] / self._env2
        density = p.dens_sink * (
            self._sum(ei, self._source[ej] * t * t)
            + self._sum(ej, self._source[ei] * t * t)
        )
        if not np.all(np.isfinite(density)):
            raise FloatingPointError("density contains non-finite values")
        active = self._active
        inner = np.tanh(self._k_over_rho * density[active])
        outer = np.tanh(self._amplitude[active] * inner / g.q_delta_clamp)
        delta_q = np.zeros(self._n, dtype=self.DTYPE)
        delta_q[active] = g.q_delta_clamp * outer
        q = p.qbase + delta_q
        dq = np.zeros(self._n, dtype=self.DTYPE)
        dq[active] = self._amplitude[active] * self._k_over_rho * (1 - outer * outer) * (1 - inner * inner)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            raise FloatingPointError("Q or dQ/ddensity contains non-finite values")

        energy_direct = energy_penalty = 0.0
        adjoint = np.zeros(self._n, dtype=self.DTYPE)
        direct = np.zeros((self._n, 3), dtype=self.DTYPE)
        chain = np.zeros((self._n, 3), dtype=self.DTYPE)
        if include_energy or include_forces:
            invr = 1.0 / distances
            x2 = r2 * self._invrc * self._invrc
            c0, c2, c4, c6 = self._coefficients
            polynomial = (c0 + x2 * (c2 + x2 * (c4 + x2 * c6))) * self._invrc
            closure = invr + polynomial
            qprod = q[i] * q[j]
            difference = p.qbase[i] * delta_q[j] + p.qbase[j] * delta_q[i] + delta_q[i] * delta_q[j]
            adjoint = g.one_4pi_eps0 * (
                self._sum(i, q[j] * closure) + self._sum(j, q[i] * closure)
            )
            if include_energy:
                energy_direct = float(g.one_4pi_eps0 * np.sum(
                    difference * invr + qprod * polynomial + p.qbase[i] * p.qbase[j] * self._invrc,
                ))
            if g.use_q_penalty:
                adjoint += g.q_penalty_strength * delta_q
                if include_energy:
                    energy_penalty = float(0.5 * g.q_penalty_strength * np.dot(delta_q, delta_q))
            if include_forces:
                # P'(r)/r: the radial normalization cancels analytically.
                derivative_over_r = (2 * c2 + x2 * (4 * c4 + 6 * c6 * x2)) * self._invrc**3
                scale = g.one_4pi_eps0 * (-difference * invr**3 + qprod * derivative_over_r)
                direct = self._forces(i, j, scale[:, None] * delta)
                response = adjoint * dq * p.dens_sink
                scale = (-4.0 / self._env2 * t
                         * (response[ei] * self._source[ej] + response[ej] * self._source[ei]))
                chain = self._forces(ei, ej, scale[:, None] * delta[edge])
        return LocalCWLDResult(density, q, dq, adjoint, energy_direct, energy_penalty, direct, chain)
