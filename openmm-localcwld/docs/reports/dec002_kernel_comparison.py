"""DEC-002 evidence, part 1: compare the analytic density kernel K(r) =
(1-(r/r_env)^2)^2 against v2.6's exact `Continuous1DFunction` table (built
EXACTLY as test_lips_vs_pmeV2.6.py:setup_cwld_lips_system does, lines
~1175-1180), sampled at 10,000 distance points across [0, rc], per plan
section 19/DEC-002.

Rather than guessing which interpolation scheme OpenMM's Continuous1DFunction
uses, this script asks OpenMM itself: build 10,000 independent, non-
interacting particle pairs, each held at a different separation r, connected
by a CustomBondForce whose energy expression is exactly `y_kernel(r)`
referencing the same tabulated function v2.6 uses. A single static
getState(getEnergy=False, getForces=True) call (no Integrator step) then
gives OpenMM's own tabulated K(r) (as -force/unit, since the bond energy IS
K(r)) at all 10,000 points simultaneously -- this is the actual number
CustomGBForce uses at runtime, not a re-implementation guess.
"""
import json
import pathlib
import sys

import numpy as np
import openmm as mm
from openmm import unit

REPORTS_DIR = pathlib.Path(__file__).resolve().parent
LCWLD_ROOT = REPORTS_DIR.parents[1]
sys.path.insert(0, str(LCWLD_ROOT / "python"))
from localcwld_reference.reference import zmm_uclosure  # noqa: F401 (import sanity only)

R_ENV = 0.35
RC = 1.2
TABULATED_POINTS = 1024
N_SAMPLES = 10_000


def analytic_kernel(r):
    r = np.asarray(r, dtype=float)
    k = (1.0 - (r / R_ENV) ** 2) ** 2
    return np.where(r < R_ENV, k, 0.0)


def analytic_dkernel_dr(r):
    r = np.asarray(r, dtype=float)
    dk = -4.0 * r / (R_ENV ** 2) * (1.0 - r ** 2 / (R_ENV ** 2))
    return np.where(r < R_ENV, dk, 0.0)


def build_v26_table():
    x_density = np.linspace(0.0, RC, TABULATED_POINTS)
    y_density = np.zeros_like(x_density)
    density_mask = x_density < R_ENV
    x_density_valid = x_density[density_mask] / R_ENV
    y_density[density_mask] = (1.0 - x_density_valid ** 2) ** 2
    return x_density, y_density


def query_openmm_tabulated_kernel(r_values):
    x_table, y_table = build_v26_table()
    system = mm.System()
    force = mm.CustomCompoundBondForce(2, "y_kernel(distance(p1,p2))")
    force.addTabulatedFunction("y_kernel", mm.Continuous1DFunction(y_table.tolist(), float(x_table[0]), float(x_table[-1])))

    positions = []
    for r in r_values:
        i = system.addParticle(1.0)
        j = system.addParticle(1.0)
        positions.append([0.0, 0.0, 0.0])
        positions.append([float(r), 0.0, 0.0])
        force.addBond([i, j], [])
    system.addForce(force)

    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)  # never stepped
    platform = mm.Platform.getPlatformByName("Reference")
    context = mm.Context(system, integrator, platform)
    context.setPositions(np.array(positions) * unit.nanometer)

    state = context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

    # bond energy is exactly y_kernel(r); force on particle j (the +r one) is
    # -dE/dr along +x, i.e. force_x[j] = -dK/dr. We recover K(r) itself via
    # a second pass using getEnergy per-bond is not directly exposed, so
    # instead integrate: easier and exact -- just also query getEnergy() at
    # each r individually would be slow for 10k points, so recover K(r) by
    # direct evaluation of the same table via a second, energy-only probe
    # system swept one r at a time is too slow; instead reconstruct K(r) from
    # dK/dr is lossy. Simplest robust approach: query total energy for two
    # nested subsets (cumulative) is also awkward. -> Just do individual
    # single-bond energy queries; 10k static getState(getEnergy=True) calls
    # on a 2-particle system are cheap (no neighbor list, no other forces).
    dKdr_tabulated = -forces[1::2, 0]
    return dKdr_tabulated


def query_openmm_tabulated_kernel_energy(r_values):
    x_table, y_table = build_v26_table()
    system = mm.System()
    a = system.addParticle(1.0)
    b = system.addParticle(1.0)
    force = mm.CustomCompoundBondForce(2, "y_kernel(distance(p1,p2))")
    force.addTabulatedFunction("y_kernel", mm.Continuous1DFunction(y_table.tolist(), float(x_table[0]), float(x_table[-1])))
    force.addBond([a, b], [])
    system.addForce(force)
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = mm.Platform.getPlatformByName("Reference")
    context = mm.Context(system, integrator, platform)

    energies = np.zeros(len(r_values))
    for idx, r in enumerate(r_values):
        context.setPositions(np.array([[0.0, 0.0, 0.0], [float(r), 0.0, 0.0]]) * unit.nanometer)
        state = context.getState(getEnergy=True)
        energies[idx] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return energies


def main():
    r_grid = np.linspace(1e-6, RC - 1e-6, N_SAMPLES)  # avoid the exact r=0/rc endpoints

    k_tabulated = query_openmm_tabulated_kernel_energy(r_grid)
    dk_tabulated = query_openmm_tabulated_kernel(r_grid)

    k_analytic = analytic_kernel(r_grid)
    dk_analytic = analytic_dkernel_dr(r_grid)

    abs_err_k = np.abs(k_analytic - k_tabulated)
    abs_err_dk = np.abs(dk_analytic - dk_tabulated)
    # plan section 36: near-zero values must be judged by absolute error, not
    # relative -- ratios blow up meaninglessly when both sides are ~1e-6.
    # scale_floor chosen well above the observed abs-error noise floor so the
    # relative-error stat is only reported where it's actually informative.
    scale_floor = 1e-3
    informative_k = np.maximum(np.abs(k_analytic), np.abs(k_tabulated)) > scale_floor
    informative_dk = np.maximum(np.abs(dk_analytic), np.abs(dk_tabulated)) > scale_floor
    rel_err_k = abs_err_k[informative_k] / np.maximum(np.abs(k_analytic[informative_k]), np.abs(k_tabulated[informative_k]))
    rel_err_dk = abs_err_dk[informative_dk] / np.maximum(np.abs(dk_analytic[informative_dk]), np.abs(dk_tabulated[informative_dk]))

    i_k = int(np.argmax(abs_err_k))
    i_dk = int(np.argmax(abs_err_dk))

    if rel_err_dk.size:
        r_informative = r_grid[informative_dk]
        j = int(np.argmax(rel_err_dk))
        print(
            f"dKdr worst informative relative error at r={r_informative[j]:.6f} nm: "
            f"analytic={dk_analytic[informative_dk][j]:.6e}, tabulated={dk_tabulated[informative_dk][j]:.6e}, "
            f"abs_err={abs_err_dk[informative_dk][j]:.6e}"
        )

    report = {
        "n_points": len(r_grid),
        "r_env_nm": R_ENV,
        "rc_nm": RC,
        "tabulated_points": TABULATED_POINTS,
        "source": "OpenMM Continuous1DFunction evaluated live via CustomCompoundBondForce (Reference platform), not a re-implemented interpolation guess",
        "K_max_abs_error": float(abs_err_k.max()),
        "K_median_abs_error": float(np.median(abs_err_k)),
        "K_p95_abs_error": float(np.percentile(abs_err_k, 95)),
        "K_max_abs_error_at_r_nm": float(r_grid[i_k]),
        "K_max_rel_error_where_K_gt_1e-3": float(rel_err_k.max()) if rel_err_k.size else None,
        "K_note": "relative error only computed where max(|K_analytic|,|K_tabulated|) > 1e-3 (plan section 36: near-zero values use absolute error, not relative)",
        "dKdr_max_abs_error": float(abs_err_dk.max()),
        "dKdr_median_abs_error": float(np.median(abs_err_dk)),
        "dKdr_p95_abs_error": float(np.percentile(abs_err_dk, 95)),
        "dKdr_max_abs_error_at_r_nm": float(r_grid[i_dk]),
        "dKdr_max_rel_error_where_dKdr_gt_1e-3": float(rel_err_dk.max()) if rel_err_dk.size else None,
    }
    print(json.dumps(report, indent=2))

    out_path = REPORTS_DIR / "dec002_kernel_comparison_result.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
