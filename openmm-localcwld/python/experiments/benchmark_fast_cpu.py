"""Explicit opt-in benchmark: pytest .../benchmark_fast_cpu.py -q -s.

Not named test_*.py, so ordinary pytest discovery does not run timings. CPU
analytic float32 only (DEC-004); complete compute includes rebuilding the
neighbor list. The reference it is compared against is float64, so the timing
ratio mixes an algorithmic change with a precision change and is not a
float32-vs-float64 measurement of the same algorithm.
This is NOT a comparison against CustomGB/GPU MD, and has no timing assertions.
"""

from dataclasses import fields
from statistics import median
from time import perf_counter
import platform

import numpy as np
import pytest
import scipy

from localcwld_fast import LocalCWLDFastEvaluator
from localcwld_reference import Globals, ParticleArrays, compute_local_cwld


@pytest.mark.parametrize("n", [32, 128, 512])
def test_cpu_snapshot_speed(n, record_property):
    rng = np.random.default_rng(881)
    side = int(np.ceil(n**(1/3)))
    grid = np.indices((side, side, side)).reshape(3, -1).T[:n]
    positions = .30 * grid + rng.uniform(-.015, .015, (n, 3))
    box = np.eye(3) * max(3., side * .30 + .6)
    p = ParticleArrays(qbase=rng.uniform(-1, 1, n), charge_mod=rng.uniform(.5, 2, n),
                      dpolar=rng.uniform(-.15, .15, n), is_polar=np.ones(n),
                      dens_source=(np.arange(n) % 3 == 0).astype(float), dens_sink=np.ones(n),
                      source_class_weight=np.ones(n), static_phase=np.ones(n), residue_id=np.arange(n)//3)
    exclusions = [(i, i+1) for i in range(0, n-1, 3)]
    g = Globals(r_env=.35, rc=1.2, use_q_penalty=True)
    start = perf_counter()
    evaluator = LocalCWLDFastEvaluator(p, exclusions, g)
    preparation = perf_counter() - start
    reference_call = lambda: compute_local_cwld(positions, box, p, exclusions, g)
    fast_call = lambda: evaluator.compute(positions, box)

    # Correctness before timing; both calls request all energy/force outputs.
    reference = reference_call()
    fast = fast_call()
    for field in fields(reference):
        # Single-precision agreement band; see test_fast_evaluator._RTOL/_ATOL.
        np.testing.assert_allclose(getattr(fast, field.name), getattr(reference, field.name),
                                   rtol=2e-4, atol=2e-3, err_msg=field.name)
    assert np.max(np.abs(reference.force_chain)) > 0
    reference_times, fast_times = [], []
    for repeat in range(5):
        # Alternate order to reduce simple warm-cache/order bias.
        calls = [(reference_call, reference_times), (fast_call, fast_times)]
        if repeat % 2:
            calls.reverse()
        for function, samples in calls:
            start = perf_counter()
            function()
            samples.append(perf_counter() - start)
    ref_median, fast_median = median(reference_times), median(fast_times)
    stats = {
        "n": n, "precision": "float32", "backend": "CPU NumPy/SciPy analytic",
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "prepare_ms": preparation * 1000, "reference_median_ms": ref_median * 1000,
        "fast_median_ms": fast_median * 1000, "speedup": ref_median / fast_median,
        "reference_samples_ms": [v * 1000 for v in reference_times],
        "fast_samples_ms": [v * 1000 for v in fast_times],
        "force_max_abs_error": float(np.max(np.abs(fast.force_total-reference.force_total))),
    }
    for key, value in stats.items():
        record_property(key, value)
    print(stats)
