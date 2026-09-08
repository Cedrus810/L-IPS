#!/usr/bin/env python3
"""Pins the gate's diagnosis boundary against the two real LCWLD-161 runs.

Mirrors the predicate in BenchmarkCudaLocalCWLD.cpp (NOISE / 0.05 decoupling
threshold). It is a copy, not the real code -- if you change the thresholds
there, change them here. It exists because the boundary, not the code, is what
was wrong: a 0.9% wiggle in `new` used to read as "another process on the GPU".
"""
NOISE, DECOUPLE = 0.02, 0.05


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def classify(gb, new):
    direction, monotonic = 0, True
    for arm in (gb, new):
        if not monotonic:
            break
        for a, b in zip(arm, arm[1:]):
            rel = (b - a) / median(arm)
            if abs(rel) < NOISE:
                continue
            d = -1 if rel < 0 else 1
            if direction == 0:
                direction = d
            elif d != direction:
                monotonic = False
                break
    if not monotonic:
        return "contention"
    drift = lambda x: (x[0] - x[-1]) / median(x)
    return "decoupled" if abs(drift(gb) - drift(new)) > DECOUPLE else "thermal"


# Run A -- old strict ForceInfo. Both arms heavy, both throttle together.
assert classify([59.08, 55.12, 52.01, 49.31],
                [91.40, 85.51, 80.37, 76.12]) == "thermal"

# Run B -- after the LCWLD-161 fix. `new` got fast enough to stop heating the
# card, so it no longer drifts with gb. Used to be misreported as contention.
assert classify([57.96, 54.06, 51.00, 48.73],
                [178.78, 178.83, 177.17, 177.50]) == "decoupled"

# Real contention, 2026-09-04: gb starved on one round only, others normal.
assert classify([59.09, 24.64, 59.43, 59.79],
                [91.40, 85.51, 80.37, 76.12]) == "contention"

print("ok: thermal / decoupled / contention all classify correctly")
