# openmm-localcwld

Native OpenMM plugin for `LocalCWLDForce` — the local-environment charge
response (CWLD) currently implemented in `test_lips_vs_pmeV2.6.py` as a
`CustomGBForce`. The goal is to replace that generic CustomGB evaluation with
a purpose-built Reference/CUDA kernel.

**Why:** `CustomGBForce` is ~85 % of step time, so replacing it with a
purpose-built kernel is worth roughly **2–3×** on a single trajectory.

Measured directly on both systems with `RUN_MODE=profile_force_cost`:

| | full CWLD | CustomGBForce removed | share | f=0 ceiling |
|---|---:|---:|---:|---:|
| 1CKK, 31358 atoms | 131.04 | 691.88 | 81.1 % | 5.28× |
| **1AAY, 32794 atoms** | **63.06** | **407.83** | **84.5 %** | **6.47×** |

(1AAY: CUDA mixed, override off, no reporters or barostat, 200 warmup + 1000 steps.
1CKK raw log at `../../2.4/L-IPS.o9808:499-511`.)

The two systems agree to 3.4 points, so this is a property of CustomGBForce's
cost structure, not of a particular system. Amdahl with `s = 0.845`:

```
    f = 0    (free kernel)   6.47×   <- ceiling, not a target
    f = 1/3  (3× faster)     2.29×   48.4 GPU-h -> 21 h
    f = 1/5  (5× faster)     3.09×   48.4 GPU-h -> 16 h
```

⚠ **Superseded — plan for ≥ 2.0×, not 2–3×.** The Amdahl ceiling above is for
*removing* CustomGBForce entirely; the kernel design keeps its fixed cost. A
double-system `rc` scan measured the one number the design rests on directly,
at `rc = 0.35 = r_env`:

```
r_env / rc cost ratio = 0.109 / 0.846 = 0.129   (design assumed 0.025)
f = (1 + 2*0.129)/3   = 0.419
end-to-end            = 1.97x        48.4 GPU-h -> 24.6 h
```

Read as a bound: **f <= 0.419, end-to-end >= 1.97x** — the 0.129 was measured with
OpenMM's 32-atom tiles, and the design's compact flat list has no tile
granularity. The design's own ceiling is **2.32-3.63x**, not 6.47x.
See `docs/reports/LCWLD-090-100-design.md` section 1.

⚠ **Even a free kernel does not reach the PME baseline.** With CustomGBForce
removed entirely CWLD runs at 407.83 ns/day against PME's ~663 — still 1.63×
slower. The reason is a cutoff mismatch, not CWLD: the PME baseline uses
`nonbondedCutoff=1.0 nm` (`src/lips/engine/v26.py:1144`) while the CWLD system's
`NonbondedForce` uses `rc = 1.2 nm` (`v26.py:1246-1247`), which is
`(1.2/1.0)³ = 1.73×` the direct-space pairs. Do not promise parity with PME.

⚠ Three earlier versions of this figure were wrong. Kept here because each one
failed in an instructive way:
* **32 % / 1.5×** — back-computed from a *failed* MTS run (2:1 MTS gave 1.19×,
  but its temperature drifted 300 K → 324 K, so the 1.19× measures MTS being
  invalid for this force, not the force's cost).
* **~90 % / 10×** — two errors at once: 90 % came from comparing CWLD against the
  *PME baseline*, a different System; and 10× quoted the `f = 0` ceiling as the
  expected gain. Reaching 10× would need a kernel **50× faster** than
  CustomGBForce, which is already compiled CUDA.
* **s ≥ 92 % (lower bound)** — argued that `CutoffPeriodic` must be cheaper than
  PME. False here: the two use different cutoffs (1.0 vs 1.2 nm), so the CWLD
  system's nonbonded term is the *more* expensive of the two. Measurement put
  s at 84.5 %, and the ceiling at 6.47× rather than 12.5×.

The common thread: every wrong number came from reasoning where a measurement
was available. `profile_force_cost` takes under a minute.

Corollary from the same 1CKK log: the `cpu_kdtree_fast` engine measured
**14.35 ns/day — 8.9× *slower* than exact**, at 2.1 % of the ceiling. CPU-side
vectorisation inside the MD loop is a net loss here.

## Status

See `STATUS.md`. In short: `LCWLD-000/010/020/030` are done, the numerical
`LocalCWLDForce` C++/CUDA implementation (`040` onward) is not written yet.
Nothing here is loadable by OpenMM at runtime today.

## Building

Requires an OpenMM installation that provides `include/OpenMM.h` and
`lib/libOpenMM.so`, plus a C++17 compiler that is **libstdc++-ABI compatible
with that OpenMM build** (see `docs/decisions/DEC-001-toolchain.md`: the
conda-forge GCC inside the environment, not a much newer system GCC).

```bash
export OPENMM_PREFIX=/path/to/envs/openmm_dev

cmake -S . -B build \
  -DOPENMM_DIR="$OPENMM_PREFIX" \
  -DCMAKE_C_COMPILER="$OPENMM_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
  -DCMAKE_CXX_COMPILER="$OPENMM_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
  -DLOCALCWLD_BUILD_REFERENCE=OFF \
  -DLOCALCWLD_BUILD_CUDA=OFF \
  -DLOCALCWLD_BUILD_PYTHON=OFF

cmake --build build --parallel
cmake --install build --prefix stage
```

With `-DLOCALCWLD_BUILD_CUDA=ON`, also pass
`-DCMAKE_CUDA_COMPILER="$OPENMM_PREFIX/bin/nvcc"` — the toolkit that built
OpenMM, not whatever `nvcc` is first on `PATH`.

Build options: `LOCALCWLD_BUILD_REFERENCE`, `LOCALCWLD_BUILD_CUDA`,
`LOCALCWLD_BUILD_PYTHON`, `BUILD_TESTING` (all `OFF` by default). Turning CUDA
off means no CUDA toolkit is required to configure.

Install layout: public headers to `include/openmm`, the API library to `lib`,
platform plugins to `lib/plugins` (override with
`-DLOCALCWLD_PLUGIN_INSTALL_DIR`).

## Python packages in this tree

These are independent of the C++ build and are used offline, not inside a
simulation:

- `python/openmm_localcwld/metadata.py` — per-particle parameter builder,
  bit-identical to v2.6 on the real 1CKK system.
- `python/localcwld_reference/` — auditable explicit-loop NumPy reference
  (float64, deliberately unoptimized).
- `python/localcwld_fast/` — vectorized evaluator for static snapshots. Not an
  OpenMM `Force`; never install it as a per-step Python callback.

## Layout

Directory ownership per plan section 21; each ticket owns its own files.
Generated artifacts (SWIG wrappers, encoded kernel sources, `build*/`,
`stage/`) stay out of the source tree.
