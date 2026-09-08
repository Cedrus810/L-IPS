/**
 * LocalCWLDForce device kernels (LCWLD-100).
 *
 * Compiled at run time by whatever NVRTC the host OpenMM is linked against
 * (12.9 here), from the string table that cmake/EncodeKernelFiles.cmake builds
 * out of this file. Nothing in the build system picks the CUDA version; see
 * docs/decisions/DEC-001-toolchain.md.
 *
 * Precision: fp32 throughout (DEC-004). Per-particle accumulation goes through
 * the platform's 64-bit fixed-point buffers, NOT hand-rolled float reductions
 * -- that is what makes results independent of scatter order, and it is why
 * `single` and `mixed` were measured to be bit-identical for forces
 * (DEC-005 section 3.3).
 *
 * WHERE THE SPEEDUP COMES FROM (docs/reports/LCWLD-090-100-design.md):
 * CustomGBForce walks the full rc neighbour list three times -- once for the
 * computed value, once for the pair energy, once for the chain rule. But
 * density and the chain term only reach r_env. With r_env=0.35 and rc=1.2 the
 * volume ratio is 2.5%, so those two passes should be almost free.
 *
 * That only materialises with a SEPARATE, COMPACT r_env pair list that is
 * rebuilt on the same schedule as OpenMM's own neighbour list. Filtering the
 * rc list by r_env every step keeps the traversal cost and buys nothing:
 *
 * The right column is the END-TO-END speedup, not the kernel's own. It is
 * Amdahl over the measured CustomGBForce share s = 0.845 (1AAY):
 *
 *     overall = 1 / ((1 - s) + s * cost/3)
 *
 *     traversal structure                          cost   f=cost/3   overall
 *     3 full-rc passes (CustomGBForce)             3.000    1.000     1.00x
 *     reuse rc list, filter by r_env               3.000    1.000     1.00x
 *     rebuild the r_env sublist every step         2.025    0.675     1.30x
 *     r_env sublist amortised over rebuilds        1.050    0.350     2.22x
 *
 * The last row is the design. `buildEnvPairs` below is therefore called from
 * the host only when the neighbour list is rebuilt, and it uses a padded
 * radius so pairs that drift inside r_env between rebuilds are already
 * present. Every consumer re-tests the exact r_env, so the padding never
 * changes the physics -- only how often the list is regenerated.
 */

#define ENV_PAIRS_OVERFLOW_FLAG 0

/**
 * Read back a value the platform accumulated in 64-bit fixed point.
 *
 * OpenMM's preamble supplies `realToFixedPoint()` for the forward direction but
 * no inverse, because its own kernels never read these buffers back -- forces
 * are converted by the platform itself. We do read them back: `dens` and the
 * adjoint are intermediate per-particle quantities that later passes consume.
 *
 * The scale is 2^32, matching realToFixedPoint. The division goes through
 * double deliberately: the accumulator can hold more than 24 significant bits,
 * so converting the raw long long to float first would truncate it. This is a
 * single conversion per particle, not a compute path -- it does not make the
 * pipeline double precision (DEC-004).
 */
/**
 * The pair arithmetic, shared verbatim by all three tile loops.
 *
 * ⚠ **The arithmetic below is unchanged from the version that passed the
 * DEC-005 per-particle acceptance on real 1AAY.** Only the parallel structure
 * around it was rewritten (one-thread-per-pair -> warp tiles). Keeping the
 * expressions byte-identical is what lets the acceptance test act as a
 * regression check on the rewrite: if it still passes, the restructuring did
 * not disturb the physics.
 *
 * A macro rather than three copies, for the same reason as before: three
 * hand-kept copies of this drift, and the drift would be confined to one kind
 * of tile -- a small, spatially clustered force error, exactly what DEC-005
 * section 3.7 exists to catch.
 *
 * Sign, derived rather than guessed:
 *     r = |r2 - r1|,  delta = r2 - r1,  dr/dx1 = -delta_x/r
 *     F1_x = -dU/dx1 = (dU/dr)(delta_x/r) = scale*delta_x
 * so atom1 takes +scale*delta and atom2 takes -scale*delta. The first version
 * had these swapped, and nothing caught it: net force stays zero under a global
 * sign flip and the energy expression does not contain the force sign at all.
 * It took a per-particle comparison against production CustomGBForce, which
 * reported <got,ref>/|ref|^2 = -1.0000 at |got|/|ref| = 1.0000.
 *
 * Expects in scope: delta, r2, qb1, q1, dq1, qb2, q2, invRc, invRc2, invRc3,
 * c0..c6, one4PiEps0. Defines: pairEnergy, scale, closure.
 */
#define LOCALCWLD_PAIR_CORE                                                          \
    const real invR = RSQRT(r2);                                                     \
    const real x2 = r2 * invRc2;                                                     \
    const real dq2 = q2 - qb2;                                                       \
    const real poly = (c0 + x2 * (c2 + x2 * (c4 + x2 * c6))) * invRc;                \
    const real difference = qb1 * dq2 + qb2 * dq1 + dq1 * dq2;                       \
    const real pairEnergy = one4PiEps0 * (difference * invR + q1 * q2 * poly         \
                                          + qb1 * qb2 * invRc);                      \
    /* dP/dr divided by r; the radial normalisation cancels analytically. */         \
    const real dPolyOverR = (2 * c2 + x2 * (4 * c4 + 6 * c6 * x2)) * invRc3;         \
    const real scale = one4PiEps0 * (-difference * invR * invR * invR                \
                                     + q1 * q2 * dPolyOverR);                        \
    const real closure = invR + poly;

/**
 * Flush one contiguous-block tile: one atomic per atom per component, instead
 * of six per pair. This is the whole point of the warp-tile rewrite.
 *
 * The y side is only written for off-diagonal tiles; on the diagonal every pair
 * was counted twice into atom1, so there is nothing separate to flush.
 */
#define LOCALCWLD_STORE_TILE(x, y) {                                                 \
    const int store1 = (x) * TILE_SIZE + tgx;                                        \
    if (store1 < NUM_ATOMS) {                                                        \
        ATOMIC_ADD(&forceBuffers[store1], (mm_ulong) realToFixedPoint(fx1));         \
        ATOMIC_ADD(&forceBuffers[store1 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(fy1)); \
        ATOMIC_ADD(&forceBuffers[store1 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(fz1)); \
        ATOMIC_ADD(&adjointBuffer[store1], (mm_ulong) realToFixedPoint(adj1));       \
    }                                                                                \
    if ((x) != (y)) {                                                                \
        const int store2 = (y) * TILE_SIZE + tgx;                                    \
        if (store2 < NUM_ATOMS) {                                                    \
            ATOMIC_ADD(&forceBuffers[store2], (mm_ulong) realToFixedPoint(localForce[LOCAL_ID].x)); \
            ATOMIC_ADD(&forceBuffers[store2 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localForce[LOCAL_ID].y)); \
            ATOMIC_ADD(&forceBuffers[store2 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localForce[LOCAL_ID].z)); \
            ATOMIC_ADD(&adjointBuffer[store2], (mm_ulong) realToFixedPoint(localAdjoint[LOCAL_ID])); \
        }                                                                            \
    }                                                                                \
    SYNC_WARPS;                                                                      \
}

/* Scaling by an exact power of two, in `real`. The first version divided by
 * 0x100000000 in DOUBLE, which is an fp64 op in device code -- on the consumer
 * card this is measured on, fp64 runs at 1/32 of fp32, and DEC-004 says the GPU
 * path is fp32 end to end. No precision is lost: 2^-32 is exact, so this is a
 * single conversion plus an exact scale, where the old form rounded twice. */
#define fixedPointToReal(x) ((real) (x) * ((real) 0x1p-32))

/**
 * Traversal notes for the two kernels below (source: OpenMM commit 36a30cb,
 * platforms/common/src/kernels/customGBValueN2.cc -- the layout is not in the
 * installed headers).
 *
 *   exclusionTiles[pos]                 int2(x, y) block pair containing exclusions
 *   exclusions[pos*TILE_SIZE + tgx]     32-bit mask for atom x*32+tgx against
 *                                       the 32 atoms of block y; bit j SET means
 *                                       NOT excluded
 *   tiles[pos]                          x-block of interaction pos
 *   interactingAtoms[pos*TILE_SIZE + j] the y-side atoms of interaction pos
 *   interactionCount[0]                 number of interactions; > maxTiles means
 *                                       the list overflowed and must be skipped
 *
 * Two loops, not one. Tiles containing exclusions are visited separately with
 * the bitmask; the neighbour-list loop then covers the rest. Crucially, under
 * USE_CUTOFF the two are already disjoint -- OpenMM's list builder omits
 * exclusion tiles, so no skip logic is needed here (the `skipTiles` dance in
 * OpenMM's kernel is only in its no-cutoff branch). Missing that first loop
 * would drop EVERY pair in those tiles, not just the excluded ones, and the
 * symptom would be a small force error that looks like precision noise --
 * which is why DEC-005 section 3.7 checks residuals for tile clustering.
 *
 * These are written the simple way: one thread per (tile, lane), each thread
 * looping serially over the 32 partner atoms. No warp shuffle, no LOCAL
 * buffers. Correctness first; DEC-005 must pass before any optimisation
 * (that is LCWLD-150).
 *
 * Diagonal exclusion tiles (x == y) enumerate the block against itself, so they
 * take the upper triangle only. Off-diagonal and neighbour-list tiles pair two
 * distinct blocks, so all 32x32 combinations are distinct.
 */

/** True when this pair can contribute density in at least one direction. */
inline DEVICE bool localCWLDPairMatters(real sink1, real src1, real sink2, real src2) {
    return !((sink1 == 0 || src2 == 0) && (sink2 == 0 || src1 == 0));
}

/** Append one pair to the compact list, flagging overflow instead of writing past the end. */
inline DEVICE void localCWLDEmitPair(int i, int j, GLOBAL int2* envPairs,
                                     GLOBAL int* envPairCount, GLOBAL int* status,
                                     int maxEnvPairs) {
    const int slot = ATOMIC_ADD(envPairCount, 1);
    if (slot < maxEnvPairs)
        envPairs[slot] = make_int2(i, j);
    else
        status[ENV_PAIRS_OVERFLOW_FLAG] = 1;
}

/**
 * Build the compact r_env pair list. Run only when OpenMM rebuilds its own
 * neighbour list -- that amortisation is what turns a 1.36x kernel into a
 * 2.36x one (see the header comment).
 *
 * The padded radius means pairs that drift inside r_env between rebuilds are
 * already in the list; every consumer re-tests the exact r_env, so the padding
 * cannot change the physics.
 */
KERNEL void buildEnvPairs(
        GLOBAL const real4* RESTRICT posq,
        GLOBAL const real* RESTRICT source,        // densSource * srcWeight * chargeMod
        GLOBAL const real* RESTRICT densSink,
        GLOBAL const int* RESTRICT residueId,
        GLOBAL const unsigned int* RESTRICT exclusions,
        GLOBAL const int2* RESTRICT exclusionTiles,
        GLOBAL const int* RESTRICT tiles,
        GLOBAL const unsigned int* RESTRICT interactionCount,
        GLOBAL const int* RESTRICT interactingAtoms,
        GLOBAL int2* RESTRICT envPairs,
        GLOBAL int* RESTRICT envPairCount,
        GLOBAL int* RESTRICT status,
        int maxEnvPairs,
        unsigned int maxTiles,
        real paddedEnvCutoffSq,
        real4 periodicBoxSize,
        real4 invPeriodicBoxSize,
        real4 periodicBoxVecX,
        real4 periodicBoxVecY,
        real4 periodicBoxVecZ,
        GLOBAL const int* RESTRICT rebuildFlag,
        int forceRebuild) {

    // Read the platform's rebuild flag ON THE DEVICE. OpenMM's own kernels read
    // it the same way and never sync for it. The host used to download these
    // four bytes every force evaluation; that blocking read drains the queued
    // pipeline and was measured at 0.297 rc-traversal-equivalents per step --
    // more than three times the amortised cost of the rebuild it was gating.
    // Launching this kernel unconditionally and returning immediately costs a
    // few microseconds instead.
    //
    // `forceRebuild` is the host's override. Two cases need it and neither can
    // be expressed through the platform's flag: the very first evaluation
    // (where nothing has built the list yet) and the evaluation after the
    // buffer was grown (where the list must be refilled at the new capacity).
    // The old host-side path got the first case from `envRebuildCounter < 0`;
    // dropping that check when the decision moved to the device would have left
    // an empty list on any step where the platform happened not to rebuild.
    // forceRebuild: 0 follow the platform, +1 rebuild now, -1 never rebuild.
    // The -1 state is the LCWLD-150 timing probe. It has to live here rather
    // than on the host, because the host no longer makes this decision --
    // when the check moved onto the device the probes went with it, and for a
    // while they were silently no-ops that made a mutation test pass by
    // testing nothing.
    if (forceRebuild < 0)
        return;
    if (rebuildFlag[0] == 0 && forceRebuild == 0)
        return;

    // ---- Loop 1: tiles containing exclusions -----------------------------
    for (int idx = GLOBAL_ID; idx < NUM_TILES_WITH_EXCLUSIONS * TILE_SIZE; idx += GLOBAL_SIZE) {
        const int pos = idx / TILE_SIZE;
        const int tgx = idx - pos * TILE_SIZE;
        const int2 tile = exclusionTiles[pos];
        const int atom1 = tile.x * TILE_SIZE + tgx;
        if (atom1 >= NUM_ATOMS)
            continue;
        const unsigned int excl = exclusions[pos * TILE_SIZE + tgx];
        const bool diagonal = (tile.x == tile.y);
        const real4 pos1 = posq[atom1];
        const int res1 = residueId[atom1];
        const real sink1 = densSink[atom1], src1 = source[atom1];

        for (int j = 0; j < TILE_SIZE; j++) {
            const int atom2 = tile.y * TILE_SIZE + j;
            if (atom2 >= NUM_ATOMS)
                continue;
            if (diagonal ? (atom2 <= atom1) : false)
                continue;                       // upper triangle on the diagonal
            if (((excl >> j) & 0x1) == 0)
                continue;                       // excluded pair: no density
            if (residueId[atom2] == res1)
                continue;                       // same residue: density is zero
            if (!localCWLDPairMatters(sink1, src1, densSink[atom2], source[atom2]))
                continue;
            real4 delta = make_real4(posq[atom2].x - pos1.x, posq[atom2].y - pos1.y,
                                     posq[atom2].z - pos1.z, 0);
            APPLY_PERIODIC_TO_DELTA(delta)
            if (delta.x * delta.x + delta.y * delta.y + delta.z * delta.z >= paddedEnvCutoffSq)
                continue;
            const int lo = atom1 < atom2 ? atom1 : atom2;
            const int hi = atom1 < atom2 ? atom2 : atom1;
            localCWLDEmitPair(lo, hi, envPairs, envPairCount, status, maxEnvPairs);
        }
    }

    // ---- Loop 2: neighbour-list tiles (disjoint from loop 1) -------------
    const unsigned int numTiles = interactionCount[0];
    if (numTiles > maxTiles)
        return;                                 // list overflowed; OpenMM rebuilds
    for (int idx = GLOBAL_ID; idx < (int) (numTiles * TILE_SIZE); idx += GLOBAL_SIZE) {
        const int pos = idx / TILE_SIZE;
        const int tgx = idx - pos * TILE_SIZE;
        const int atom1 = tiles[pos] * TILE_SIZE + tgx;
        if (atom1 >= NUM_ATOMS)
            continue;
        const real4 pos1 = posq[atom1];
        const int res1 = residueId[atom1];
        const real sink1 = densSink[atom1], src1 = source[atom1];

        for (int j = 0; j < TILE_SIZE; j++) {
            const int atom2 = interactingAtoms[pos * TILE_SIZE + j];
            if (atom2 >= NUM_ATOMS || atom2 == atom1)
                continue;
            if (residueId[atom2] == res1)
                continue;
            if (!localCWLDPairMatters(sink1, src1, densSink[atom2], source[atom2]))
                continue;
            real4 delta = make_real4(posq[atom2].x - pos1.x, posq[atom2].y - pos1.y,
                                     posq[atom2].z - pos1.z, 0);
            APPLY_PERIODIC_TO_DELTA(delta)
            if (delta.x * delta.x + delta.y * delta.y + delta.z * delta.z >= paddedEnvCutoffSq)
                continue;
            const int lo = atom1 < atom2 ? atom1 : atom2;
            const int hi = atom1 < atom2 ? atom2 : atom1;
            localCWLDEmitPair(lo, hi, envPairs, envPairCount, status, maxEnvPairs);
        }
    }
}

/**
 * Pass A: local density.
 *
 *   dens_i = densSink_i * SUM_j source_j * (1 - r^2/r_env^2)^2 ,  r < r_env
 *
 * Both directions of every stored pair are accumulated, since the list holds
 * i<j only. The exact r_env test lives here, so the padding used when the list
 * was built cannot leak into the physics.
 */
KERNEL void computeDensity(
        GLOBAL const real4* RESTRICT posq,
        GLOBAL const real* RESTRICT source,
        GLOBAL const real* RESTRICT densSink,
        GLOBAL const int2* RESTRICT envPairs,
        GLOBAL const int* RESTRICT envPairCount,
        GLOBAL mm_ulong* RESTRICT densBuffer,      // fixed point
        real invEnvCutoffSq,
        real envCutoffSq,
        real4 periodicBoxSize,
        real4 invPeriodicBoxSize,
        real4 periodicBoxVecX,
        real4 periodicBoxVecY,
        real4 periodicBoxVecZ) {
    const int count = envPairCount[0];
    for (int p = GLOBAL_ID; p < count; p += GLOBAL_SIZE) {
        const int2 pair = envPairs[p];
        const int i = pair.x, j = pair.y;
        real4 delta = make_real4(posq[j].x - posq[i].x, posq[j].y - posq[i].y,
                                 posq[j].z - posq[i].z, 0);
        APPLY_PERIODIC_TO_DELTA(delta)
        const real r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        if (r2 >= envCutoffSq)
            continue;                            // padded list, exact test here
        const real t = 1 - r2 * invEnvCutoffSq;
        const real w = t * t;
        // i receives from j, and j from i.
        const real toI = densSink[i] * source[j] * w;
        const real toJ = densSink[j] * source[i] * w;
        if (toI != 0)
            ATOMIC_ADD(&densBuffer[i], (mm_ulong) realToFixedPoint(toI));
        if (toJ != 0)
            ATOMIC_ADD(&densBuffer[j], (mm_ulong) realToFixedPoint(toJ));
    }
}

/**
 * Per-particle: density -> response charge. No neighbour traversal.
 *
 *   inner  = tanh(k_polar * dens / rho0)
 *   raw    = amplitude * inner                    amplitude = phase*isPolar*dpolar
 *   dQ     = clamp * tanh(raw / clamp)
 *   Q      = qbase + dQ
 *   dQ/ddens = amplitude * (k/rho0) * (1-outer^2) * (1-inner^2)
 *
 * The nested tanh is the soft clamp. At the measured operating point
 * |raw|/clamp is around 0.01, so the outer tanh is linear to ~1e-4 -- but that
 * is a property of this system, not of the formula, so it is computed rather
 * than assumed (DEC-005 discussion).
 */
KERNEL void computeQ(
        GLOBAL const real* RESTRICT qbase,
        GLOBAL const real* RESTRICT amplitude,     // staticPhase * isPolar * dpolar
        GLOBAL const mm_long* RESTRICT densBuffer,
        GLOBAL real* RESTRICT dens,
        GLOBAL real* RESTRICT charge,
        GLOBAL real* RESTRICT dQdDens,
        real kOverRho0,
        real chargeDeltaClamp,
        real invChargeDeltaClamp) {
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        const real d = fixedPointToReal(densBuffer[i]);
        dens[i] = d;
        const real amp = amplitude[i];
        if (amp == 0) {
            charge[i] = qbase[i];
            dQdDens[i] = 0;
            continue;
        }
        const real inner = tanh(kOverRho0 * d);
        const real outer = tanh(amp * inner * invChargeDeltaClamp);
        const real deltaQ = chargeDeltaClamp * outer;
        charge[i] = qbase[i] + deltaQ;
        dQdDens[i] = amp * kOverRho0 * (1 - outer * outer) * (1 - inner * inner);
    }
}

/**
 * Pass B: pair energy, direct force, and the adjoint. The only pass that sees
 * the whole rc neighbour list, and therefore the only one whose per-pair cost
 * matters.
 *
 *   closure(r)  = 1/r + P(r),  P(r) = (c0 + x2(c2 + x2(c4 + x2 c6)))/rc,  x2 = r^2/rc^2
 *   difference  = qbase_i*dQ_j + qbase_j*dQ_i + dQ_i*dQ_j
 *   U          += k*[ difference/r + q_i q_j P(r) + qbase_i qbase_j / rc ]
 *   F          += -dU/dr
 *   lam_i      += k * q_j * closure(r)        (consumed by pass C)
 *
 * ---------------------------------------------------------------------------
 * Why this is warp-tiled rather than one thread per pair
 * ---------------------------------------------------------------------------
 * The first version was written "correctness first": one thread per (tile,
 * lane), a serial loop over the 32 partners, and six ATOMIC_ADDs per pair
 * (three components on each of the two atoms). It was correct -- it passed the
 * DEC-005 per-particle acceptance on real 1AAY -- and it was **1.42x SLOWER
 * than the CustomGBForce it replaces**, making the whole plugin a 0.736x
 * regression end to end.
 *
 * The measurement that said so also said where it went: the compact r_env list
 * is 304k pairs against 11.3M rc pairs (2.7%), so the design premise -- density
 * and chain are nearly free -- held. What failed was the constant. At 11.3M
 * pairs, six atomics each is 68M atomic operations per step.
 *
 * So each warp now owns one tile: the 32 y-side atoms live in LOCAL memory,
 * each lane keeps its x-side atom's force and adjoint in registers, and the
 * partner index rotates so no two lanes touch the same LOCAL slot in the same
 * iteration. One atomic per atom per tile at the end:
 *
 *     per tile   before: 1024 pairs x 6 = 6144 atomics
 *                after:  32 x-atoms x 4 + 32 y-atoms x 4 = 256 atomics
 *
 * Exclusion mask handling is the fiddly part and is taken verbatim from
 * OpenMM's own customGBValueN2.cc (commit 36a30cb): the mask is rotated by tgx
 * up front so that bit j of the rotated word still refers to y-atom
 * (tgx + j) mod 32, then shifted one bit per iteration. Diagonal tiles do NOT
 * rotate, because there the partner index does not.
 *
 * Diagonal tiles see every pair twice (lane i as atom1=i, and lane j as
 * atom1=j), so they accumulate onto atom1 only and halve the energy.
 * Off-diagonal and neighbour-list tiles see each pair once and accumulate both
 * sides at full weight.
 */
KERNEL void computePairAndAdjoint(
        GLOBAL const real4* RESTRICT posq,
        GLOBAL const real* RESTRICT qbase,
        GLOBAL const real* RESTRICT charge,
        GLOBAL const unsigned int* RESTRICT exclusions,
        GLOBAL const int2* RESTRICT exclusionTiles,
        GLOBAL const int* RESTRICT tiles,
        GLOBAL const unsigned int* RESTRICT interactionCount,
        GLOBAL const int* RESTRICT interactingAtoms,
        GLOBAL mm_ulong* RESTRICT forceBuffers,
        GLOBAL mm_ulong* RESTRICT adjointBuffer,
        GLOBAL mixed* RESTRICT energyBuffer,
        unsigned int maxTiles,
        real cutoffSq,
        real invCutoff,
        real one4PiEps0,
        real c0, real c2, real c4, real c6,
        real4 periodicBoxSize,
        real4 invPeriodicBoxSize,
        real4 periodicBoxVecX,
        real4 periodicBoxVecY,
        real4 periodicBoxVecZ) {
    const unsigned int totalWarps = GLOBAL_SIZE / TILE_SIZE;
    const unsigned int warp = GLOBAL_ID / TILE_SIZE;
    const unsigned int tgx = LOCAL_ID & (TILE_SIZE - 1);
    const unsigned int tbx = LOCAL_ID - tgx;

    LOCAL real4 localPos[LOCAL_BUFFER_SIZE];
    LOCAL real localQ[LOCAL_BUFFER_SIZE];
    LOCAL real localQbase[LOCAL_BUFFER_SIZE];
    LOCAL real3 localForce[LOCAL_BUFFER_SIZE];
    LOCAL real localAdjoint[LOCAL_BUFFER_SIZE];
    /*
     * The y-side atom indices, cached. The neighbour-list loop used to re-read
     * interactingAtoms[pos*TILE_SIZE + tj] from GLOBAL memory on every one of
     * the 32 inner iterations, and tj rotates, so those reads are uncoalesced:
     * 1024 scattered global loads per tile where 32 coalesced ones suffice.
     * With ~62000 neighbour-list tiles a step that is ~6e7 extra scattered
     * loads per step, in the hottest loop in the plugin.
     *
     * OpenMM's own tiled kernels cache it exactly like this
     * (customGBEnergyN2.cc:183/243/270, commit 36a30cb). Costs 4 bytes per
     * thread of LOCAL memory.
     */
    LOCAL int atomIndices[LOCAL_BUFFER_SIZE];

    mixed energy = 0;
    const real invRc = invCutoff;
    const real invRc2 = invRc * invRc;
    const real invRc3 = invRc2 * invRc;

    // ---- Loop 1: tiles containing exclusions -----------------------------
    const int firstExclusionTile = (int) (warp * (mm_long) NUM_TILES_WITH_EXCLUSIONS / totalWarps);
    const int lastExclusionTile = (int) ((warp + 1) * (mm_long) NUM_TILES_WITH_EXCLUSIONS / totalWarps);
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const int2 tile = exclusionTiles[pos];
        const int x = tile.x, y = tile.y;
        const int atom1 = x * TILE_SIZE + tgx;
        const real4 pos1 = posq[atom1];
        const real qb1 = qbase[atom1], q1 = charge[atom1];
        const real dq1 = q1 - qb1;
        real fx1 = 0, fy1 = 0, fz1 = 0, adj1 = 0;
        unsigned int excl = exclusions[pos * TILE_SIZE + tgx];

        if (x == y) {
            // Diagonal: the block against itself. Every pair is visited twice,
            // so only atom1 accumulates and the energy is halved.
            localPos[LOCAL_ID] = pos1;
            localQ[LOCAL_ID] = q1;
            localQbase[LOCAL_ID] = qb1;
            SYNC_WARPS;
            for (int j = 0; j < TILE_SIZE; j++) {
                const int atom2 = y * TILE_SIZE + j;
                real4 delta = make_real4(localPos[tbx+j].x - pos1.x,
                                         localPos[tbx+j].y - pos1.y,
                                         localPos[tbx+j].z - pos1.z, 0);
                APPLY_PERIODIC_TO_DELTA(delta)
                const real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                const bool ok = (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS &&
                                 atom1 != atom2 && (excl & 0x1) && r2 < cutoffSq);
                if (ok) {
                    const real qb2 = localQbase[tbx+j], q2 = localQ[tbx+j];
                    LOCALCWLD_PAIR_CORE
                    energy += (mixed) (0.5f * pairEnergy);
                    fx1 += scale * delta.x;
                    fy1 += scale * delta.y;
                    fz1 += scale * delta.z;
                    adj1 += one4PiEps0 * q2 * closure;
                }
                excl >>= 1;
                SYNC_WARPS;
            }
        }
        else {
            // Off-diagonal: two distinct blocks, every pair once.
            const int atom2Base = y * TILE_SIZE + tgx;
            localPos[LOCAL_ID] = posq[atom2Base];
            localQ[LOCAL_ID] = charge[atom2Base];
            localQbase[LOCAL_ID] = qbase[atom2Base];
            localForce[LOCAL_ID] = make_real3(0, 0, 0);
            localAdjoint[LOCAL_ID] = 0;
            // Rotate so bit j of `excl` still names y-atom (tgx+j) mod 32.
            excl = (excl >> tgx) | (excl << (TILE_SIZE - tgx));
            SYNC_WARPS;
            unsigned int tj = tgx;
            for (int j = 0; j < TILE_SIZE; j++) {
                const int atom2 = y * TILE_SIZE + tj;
                real4 delta = make_real4(localPos[tbx+tj].x - pos1.x,
                                         localPos[tbx+tj].y - pos1.y,
                                         localPos[tbx+tj].z - pos1.z, 0);
                APPLY_PERIODIC_TO_DELTA(delta)
                const real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                const bool ok = (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS &&
                                 (excl & 0x1) && r2 < cutoffSq);
                if (ok) {
                    const real qb2 = localQbase[tbx+tj], q2 = localQ[tbx+tj];
                    LOCALCWLD_PAIR_CORE
                    energy += (mixed) pairEnergy;
                    fx1 += scale * delta.x;
                    fy1 += scale * delta.y;
                    fz1 += scale * delta.z;
                    adj1 += one4PiEps0 * q2 * closure;
                    localForce[tbx+tj].x -= scale * delta.x;
                    localForce[tbx+tj].y -= scale * delta.y;
                    localForce[tbx+tj].z -= scale * delta.z;
                    localAdjoint[tbx+tj] += one4PiEps0 * q1 * closure;
                }
                excl >>= 1;
                tj = (tj + 1) & (TILE_SIZE - 1);
                SYNC_WARPS;
            }
        }
        LOCALCWLD_STORE_TILE(x, y)
    }

    // ---- Loop 2: neighbour-list tiles (disjoint from loop 1) -------------
    const unsigned int numTiles = interactionCount[0];
    if (numTiles <= maxTiles) {
        const int first = (int) (warp * (mm_long) numTiles / totalWarps);
        const int last = (int) ((warp + 1) * (mm_long) numTiles / totalWarps);
        for (int pos = first; pos < last; pos++) {
            const int x = tiles[pos];
            const int atom1 = x * TILE_SIZE + tgx;
            const real4 pos1 = posq[atom1];
            const real qb1 = qbase[atom1], q1 = charge[atom1];
            const real dq1 = q1 - qb1;
            real fx1 = 0, fy1 = 0, fz1 = 0, adj1 = 0;

            const int atom2Base = interactingAtoms[pos * TILE_SIZE + tgx];
            localPos[LOCAL_ID] = posq[atom2Base];
            localQ[LOCAL_ID] = charge[atom2Base];
            localQbase[LOCAL_ID] = qbase[atom2Base];
            localForce[LOCAL_ID] = make_real3(0, 0, 0);
            localAdjoint[LOCAL_ID] = 0;
            atomIndices[LOCAL_ID] = atom2Base;
            SYNC_WARPS;
            unsigned int tj = tgx;
            for (int j = 0; j < TILE_SIZE; j++) {
                const int atom2 = atomIndices[tbx + tj];
                real4 delta = make_real4(localPos[tbx+tj].x - pos1.x,
                                         localPos[tbx+tj].y - pos1.y,
                                         localPos[tbx+tj].z - pos1.z, 0);
                APPLY_PERIODIC_TO_DELTA(delta)
                const real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                const bool ok = (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS &&
                                 atom1 != atom2 && r2 < cutoffSq);
                if (ok) {
                    const real qb2 = localQbase[tbx+tj], q2 = localQ[tbx+tj];
                    LOCALCWLD_PAIR_CORE
                    energy += (mixed) pairEnergy;
                    fx1 += scale * delta.x;
                    fy1 += scale * delta.y;
                    fz1 += scale * delta.z;
                    adj1 += one4PiEps0 * q2 * closure;
                    localForce[tbx+tj].x -= scale * delta.x;
                    localForce[tbx+tj].y -= scale * delta.y;
                    localForce[tbx+tj].z -= scale * delta.z;
                    localAdjoint[tbx+tj] += one4PiEps0 * q1 * closure;
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
                SYNC_WARPS;
            }
            // The y side is a scattered atom list, not a contiguous block, so
            // the store cannot use the x/y helper.
            ATOMIC_ADD(&forceBuffers[atom1], (mm_ulong) realToFixedPoint(fx1));
            ATOMIC_ADD(&forceBuffers[atom1 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(fy1));
            ATOMIC_ADD(&forceBuffers[atom1 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(fz1));
            ATOMIC_ADD(&adjointBuffer[atom1], (mm_ulong) realToFixedPoint(adj1));
            if (atom2Base < NUM_ATOMS) {
                ATOMIC_ADD(&forceBuffers[atom2Base], (mm_ulong) realToFixedPoint(localForce[LOCAL_ID].x));
                ATOMIC_ADD(&forceBuffers[atom2Base + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localForce[LOCAL_ID].y));
                ATOMIC_ADD(&forceBuffers[atom2Base + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localForce[LOCAL_ID].z));
                ATOMIC_ADD(&adjointBuffer[atom2Base], (mm_ulong) realToFixedPoint(localAdjoint[LOCAL_ID]));
            }
            SYNC_WARPS;
        }
    }
    energyBuffer[GLOBAL_ID] += energy;
}

/**
 * Pass C: chain force, over the compact r_env list.
 *
 *   response_i = lam_i * (dQ_i/ddens_i) * densSink_i
 *   F_pair = -4/r_env^2 * (1 - r^2/r_env^2)
 *            * (response_i*source_j + response_j*source_i) * delta
 *
 * Same list, same exact-r_env test as pass A.
 */
KERNEL void computeChainForce(
        GLOBAL const real4* RESTRICT posq,
        GLOBAL const real* RESTRICT source,
        GLOBAL const real* RESTRICT densSink,
        GLOBAL const real* RESTRICT dQdDens,
        GLOBAL const mm_long* RESTRICT adjointBuffer,
        GLOBAL const int2* RESTRICT envPairs,
        GLOBAL const int* RESTRICT envPairCount,
        GLOBAL mm_ulong* RESTRICT forceBuffers,
        real invEnvCutoffSq,
        real envCutoffSq,
        real4 periodicBoxSize,
        real4 invPeriodicBoxSize,
        real4 periodicBoxVecX,
        real4 periodicBoxVecY,
        real4 periodicBoxVecZ) {
    const int count = envPairCount[0];
    for (int p = GLOBAL_ID; p < count; p += GLOBAL_SIZE) {
        const int2 pair = envPairs[p];
        const int i = pair.x, j = pair.y;
        real4 delta = make_real4(posq[j].x - posq[i].x, posq[j].y - posq[i].y,
                                 posq[j].z - posq[i].z, 0);
        APPLY_PERIODIC_TO_DELTA(delta)
        const real r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        if (r2 >= envCutoffSq)
            continue;
        const real t = 1 - r2 * invEnvCutoffSq;
        const real respI = fixedPointToReal(adjointBuffer[i]) * dQdDens[i] * densSink[i];
        const real respJ = fixedPointToReal(adjointBuffer[j]) * dQdDens[j] * densSink[j];
        const real scale = -4.0f * invEnvCutoffSq * t
                           * (respI * source[j] + respJ * source[i]);
        if (scale == 0)
            continue;
        const real fx = scale * delta.x;
        const real fy = scale * delta.y;
        const real fz = scale * delta.z;
        // Same sign convention as LOCALCWLD_PAIR_TERM: i takes +f, j takes -f.
        // Newton's third law holds by construction either way, which is exactly
        // why it could not catch the swapped version.
        ATOMIC_ADD(&forceBuffers[i], (mm_ulong) realToFixedPoint(fx));
        ATOMIC_ADD(&forceBuffers[i + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(fy));
        ATOMIC_ADD(&forceBuffers[i + 2 * PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(fz));
        ATOMIC_ADD(&forceBuffers[j], (mm_ulong) realToFixedPoint(-fx));
        ATOMIC_ADD(&forceBuffers[j + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-fy));
        ATOMIC_ADD(&forceBuffers[j + 2 * PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-fz));
    }
}

/**
 * Zero the fixed-point scratch buffers between steps, and the compact-list
 * counter on the steps that are about to rebuild it.
 *
 * The counter has to be zeroed here rather than by the host, because the host
 * no longer knows whether a rebuild is happening: reading `rebuildFlag` on the
 * host cost a blocking device->host transfer every step, measured at 16.4% of
 * this force's entire cost (design doc section 9). The decision moved onto the
 * device, so the zeroing had to follow it.
 *
 * `status` is deliberately NOT cleared here. It is sticky: once the build
 * overflows, the flag stays raised until the host reads and clears it. Clearing
 * it every step would let an overflow at rebuild k be erased by rebuild k+1
 * before the host's next (infrequent) check ever saw it.
 */
KERNEL void clearBuffers(
        GLOBAL mm_ulong* RESTRICT densBuffer,
        GLOBAL mm_ulong* RESTRICT adjointBuffer,
        GLOBAL const int* RESTRICT rebuildFlag,
        GLOBAL int* RESTRICT envPairCount,
        int forceRebuild) {
    for (int i = GLOBAL_ID; i < PADDED_NUM_ATOMS; i += GLOBAL_SIZE) {
        densBuffer[i] = 0;
        adjointBuffer[i] = 0;
    }
    if (GLOBAL_ID == 0 && forceRebuild >= 0 &&
        (rebuildFlag[0] != 0 || forceRebuild > 0))
        envPairCount[0] = 0;
}
