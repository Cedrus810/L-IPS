#include "CommonLocalCWLDKernels.h"
#include <functional>
#include <map>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "LocalCWLDForceInfo.h"
#include "LocalCWLDKernelSources.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ComputeParameterInfo.h"
#include "openmm/common/CommonKernelUtilities.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/NonbondedUtilities.h"
#include "openmm/internal/ContextImpl.h"

using namespace LocalCWLDPlugin;
using OpenMM::ComputeContext;
using OpenMM::ContextImpl;
using OpenMM::OpenMMException;

namespace {

/** Evaluations during which every build is checked and the buffer may grow. */
const int CAPACITY_DISCOVERY_EVALS = 8;
/** After discovery, one evaluation in this many pays the blocking check. */
const int CAPACITY_CHECK_INTERVAL = 100;
/**
 * Capacity as a multiple of the observed requirement. 1.3 was enough while the
 * list was checked after every rebuild; with the check now spaced out, the
 * margin has to cover ordinary fluctuation on its own, because firing is a
 * hard error rather than a growth trigger.
 */
const double CAPACITY_HEADROOM = 1.6;



/**
 * ZMM closure polynomial coefficients, frozen by plan section 18.3.
 *
 * A property worth knowing before touching these: for every order the
 * coefficients sum to -1, which is exactly what makes the pair *energy* vanish
 * at rc (closure(rc) = 1/rc - 1/rc = 0). The *force* does not vanish there --
 * that residual is what makes a cutoff-membership flip worth ~29 kJ/mol/nm in
 * fp32, and why DEC-005 section 4 gives boundary pairs their own rule.
 */
void zmmCoefficients(int order, double& c0, double& c2, double& c4, double& c6) {
    switch (order) {
        case 1: c0 = -1.5;      c2 = 0.5;       c4 = 0.0;        c6 = 0.0;       break;
        case 2: c0 = -15.0 / 8; c2 = 5.0 / 4;   c4 = -3.0 / 8;   c6 = 0.0;       break;
        case 3: c0 = -35.0 / 16; c2 = 35.0 / 16; c4 = -21.0 / 16; c6 = 5.0 / 16; break;
        default: {
            std::ostringstream message;
            message << "LocalCWLDForce: zmm_order must be 1, 2 or 3, got " << order;
            throw OpenMMException(message.str());
        }
    }
}

} // namespace

/**
 * True when every residue's atoms are connected to one another by exclusions.
 *
 * Exclusions are 1-2/1-3 pairs, so their transitive closure is the bonded
 * connectivity, which is what OpenMM builds molecules out of. If a residue sits
 * entirely inside one component then OpenMM -- which only ever permutes whole
 * molecules -- cannot move part of a residue without the rest, and the residue
 * PARTITION that the density term depends on survives reordering even though
 * the residueId labels stay with the slots. That is the precondition for
 * leaving residueId out of LocalCWLDForceInfo::areParticlesIdentical, which is
 * worth 3.07 rc traversals (LCWLD-161).
 */
static bool residuesFitInMolecules(const LocalCWLDForce& force) {
    std::vector<int> parent(force.getNumParticles());
    for (int i = 0; i < (int) parent.size(); i++)
        parent[i] = i;
    std::function<int(int)> find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int p1, p2;
        force.getExclusionParticles(i, p1, p2);
        const int a = find(p1), b = find(p2);
        if (a != b) parent[a] = b;
    }
    std::map<int, int> component;          // residueId -> the component it lives in
    for (int i = 0; i < force.getNumParticles(); i++) {
        double q, m, d, pol, src, snk, w, ph;
        int res;
        force.getParticleParameters(i, q, m, d, pol, src, snk, w, ph, res);
        if (res < 0)
            continue;                      // unassigned: matches nothing
        auto it = component.find(res);
        if (it == component.end())
            component[res] = find(i);
        else if (it->second != find(i))
            return false;
    }
    return true;
}

void CommonCalcLocalCWLDForceKernel::initialize(const OpenMM::System& system,
                                                const LocalCWLDForce& force) {
    OpenMM::ContextSelector selector(cc);

    envCutoff = force.getEnvironmentCutoff();
    cutoff = force.getCutoffDistance();
    rho0 = force.getRho0();
    kPolar = force.getKPolar();
    chargeDeltaClamp = force.getChargeDeltaClamp();
    one4PiEps0 = force.getOne4PiEps0();
    zmmOrder = force.getZMMOrder();
    useQPenalty = force.getUseQPenalty();
    qPenaltyStrength = force.getQPenaltyStrength();
    zmmCoefficients(zmmOrder, c0, c2, c4, c6);

    const int numAtoms = cc.getPaddedNumAtoms();
    qbase.initialize<float>(cc, numAtoms, "localCWLDQbase");
    source.initialize<float>(cc, numAtoms, "localCWLDSource");
    amplitude.initialize<float>(cc, numAtoms, "localCWLDAmplitude");
    densSink.initialize<float>(cc, numAtoms, "localCWLDDensSink");
    dQdDens.initialize<float>(cc, numAtoms, "localCWLDdQdDens");
    charge.initialize<float>(cc, numAtoms, "localCWLDCharge");
    dens.initialize<float>(cc, numAtoms, "localCWLDDens");
    residueId.initialize<int>(cc, numAtoms, "localCWLDResidueId");
    densBuffer.initialize<long long>(cc, numAtoms, "localCWLDDensBuffer");
    adjointBuffer.initialize<long long>(cc, numAtoms, "localCWLDAdjointBuffer");
    status.initialize<int>(cc, 1, "localCWLDStatus");
    envPairCount.initialize<int>(cc, 1, "localCWLDEnvPairCount");
    // initialize() allocates without zeroing. That was harmless while
    // clearBuffers reset `status` every step; now that the flag is sticky --
    // deliberately, so a rare overflow cannot be erased before the host's
    // infrequent check sees it -- nobody else ever writes it, and the first
    // check would read whatever the allocator handed back and throw at random.
    const int zeroInit = 0;
    status.upload(&zeroInit);
    envPairCount.upload(&zeroInit);

    // First guess at the compact r_env list size. Scaled from the volume ratio
    // rather than a round number, and grown on overflow -- the device kernel
    // sets a status flag instead of writing out of bounds.
    const double ratio = (envCutoff / cutoff) * (envCutoff / cutoff) * (envCutoff / cutoff);
    maxEnvPairs = std::max(1024, (int) (numAtoms * 64 * ratio * 4));
    envPairs.initialize(cc, maxEnvPairs, 2 * sizeof(int), "localCWLDEnvPairs");

    uploadParameters(force);

    // LCWLD-161. Can residueId be left out of areParticlesIdentical? Only if
    // OpenMM can never move part of a residue without the rest. OpenMM permutes
    // whole molecules, and exclusions are 1-2/1-3, so their transitive closure
    // is the bonded connectivity: if every residue lies inside one such
    // component, a permutation carries each residue's atoms together and the
    // residue PARTITION survives even though the labels stay with the slots.
    // (The kernels only ever test residueId for equality, never read its value
    // -- localCWLD.cc:264,297.)
    //
    // When it does NOT hold -- a residue made of atoms with no exclusions
    // between them, as two of the synthetic test systems have -- the answer is
    // to fall back to the strict comparison, NOT to refuse the system. Slow and
    // correct beats fast and silently wrong, and beats not running at all.
    const bool residuesAreMoleculeLocal = residuesFitInMolecules(force);
    if (std::getenv("LOCALCWLD_PROBE_NO_FORCEINFO") == nullptr)
        cc.addForce(new LocalCWLDForceInfo(force, residuesAreMoleculeLocal));
    else
        std::printf("\n*** PROBE E ACTIVE: no ComputeForceInfo registered.\n\n");

    // The pair pass needs the platform's rc neighbour list; asking for it here
    // is what makes the platform build one at our cutoff.
    std::vector<std::vector<int> > exclusionList(force.getNumParticles());
    for (int i = 0; i < force.getNumParticles(); i++)
        exclusionList[i].push_back(i);
    // (LCWLD-161 tried a self-exclusions-only probe here to test whether our
    // exclusion set was what made the shared list expensive. It cannot be run:
    // OpenMM throws "All Forces must have identical exceptions". That is a
    // stronger answer than a timing number -- our exclusions are required to
    // match the NonbondedForce's, so they cannot be the difference.)
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int p1, p2;
        force.getExclusionParticles(i, p1, p2);
        exclusionList[p1].push_back(p2);
        exclusionList[p2].push_back(p1);
    }
    cc.getNonbondedUtilities().addInteraction(true, true, true, cutoff, exclusionList,
                                              "", force.getForceGroup());

}

void CommonCalcLocalCWLDForceKernel::uploadParameters(const LocalCWLDForce& force) {
    const int numAtoms = cc.getPaddedNumAtoms();
    std::vector<float> qbaseHost(numAtoms, 0.0f), sourceHost(numAtoms, 0.0f),
                       amplitudeHost(numAtoms, 0.0f), sinkHost(numAtoms, 0.0f);
    std::vector<int> residueHost(numAtoms, -1);

    for (int i = 0; i < force.getNumParticles(); i++) {
        double q, chargeMod, dpolar, isPolar, dSource, dSink, weight, phase;
        int residue;
        force.getParticleParameters(i, q, chargeMod, dpolar, isPolar, dSource, dSink,
                                    weight, phase, residue);
        qbaseHost[i] = (float) q;
        // Two products the device never needs to recompute. They also make the
        // "can this pair contribute density at all" test in buildEnvPairs a
        // single comparison instead of three.
        sourceHost[i] = (float) (dSource * weight * chargeMod);
        amplitudeHost[i] = (float) (phase * isPolar * dpolar);
        sinkHost[i] = (float) dSink;
        residueHost[i] = residue;
    }
    // Padding atoms must not participate: residue -1 differs from every real
    // residue, and zero source/sink makes them inert in both directions.
    qbase.upload(qbaseHost);
    source.upload(sourceHost);
    amplitude.upload(amplitudeHost);
    densSink.upload(sinkHost);
    residueId.upload(residueHost);
}

/**
 * Verify the compact list did not overflow -- rarely, because it is blocking.
 *
 * Capacity is discovered in the first few evaluations, where every build is
 * checked and the buffer is grown to fit. After that the count is essentially
 * constant (fixed particle count, and the box moves slowly even under NPT), so
 * the check drops to one evaluation in CAPACITY_CHECK_INTERVAL and exists as a
 * backstop rather than as flow control.
 *
 * A backstop that fires means the run is already wrong: up to
 * CAPACITY_CHECK_INTERVAL evaluations used a truncated list. So it throws
 * rather than growing and continuing. Silently recovering would hide exactly
 * the class of error this project has spent the most time on -- a result that
 * looks fine and is not. The headroom below is set so that firing means a real
 * anomaly, not ordinary density fluctuation.
 */
void CommonCalcLocalCWLDForceKernel::checkEnvCapacity() {
    envEvalCount++;
    const bool debug = std::getenv("LOCALCWLD_DEBUG") != nullptr;
    const bool discovering = envEvalCount <= CAPACITY_DISCOVERY_EVALS;
    // LOCALCWLD_DEBUG restores the per-step blocking read, because the rebuild
    // count and interval cannot be recovered any other way. It therefore SLOWS
    // THE RUN DOWN -- do not benchmark with it set.
    if (!discovering && !debug && (envEvalCount % CAPACITY_CHECK_INTERVAL) != 0)
        return;

    std::vector<int> statusHost(1), countHost(1);

    // While discovering the capacity, grow and REBUILD WITHIN THIS EVALUATION.
    //
    // Deferring the refill to the next step was wrong and would have been
    // caught only by the acceptance harness: the very first evaluation is a
    // static getState() -- that is exactly what DEC-005 measures -- and the
    // initial capacity estimate from the r_env/rc volume ratio undershoots real
    // 1AAY by more than 2x. The first forces would have been computed from a
    // truncated list.
    for (int attempt = 0; attempt < 3; attempt++) {
        status.download(statusHost);
        envPairCount.download(countHost);
        if (statusHost[0] == 0)
            break;
        if (!discovering)
            throw OpenMMException(
                "LocalCWLDForce: the r_env pair list overflowed after warmup (needed " +
                std::to_string(countHost[0]) + ", capacity " +
                std::to_string(maxEnvPairs) + "). Up to " +
                std::to_string(CAPACITY_CHECK_INTERVAL) +
                " force evaluations were computed with a truncated list and are "
                "wrong. Growing and continuing would hide that.");
        if (attempt == 2)
            throw OpenMMException(
                "LocalCWLDForce: the r_env pair list kept overflowing after two "
                "grow attempts; something is wrong with the count, not the capacity.");
        maxEnvPairs = (int) (CAPACITY_HEADROOM * countHost[0]);
        envPairs.resize(maxEnvPairs);       // already initialized; resize, not re-init
        buildEnvKernel->setArg(9, envPairs);
        buildEnvKernel->setArg(12, maxEnvPairs);
        densityKernel->setArg(3, envPairs);
        chainKernel->setArg(5, envPairs);
        const int reset = 0;
        status.upload(&reset);
        envPairCount.upload(&reset);        // clearBuffers already ran this step
        buildEnvKernel->setArg(21, (int) 1);            // force the refill
        buildEnvKernel->execute(pairNumThreads, pairBlockSize);
    }

    if (debug) {
        // Read the platform flag as well, so the rebuild count is exact.
        // Inferring "a rebuild happened" from the list length changing would
        // miss two consecutive rebuilds that produced the same count, and the
        // rebuild INTERVAL is the number the whole cost model rests on.
        std::vector<int> flagHost(1);
        cc.getNonbondedUtilities().getRebuildNeighborList().download(flagHost);
        if (flagHost[0] != 0 || envEvalCount == 1) {
            envRebuildCount++;
            const int since = cc.getComputeForceCount() - lastRebuildForceCount;
            std::printf("[localcwld] env rebuild #%d: %d pairs (capacity %d), "
                        "%d force evaluations since the previous rebuild\n",
                        envRebuildCount, countHost[0], maxEnvPairs,
                        lastRebuildForceCount < 0 ? -1 : since);
            std::fflush(stdout);
            lastRebuildForceCount = cc.getComputeForceCount();
        }
    }
}


double CommonCalcLocalCWLDForceKernel::execute(ContextImpl& context, bool includeForces,
                                               bool includeEnergy) {
    // LCWLD-161 probe D: register with NonbondedUtilities, then launch nothing.
    //
    // Strictly one variable. initialize() -- and therefore the addInteraction()
    // that puts our cutoff and our exclusion list into the platform's shared
    // neighbour list -- is untouched; this removes only our six kernel
    // launches. LCWLD-161 measured that adding this force makes OpenMM's OWN
    // computeNonbonded go 225 -> 551 us/step and findBlocksWithInteractions
    // 48 -> 176, which is 1.51 rc traversals (32% of the force's cost) sitting
    // outside every kernel we wrote. Two candidates, and this separates them:
    //
    //   computeNonbonded stays ~551  -> the registration reshaped the shared
    //                                   list; look at what we pass to
    //                                   addInteraction.
    //   computeNonbonded drops ~225  -> our kernels running is what costs it
    //                                   (cache, occupancy, list state), not
    //                                   the registration.
    //
    // The physics is WRONG under this setting: the force contributes no energy
    // and no forces. Timing probe only, never an acceptance configuration.
    if (std::getenv("LOCALCWLD_PROBE_NO_KERNELS") != nullptr) {
        static bool announced = false;
        if (!announced) {
            announced = true;
            std::printf("\n*** PROBE D ACTIVE: LocalCWLDForce registers with the\n"
                        "*** neighbour list but launches no kernels.\n"
                        "*** The physics is wrong. This is a timing probe.\n"
                        "*** Do NOT quote the resulting f or ns/day as a result.\n\n");
            std::fflush(stdout);
        }
        return 0.0;
    }

    OpenMM::ContextSelector selector(cc);
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        std::map<std::string, std::string> defines;
        defines["NUM_ATOMS"] = cc.intToString(cc.getNumAtoms());
        defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
        defines["TILE_SIZE"] = "32";
        // The warp-tiled pair kernel keeps the y-side atoms in LOCAL memory,
        // so the buffer must be exactly one thread block wide and the launch
        // must use that same block size. Mismatch here is silent corruption:
        // tbx = LOCAL_ID - tgx would index past the buffer.
        //
        // Take the geometry from the platform rather than picking it. The first
        // version hardcoded 64 blocks x 128 threads = 8192 threads. On the
        // 68-SM device this was measured on, OpenMM runs the *same* tile
        // traversal with 4*68 = 272 blocks x 256 threads = 69632 -- 8.5x more,
        // and 64 blocks does not even cover 68 SMs. That launch is why the
        // first LCWLD-150 run came back at f = 1.076 instead of 0.419: a
        // latency-bound tile traversal at ~12% occupancy.
        //
        // Safe by construction: CudaContext sizes energyBuffer to at least
        // nonbonded->getNumEnergyBuffers() == numForceThreadBlocks *
        // forceThreadBlockSize (CudaContext.cpp:400, CudaNonbondedUtilities.h:124),
        // so `energyBuffer[GLOBAL_ID]` at this launch size is exactly what
        // OpenMM's own nonbonded kernel does.
        pairBlockSize = cc.getNonbondedUtilities().getForceThreadBlockSize();
        pairNumThreads = cc.getNonbondedUtilities().getNumForceThreadBlocks() * pairBlockSize;
        defines["LOCAL_BUFFER_SIZE"] = cc.intToString(pairBlockSize);
        if (std::getenv("LOCALCWLD_DEBUG") != nullptr) {
            std::printf("[localcwld] launch: %d threads in blocks of %d "
                        "(OpenMM's own nonbonded geometry)\n",
                        pairNumThreads, pairBlockSize);
            std::fflush(stdout);
        }
        defines["NUM_BLOCKS"] = cc.intToString(cc.getNumAtomBlocks());
        defines["NUM_TILES_WITH_EXCLUSIONS"] =
            cc.intToString(cc.getNonbondedUtilities().getExclusionTiles().getSize());
        OpenMM::ComputeProgram program =
            cc.compileProgram(LocalCWLDKernelSources::localCWLD, defines);
        clearKernel = program->createKernel("clearBuffers");
        buildEnvKernel = program->createKernel("buildEnvPairs");
        densityKernel = program->createKernel("computeDensity");
        qKernel = program->createKernel("computeQ");
        pairKernel = program->createKernel("computePairAndAdjoint");
        chainKernel = program->createKernel("computeChainForce");

        // Scalar arguments must match the platform's `real`, which is double
        // only when Precision=double. DEC-004 fixes the physics at fp32, but
        // the argument type still has to follow the platform or the kernel
        // reads garbage.
        const bool useDouble = cc.getUseDoublePrecision();
        auto addReal = [&](OpenMM::ComputeKernel& k, double v) {
            if (useDouble) k->addArg(v); else k->addArg((float) v);
        };

        OpenMM::NonbondedUtilities& nbu = cc.getNonbondedUtilities();
        const int nbMaxTiles = nbu.getInteractingTiles().getSize();
        const double envSq = envCutoff * envCutoff;
        // Padding so pairs drifting inside r_env between rebuilds are already
        // listed. Uses the platform's own list padding for consistency.
        const double paddedEnv = envCutoff + 0.1;

        clearKernel->addArg(densBuffer);
        clearKernel->addArg(adjointBuffer);
        clearKernel->addArg();              // rebuildFlag, bound per step
        clearKernel->addArg(envPairCount);
        clearKernel->addArg((int) 0);       // forceRebuild, set per step

        buildEnvKernel->addArg(cc.getPosq());
        buildEnvKernel->addArg(source);
        buildEnvKernel->addArg(densSink);
        buildEnvKernel->addArg(residueId);
        buildEnvKernel->addArg(nbu.getExclusions());
        buildEnvKernel->addArg(nbu.getExclusionTiles());
        buildEnvKernel->addArg(nbu.getInteractingTiles());
        buildEnvKernel->addArg(nbu.getInteractionCount());
        buildEnvKernel->addArg(nbu.getInteractingAtoms());
        buildEnvKernel->addArg(envPairs);
        buildEnvKernel->addArg(envPairCount);
        buildEnvKernel->addArg(status);
        buildEnvKernel->addArg(maxEnvPairs);
        buildEnvKernel->addArg((unsigned int) nbMaxTiles);
        addReal(buildEnvKernel, paddedEnv * paddedEnv);
        for (int i = 0; i < 5; i++) buildEnvKernel->addArg();   // box, set per call
        buildEnvKernel->addArg();                               // rebuildFlag, per step
        buildEnvKernel->addArg((int) 0);                        // forceRebuild, per step

        densityKernel->addArg(cc.getPosq());
        densityKernel->addArg(source);
        densityKernel->addArg(densSink);
        densityKernel->addArg(envPairs);
        densityKernel->addArg(envPairCount);
        densityKernel->addArg(densBuffer);
        addReal(densityKernel, 1.0 / envSq);
        addReal(densityKernel, envSq);
        for (int i = 0; i < 5; i++) densityKernel->addArg();

        qKernel->addArg(qbase);
        qKernel->addArg(amplitude);
        qKernel->addArg(densBuffer);
        qKernel->addArg(dens);
        qKernel->addArg(charge);
        qKernel->addArg(dQdDens);
        addReal(qKernel, kPolar / rho0);
        addReal(qKernel, chargeDeltaClamp);
        addReal(qKernel, 1.0 / chargeDeltaClamp);

        pairKernel->addArg(cc.getPosq());
        pairKernel->addArg(qbase);
        pairKernel->addArg(charge);
        pairKernel->addArg(nbu.getExclusions());
        pairKernel->addArg(nbu.getExclusionTiles());
        pairKernel->addArg(nbu.getInteractingTiles());
        pairKernel->addArg(nbu.getInteractionCount());
        pairKernel->addArg(nbu.getInteractingAtoms());
        pairKernel->addArg(cc.getLongForceBuffer());
        pairKernel->addArg(adjointBuffer);
        pairKernel->addArg(cc.getEnergyBuffer());
        pairKernel->addArg((unsigned int) nbMaxTiles);
        addReal(pairKernel, cutoff * cutoff);
        addReal(pairKernel, 1.0 / cutoff);
        addReal(pairKernel, one4PiEps0);
        addReal(pairKernel, c0);
        addReal(pairKernel, c2);
        addReal(pairKernel, c4);
        addReal(pairKernel, c6);
        for (int i = 0; i < 5; i++) pairKernel->addArg();

        chainKernel->addArg(cc.getPosq());
        chainKernel->addArg(source);
        chainKernel->addArg(densSink);
        chainKernel->addArg(dQdDens);
        chainKernel->addArg(adjointBuffer);
        chainKernel->addArg(envPairs);
        chainKernel->addArg(envPairCount);
        chainKernel->addArg(cc.getLongForceBuffer());
        addReal(chainKernel, 1.0 / envSq);
        addReal(chainKernel, envSq);
        for (int i = 0; i < 5; i++) chainKernel->addArg();
    }

    // ---- Re-bind the neighbour-list arguments EVERY step ------------------
    //
    // OpenMM grows its neighbour list when it overflows: computeInteractions()
    // sets maxTiles = 1.2*count, reallocates interactingTiles/interactingAtoms,
    // and rebuilds. Capturing maxTiles (and the arrays) once at setup therefore
    // goes stale the first time the list outgrows its initial 20*numBlocks.
    //
    // The failure is silent and total: with a stale maxTiles the kernels' own
    // `if (numTiles > maxTiles) return;` guard skips the ENTIRE neighbour-list
    // loop, leaving only the exclusion tiles. On real 1AAY that was
    // interactionCount = 62010 vs a captured maxTiles = 20500 -- energy came out
    // -172.5 instead of +2097.8, while every small test still passed because
    // small systems never overflow. Caught by the DEC-005 acceptance harness.
    {
        OpenMM::NonbondedUtilities& nbu = cc.getNonbondedUtilities();
        const unsigned int liveMaxTiles =
            (unsigned int) nbu.getInteractingTiles().getSize();
        buildEnvKernel->setArg(6, nbu.getInteractingTiles());
        buildEnvKernel->setArg(7, nbu.getInteractionCount());
        buildEnvKernel->setArg(8, nbu.getInteractingAtoms());
        buildEnvKernel->setArg(13, liveMaxTiles);
        pairKernel->setArg(5, nbu.getInteractingTiles());
        pairKernel->setArg(6, nbu.getInteractionCount());
        pairKernel->setArg(7, nbu.getInteractingAtoms());
        pairKernel->setArg(11, liveMaxTiles);
        // The rebuild flag is now read on the device by both kernels; it is
        // re-bound here with the rest of the neighbour-list state because
        // NonbondedUtilities may reallocate it, exactly as it does the tile
        // arrays (that reallocation is what the maxTiles bug above was).
        clearKernel->setArg(2, nbu.getRebuildNeighborList());
        buildEnvKernel->setArg(20, nbu.getRebuildNeighborList());
    }

    // Box vectors change every step under NPT, so they are set per call at the
    // index where the five placeholders were added.
    setPeriodicBoxArgs(cc, buildEnvKernel, 15);
    setPeriodicBoxArgs(cc, densityKernel, 8);
    setPeriodicBoxArgs(cc, pairKernel, 19);
    setPeriodicBoxArgs(cc, chainKernel, 10);
    OpenMM::NonbondedUtilities& nb = cc.getNonbondedUtilities();

    // LCWLD-150 timing probes. Kept alive across the move of the rebuild
    // decision onto the device: when they lived in the old host-side
    // needsEnvRebuild() they were deleted with it, and a mutation test that
    // relied on LOCALCWLD_NEVER_REBUILD then "passed" while changing nothing.
    int forceRebuild = (envEvalCount == 0) ? 1 : 0;
    if (envEvalCount > 0) {
        if (std::getenv("LOCALCWLD_ALWAYS_REBUILD") != nullptr)
            forceRebuild = 1;     // probe B: rebuild every step
        else if (std::getenv("LOCALCWLD_NEVER_REBUILD") != nullptr)
            forceRebuild = -1;    // probe C: never rebuild again (physics WRONG)
    }
    clearKernel->setArg(4, forceRebuild);
    buildEnvKernel->setArg(21, forceRebuild);

    // LCWLD-150 probe: put the per-step blocking read back, and change nothing
    // else. The three-point model (design doc section 9) inferred this cost at
    // 0.099 f-units from probes B and C; removing it for real bought only
    // 0.021. Inference and experiment disagree by 5x, so measure it directly
    // rather than argue about which equation is wrong.
    if (std::getenv("LOCALCWLD_PROBE_SYNC") != nullptr) {
        std::vector<int> discard(1);
        cc.getNonbondedUtilities().getRebuildNeighborList().download(discard);
    }

    clearKernel->execute(cc.getPaddedNumAtoms());

    // Launch unconditionally; the kernel returns immediately when the platform
    // did not rebuild its neighbour list and the host is not forcing one. See
    // the comment at the top of buildEnvPairs for why the check moved to the
    // device.
    buildEnvKernel->execute(pairNumThreads, pairBlockSize);
    checkEnvCapacity();

    // These two stride over the compact pair list, which is an order of
    // magnitude longer than the atom count; launching one thread per atom left
    // each thread with ~9 pairs and the device underfilled.
    densityKernel->execute(pairNumThreads, pairBlockSize);
    qKernel->execute(cc.getPaddedNumAtoms());
    // Block size must equal LOCAL_BUFFER_SIZE: the kernel derives the base of
    // its LOCAL window as LOCAL_ID - tgx, which runs off the end otherwise.
    pairKernel->execute(pairNumThreads, pairBlockSize);
    chainKernel->execute(pairNumThreads, pairBlockSize);
    return 0.0;   // energy goes through the platform's energy buffer
}

void CommonCalcLocalCWLDForceKernel::copyParametersToContext(ContextImpl& context,
                                                             const LocalCWLDForce& force) {
    OpenMM::ContextSelector selector(cc);
    if (force.getNumParticles() != cc.getNumAtoms())
        throw OpenMMException("LocalCWLDForce: particle count changed");
    uploadParameters(force);
    // Per-particle values only (the ForceImpl already refused shape changes),
    // so the r_env list topology is unaffected and does not need rebuilding.
    cc.invalidateMolecules();
}

/*
 * ---------------------------------------------------------------------------
 * NEIGHBOUR LIST: the one piece still open, and why it is not guessed at.
 * ---------------------------------------------------------------------------
 * The kernels in localCWLD.cc currently take a flat per-atom neighbour list
 * (numNeighbors / neighborStart / neighbors). OpenMM does not provide that
 * shape. What NonbondedUtilities exposes is its tiled list:
 *
 *     getInteractionCount()   number of interacting tile pairs
 *     getInteractingTiles()   the x-block of each interaction
 *     getInteractingAtoms()   TILE_SIZE atom indices per interaction
 *     getExclusionTiles() / getExclusions() / getExclusionIndices()
 *
 * Two ways to close the gap:
 *   (a) rewrite the two traversal kernels against the tiled format, reusing
 *       the platform's list directly -- fastest, but the exact tile/exclusion
 *       layout is not documented in the installed headers, only in OpenMM's
 *       own sources, which are not present in this environment;
 *   (b) build our own flat list on device from the tiled one once per rebuild,
 *       then run both hot passes over compact lists -- one extra pass at
 *       rebuild time, amortised the same way the r_env list already is.
 *
 * (b) is the safer first version and keeps the 2.12x structure intact, because
 * the extra traversal happens on the rebuild schedule, not per step. It is
 * deliberately NOT written from a guess at the layout: getting the tile walk
 * subtly wrong produces a neighbour list that is merely *incomplete*, which
 * shows up as small force errors that look like precision noise -- exactly the
 * failure mode DEC-005's thresholds are calibrated to catch, and exactly the
 * kind of bug that is expensive to find later.
 *
 * execute() therefore throws a clear message rather than returning plausible
 * numbers from a half-wired path.
 */
