#ifndef OPENMM_COMMONLOCALCWLDKERNELS_H_
#define OPENMM_COMMONLOCALCWLDKERNELS_H_

#include <string>
#include <vector>

#include "openmm/LocalCWLDForce.h"
#include "openmm/LocalCWLDKernels.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeKernel.h"

namespace LocalCWLDPlugin {

/**
 * Common-compute implementation of LocalCWLDForce (ticket LCWLD-090).
 *
 * Platform-independent host layer: allocates the device arrays, keeps the
 * compact r_env pair list in sync with the platform's neighbour list, and
 * launches the four kernels in `platforms/common/kernels/localCWLD.cc`. The
 * physics is entirely in the device code; nothing here computes anything.
 *
 * The r_env pair list is the reason this class is more than a launcher. See
 * the header comment of localCWLD.cc: rebuilding it every step gives 1.36x,
 * rebuilding it on the platform's own neighbour-list schedule gives 2.12x, and
 * not building it at all gives nothing.
 *
 * That decision no longer lives on the host. Reading OpenMM's rebuild flag here
 * cost a blocking device->host transfer every force evaluation, measured at
 * 16.4% of this force's entire cost (design doc section 9) -- more than three
 * times the amortised cost of the rebuild it was gating. buildEnvPairs now
 * reads the flag on the device and returns immediately when it is clear, and
 * the host only checks capacity, rarely.
 */
class CommonCalcLocalCWLDForceKernel : public CalcLocalCWLDForceKernel {
public:
    CommonCalcLocalCWLDForceKernel(std::string name, const OpenMM::Platform& platform,
                                   OpenMM::ComputeContext& cc)
        : CalcLocalCWLDForceKernel(name, platform), cc(cc), hasInitializedKernels(false),
          maxEnvPairs(0) {
    }

    void initialize(const OpenMM::System& system, const LocalCWLDForce& force) override;
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;
    void copyParametersToContext(OpenMM::ContextImpl& context,
                                 const LocalCWLDForce& force) override;

private:
    /** Upload the nine per-particle arrays, deriving `source` and `amplitude`. */
    void uploadParameters(const LocalCWLDForce& force);
    /** Backstop check that the compact list did not overflow. Blocking; rare. */
    void checkEnvCapacity();

    OpenMM::ComputeContext& cc;
    bool hasInitializedKernels;

    // --- frozen globals, captured at initialize() -------------------------
    double envCutoff, cutoff, rho0, kPolar, chargeDeltaClamp, one4PiEps0;
    int zmmOrder;
    bool useQPenalty;
    double qPenaltyStrength;
    /** ZMM closure polynomial coefficients for the frozen order. */
    double c0, c2, c4, c6;

    // --- per-particle device arrays (plan section 17.2 order) -------------
    OpenMM::ComputeArray qbase, source, amplitude, densSink, dQdDens, charge, dens;
    OpenMM::ComputeArray residueId;

    // --- scratch ----------------------------------------------------------
    OpenMM::ComputeArray densBuffer, adjointBuffer, status;
    OpenMM::ComputeArray envPairs, envPairCount;
    int maxEnvPairs;
    /** Force evaluations seen, used to space out the blocking capacity check. */
    int envEvalCount = 0;
    /** LCWLD-150 diagnostics: how many rebuilds, and how far apart. */
    /** Launch geometry for the tile kernels, taken from NonbondedUtilities. */
    int pairBlockSize = 0, pairNumThreads = 0;
    int envRebuildCount = 0;
    int lastRebuildForceCount = -1;

    OpenMM::ComputeKernel clearKernel, buildEnvKernel, densityKernel,
                          qKernel, pairKernel, chainKernel;
};

} // namespace LocalCWLDPlugin

#endif /* OPENMM_COMMONLOCALCWLDKERNELS_H_ */
