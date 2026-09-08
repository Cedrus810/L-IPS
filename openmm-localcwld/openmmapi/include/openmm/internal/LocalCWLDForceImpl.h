#ifndef OPENMM_LOCALCWLDFORCEIMPL_H_
#define OPENMM_LOCALCWLDFORCEIMPL_H_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "openmm/Kernel.h"
#include "openmm/LocalCWLDForce.h"
#include "openmm/internal/ForceImpl.h"

namespace LocalCWLDPlugin {

/**
 * ForceImpl for LocalCWLDForce (ticket LCWLD-050).
 *
 * Deliberately thin. Its whole job is: validate what the kernel is allowed to
 * assume, obtain the kernel, and forward calls while respecting force groups
 * and the includeForces/includeEnergy flags. **No numerical algorithm may be
 * added here** -- that belongs to LCWLD-070 (Reference) and LCWLD-100 (device).
 *
 * The validation is the point. Every check below exists because getting it
 * wrong produces a plausible-looking trajectory rather than an error:
 *   - particle count mismatch would silently shift every per-particle array;
 *   - a non-periodic System would make the minimum-image convention wrong;
 *   - changing the exclusion topology in updateParametersInContext() would
 *     leave the device neighbour bookkeeping stale.
 */
class LocalCWLDForceImpl : public OpenMM::ForceImpl {
public:
    explicit LocalCWLDForceImpl(const LocalCWLDForce& owner);
    ~LocalCWLDForceImpl() override = default;

    void initialize(OpenMM::ContextImpl& context) override;

    const LocalCWLDForce& getOwner() const override {
        return owner;
    }

    /** LocalCWLDForce contributes nothing between integration steps. */
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) override {
    }

    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces,
                               bool includeEnergy, int groups) override;

    /** No Context-settable global parameters in v0.1. */
    std::map<std::string, double> getDefaultParameters() override {
        return {};
    }

    std::vector<std::string> getKernelNames() override;

    /**
     * Forward updated per-particle parameters to the kernel, after checking
     * that only values changed -- not the particle count, not the exclusions.
     */
    void updateParametersInContext(OpenMM::ContextImpl& context);

private:
    const LocalCWLDForce& owner;
    OpenMM::Kernel kernel;
    /** Particle count captured at initialize(), to detect later divergence. */
    int initializedNumParticles;
    /** Exclusions captured at initialize(), normalized (min, max) and sorted. */
    std::vector<std::pair<int, int> > initializedExclusions;
};

} // namespace LocalCWLDPlugin

#endif /* OPENMM_LOCALCWLDFORCEIMPL_H_ */
