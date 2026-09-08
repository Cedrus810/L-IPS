#ifndef OPENMM_LOCALCWLDKERNELS_H_
#define OPENMM_LOCALCWLDKERNELS_H_

#include <string>

#include "openmm/KernelImpl.h"
#include "openmm/LocalCWLDForce.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"

namespace LocalCWLDPlugin {

/**
 * The contract every platform implementation of LocalCWLDForce must satisfy
 * (ticket LCWLD-050).
 *
 * This header is the boundary between "what the force means" and "how a
 * particular device computes it". Nothing numerical lives here, and no
 * implementation of this interface may re-derive per-particle classification:
 * every parameter arrives already decided by the Python builder, in the frozen
 * order of plan section 17.2.
 *
 * Implementors: LCWLD-070 (Reference CPU), LCWLD-090/100/110 (common + CUDA).
 */
class CalcLocalCWLDForceKernel : public OpenMM::KernelImpl {
public:
    /**
     * The name platforms register this kernel under. It is a function rather
     * than a constant so that every implementation is forced to agree with the
     * ForceImpl on one spelling.
     */
    static std::string Name() {
        return "CalcLocalCWLDForce";
    }

    CalcLocalCWLDForceKernel(std::string name, const OpenMM::Platform& platform)
        : OpenMM::KernelImpl(name, platform) {
    }

    /**
     * Called once, before any execute(). Copies whatever the device needs out
     * of `force`; the Force object must not be retained.
     *
     * @param system  the System the Context is being created for
     * @param force   the LocalCWLDForce being implemented
     */
    virtual void initialize(const OpenMM::System& system, const LocalCWLDForce& force) = 0;

    /**
     * Compute forces and/or energy for the current positions.
     *
     * @param context        the Context to compute in
     * @param includeForces  whether forces should be accumulated
     * @param includeEnergy  whether the return value is meaningful
     * @return the potential energy in kJ/mol, or 0 when includeEnergy is false
     */
    virtual double execute(OpenMM::ContextImpl& context,
                           bool includeForces, bool includeEnergy) = 0;

    /**
     * Push updated per-particle parameters into a live Context.
     *
     * v0.1 contract: per-particle *values* only. The particle count, the
     * exclusion list and every global parameter are fixed at Context creation.
     * The ForceImpl enforces that before calling this, so an implementation may
     * assume the shape is unchanged -- but must not assume the values are.
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context,
                                         const LocalCWLDForce& force) = 0;
};

} // namespace LocalCWLDPlugin

#endif /* OPENMM_LOCALCWLDKERNELS_H_ */
