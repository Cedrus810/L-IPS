#include "openmm/internal/LocalCWLDForceImpl.h"

#include <algorithm>
#include <sstream>

#include "openmm/LocalCWLDKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace LocalCWLDPlugin;
using OpenMM::ContextImpl;
using OpenMM::OpenMMException;

namespace {

/** Snapshot the exclusions as sorted, normalized pairs. */
std::vector<std::pair<int, int> > snapshotExclusions(const LocalCWLDForce& force) {
    std::vector<std::pair<int, int> > out;
    out.reserve(force.getNumExclusions());
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int p1, p2;
        force.getExclusionParticles(i, p1, p2);
        out.push_back(std::make_pair(p1, p2));   // already (min, max) from the API
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace

LocalCWLDForceImpl::LocalCWLDForceImpl(const LocalCWLDForce& owner)
    : owner(owner), initializedNumParticles(-1) {
}

void LocalCWLDForceImpl::initialize(ContextImpl& context) {
    const OpenMM::System& system = context.getSystem();

    // The per-particle arrays are indexed by System particle index. A mismatch
    // would not throw anywhere downstream -- it would silently pair particle i
    // with particle j's charge -- so it has to be caught here.
    if (owner.getNumParticles() != system.getNumParticles()) {
        std::ostringstream message;
        message << "LocalCWLDForce: the Force has " << owner.getNumParticles()
                << " particles but the System has " << system.getNumParticles()
                << "; every per-particle parameter would be misaligned";
        throw OpenMMException(message.str());
    }

    // v0.1 is CutoffPeriodic only, so a non-periodic System is a contradiction:
    // the minimum-image convention the kernel uses would have no box to apply.
    if (!system.usesPeriodicBoundaryConditions()) {
        throw OpenMMException(
            "LocalCWLDForce: v0.1 supports only CutoffPeriodic, but the System "
            "has no periodic box vectors");
    }

    // rc must fit in the box. OpenMM's own nonbonded forces make the same check;
    // without it a particle interacts with several images of the same neighbour.
    OpenMM::Vec3 a, b, c;
    system.getDefaultPeriodicBoxVectors(a, b, c);
    const double cutoff = owner.getCutoffDistance();
    if (a[0] < 2 * cutoff || b[1] < 2 * cutoff || c[2] < 2 * cutoff) {
        std::ostringstream message;
        message << "LocalCWLDForce: the periodic box (" << a[0] << ", " << b[1]
                << ", " << c[2] << " nm) is too small for a cutoff of " << cutoff
                << " nm; each side must be at least twice the cutoff";
        throw OpenMMException(message.str());
    }

    // Exclusion indices are range-checked here rather than in addExclusion(),
    // because exclusions may legitimately be added before their particles.
    for (int i = 0; i < owner.getNumExclusions(); i++) {
        int p1, p2;
        owner.getExclusionParticles(i, p1, p2);
        if (p1 >= owner.getNumParticles() || p2 >= owner.getNumParticles()) {
            std::ostringstream message;
            message << "LocalCWLDForce: exclusion " << i << " refers to particles ("
                    << p1 << ", " << p2 << ") but the Force has only "
                    << owner.getNumParticles() << " particles";
            throw OpenMMException(message.str());
        }
    }

    initializedNumParticles = owner.getNumParticles();
    initializedExclusions = snapshotExclusions(owner);

    kernel = context.getPlatform().createKernel(CalcLocalCWLDForceKernel::Name(), context);
    kernel.getAs<CalcLocalCWLDForceKernel>().initialize(system, owner);
}

double LocalCWLDForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces,
                                               bool includeEnergy, int groups) {
    // Not in the requested force group: contribute exactly zero, and do not
    // touch the kernel. Returning anything else here silently corrupts every
    // multi-timestep or force-group-decomposed calculation.
    if ((groups & (1 << owner.getForceGroup())) == 0)
        return 0.0;
    return kernel.getAs<CalcLocalCWLDForceKernel>().execute(context, includeForces, includeEnergy);
}

std::vector<std::string> LocalCWLDForceImpl::getKernelNames() {
    return std::vector<std::string>(1, CalcLocalCWLDForceKernel::Name());
}

void LocalCWLDForceImpl::updateParametersInContext(ContextImpl& context) {
    // v0.1 allows updating per-particle *values* only. Both checks below guard
    // against a caller that changed the shape and expects it to take effect:
    // the device buffers were sized at initialize(), so the change would either
    // be ignored or read out of bounds.
    if (owner.getNumParticles() != initializedNumParticles) {
        std::ostringstream message;
        message << "LocalCWLDForce: updateParametersInContext() cannot change the "
                << "particle count (" << initializedNumParticles << " -> "
                << owner.getNumParticles() << "); create a new Context instead";
        throw OpenMMException(message.str());
    }
    if (snapshotExclusions(owner) != initializedExclusions) {
        throw OpenMMException(
            "LocalCWLDForce: updateParametersInContext() cannot change the exclusion "
            "topology; create a new Context instead");
    }
    kernel.getAs<CalcLocalCWLDForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}
