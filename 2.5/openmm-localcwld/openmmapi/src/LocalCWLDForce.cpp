#include "openmm/LocalCWLDForce.h"

#include <cmath>
#include <sstream>
#include <string>

#include "openmm/OpenMMException.h"
#include "openmm/internal/LocalCWLDForceImpl.h"

using namespace LocalCWLDPlugin;
using OpenMM::OpenMMException;

namespace {

// Every setter validates. Plan section 20 is explicit that NaN, negative
// cutoffs and bad indices must raise rather than be silently corrected -- a
// quietly clamped NaN reappears later as an unexplainable trajectory.
void requireFinite(double value, const char* name) {
    if (!std::isfinite(value)) {
        std::ostringstream message;
        message << "LocalCWLDForce: " << name << " must be finite";
        throw OpenMMException(message.str());
    }
}

void requirePositive(double value, const char* name) {
    requireFinite(value, name);
    if (value <= 0.0) {
        std::ostringstream message;
        message << "LocalCWLDForce: " << name << " must be positive, got " << value;
        throw OpenMMException(message.str());
    }
}

void requireIndex(int index, int size, const char* what) {
    if (index < 0 || index >= size) {
        std::ostringstream message;
        message << "LocalCWLDForce: " << what << " index " << index
                << " out of range [0, " << size << ")";
        throw OpenMMException(message.str());
    }
}

} // namespace

LocalCWLDForce::LocalCWLDForce()
    : environmentCutoff(0.35),      // plan section 17.1 defaults
      cutoffDistance(1.2),
      rho0(13.5),
      kPolar(0.8),
      chargeDeltaClamp(0.2),
      qPenaltyStrength(180.0),
      one4PiEps0(138.935458),
      zmmOrder(2),
      useQPenalty(false),
      nonbondedMethod(CutoffPeriodic) {
}

int LocalCWLDForce::addParticle(double qbase, double chargeMod, double dpolar,
                                double isPolar, double densSource, double densSink,
                                double sourceClassWeight, double staticPhase,
                                int residueId) {
    requireFinite(qbase, "qbase");
    requireFinite(chargeMod, "chargeMod");
    requireFinite(dpolar, "dpolar");
    requireFinite(isPolar, "isPolar");
    requireFinite(densSource, "densSource");
    requireFinite(densSink, "densSink");
    requireFinite(sourceClassWeight, "sourceClassWeight");
    requireFinite(staticPhase, "staticPhase");
    if (residueId < 0)
        throw OpenMMException("LocalCWLDForce: residueId must be non-negative");
    particles.push_back(ParticleInfo(qbase, chargeMod, dpolar, isPolar, densSource,
                                     densSink, sourceClassWeight, staticPhase, residueId));
    return static_cast<int>(particles.size()) - 1;
}

int LocalCWLDForce::getNumParticles() const {
    return static_cast<int>(particles.size());
}

void LocalCWLDForce::getParticleParameters(int index, double& qbase, double& chargeMod,
                                           double& dpolar, double& isPolar, double& densSource,
                                           double& densSink, double& sourceClassWeight,
                                           double& staticPhase, int& residueId) const {
    requireIndex(index, static_cast<int>(particles.size()), "particle");
    const ParticleInfo& p = particles[index];
    qbase = p.qbase;
    chargeMod = p.chargeMod;
    dpolar = p.dpolar;
    isPolar = p.isPolar;
    densSource = p.densSource;
    densSink = p.densSink;
    sourceClassWeight = p.sourceClassWeight;
    staticPhase = p.staticPhase;
    residueId = p.residueId;
}

void LocalCWLDForce::setParticleParameters(int index, double qbase, double chargeMod,
                                           double dpolar, double isPolar, double densSource,
                                           double densSink, double sourceClassWeight,
                                           double staticPhase, int residueId) {
    requireIndex(index, static_cast<int>(particles.size()), "particle");
    requireFinite(qbase, "qbase");
    requireFinite(chargeMod, "chargeMod");
    requireFinite(dpolar, "dpolar");
    requireFinite(isPolar, "isPolar");
    requireFinite(densSource, "densSource");
    requireFinite(densSink, "densSink");
    requireFinite(sourceClassWeight, "sourceClassWeight");
    requireFinite(staticPhase, "staticPhase");
    if (residueId < 0)
        throw OpenMMException("LocalCWLDForce: residueId must be non-negative");
    particles[index] = ParticleInfo(qbase, chargeMod, dpolar, isPolar, densSource,
                                    densSink, sourceClassWeight, staticPhase, residueId);
}

int LocalCWLDForce::addExclusion(int particle1, int particle2) {
    if (particle1 < 0 || particle2 < 0) {
        std::ostringstream message;
        message << "LocalCWLDForce: exclusion indices must be non-negative, got ("
                << particle1 << ", " << particle2 << ")";
        throw OpenMMException(message.str());
    }
    if (particle1 == particle2) {
        std::ostringstream message;
        message << "LocalCWLDForce: self exclusion (" << particle1 << ", " << particle2
                << ") is not allowed";
        throw OpenMMException(message.str());
    }
    // Normalize so (i, j) and (j, i) are the same pair, then reject duplicates.
    // Indices are deliberately NOT range-checked against getNumParticles():
    // exclusions may legitimately be added before the particles they refer to.
    // The particle-count check belongs to Context creation (LCWLD-050).
    const int low = particle1 < particle2 ? particle1 : particle2;
    const int high = particle1 < particle2 ? particle2 : particle1;
    for (std::size_t i = 0; i < exclusions.size(); i++) {
        if (exclusions[i].particle1 == low && exclusions[i].particle2 == high) {
            std::ostringstream message;
            message << "LocalCWLDForce: duplicate exclusion (" << low << ", " << high
                    << "), already present at index " << i;
            throw OpenMMException(message.str());
        }
    }
    exclusions.push_back(ExclusionInfo(low, high));
    return static_cast<int>(exclusions.size()) - 1;
}

int LocalCWLDForce::getNumExclusions() const {
    return static_cast<int>(exclusions.size());
}

void LocalCWLDForce::getExclusionParticles(int index, int& particle1, int& particle2) const {
    requireIndex(index, static_cast<int>(exclusions.size()), "exclusion");
    particle1 = exclusions[index].particle1;
    particle2 = exclusions[index].particle2;
}

void LocalCWLDForce::setExclusionParticles(int index, int particle1, int particle2) {
    requireIndex(index, static_cast<int>(exclusions.size()), "exclusion");
    if (particle1 < 0 || particle2 < 0) {
        std::ostringstream message;
        message << "LocalCWLDForce: exclusion indices must be non-negative, got ("
                << particle1 << ", " << particle2 << ")";
        throw OpenMMException(message.str());
    }
    if (particle1 == particle2) {
        std::ostringstream message;
        message << "LocalCWLDForce: self exclusion (" << particle1 << ", " << particle2
                << ") is not allowed";
        throw OpenMMException(message.str());
    }
    const int low = particle1 < particle2 ? particle1 : particle2;
    const int high = particle1 < particle2 ? particle2 : particle1;
    for (std::size_t i = 0; i < exclusions.size(); i++) {
        if (static_cast<int>(i) == index)
            continue;
        if (exclusions[i].particle1 == low && exclusions[i].particle2 == high) {
            std::ostringstream message;
            message << "LocalCWLDForce: duplicate exclusion (" << low << ", " << high
                    << "), already present at index " << i;
            throw OpenMMException(message.str());
        }
    }
    exclusions[index] = ExclusionInfo(low, high);
}

double LocalCWLDForce::getEnvironmentCutoff() const {
    return environmentCutoff;
}

void LocalCWLDForce::setEnvironmentCutoff(double distance) {
    requirePositive(distance, "r_env");
    if (distance > cutoffDistance) {
        std::ostringstream message;
        message << "LocalCWLDForce: r_env (" << distance << ") must not exceed rc ("
                << cutoffDistance << "); set the pair cutoff first";
        throw OpenMMException(message.str());
    }
    environmentCutoff = distance;
}

double LocalCWLDForce::getCutoffDistance() const {
    return cutoffDistance;
}

void LocalCWLDForce::setCutoffDistance(double distance) {
    requirePositive(distance, "rc");
    if (distance < environmentCutoff) {
        std::ostringstream message;
        message << "LocalCWLDForce: rc (" << distance << ") must not be smaller than r_env ("
                << environmentCutoff << "); lower r_env first";
        throw OpenMMException(message.str());
    }
    cutoffDistance = distance;
}

int LocalCWLDForce::getZMMOrder() const {
    return zmmOrder;
}

void LocalCWLDForce::setZMMOrder(int order) {
    if (order < 1 || order > 3) {
        std::ostringstream message;
        message << "LocalCWLDForce: zmm_order must be 1, 2 or 3, got " << order;
        throw OpenMMException(message.str());
    }
    zmmOrder = order;
}

double LocalCWLDForce::getRho0() const {
    return rho0;
}

void LocalCWLDForce::setRho0(double value) {
    requirePositive(value, "rho0");
    rho0 = value;
}

double LocalCWLDForce::getKPolar() const {
    return kPolar;
}

void LocalCWLDForce::setKPolar(double value) {
    // k_polar may legitimately be zero (response off) or negative; only
    // non-finite values are rejected.
    requireFinite(value, "k_polar");
    kPolar = value;
}

double LocalCWLDForce::getChargeDeltaClamp() const {
    return chargeDeltaClamp;
}

void LocalCWLDForce::setChargeDeltaClamp(double value) {
    // The clamp divides inside tanh(x/clamp); zero would be a division by zero.
    requirePositive(value, "q_delta_clamp");
    chargeDeltaClamp = value;
}

bool LocalCWLDForce::getUseQPenalty() const {
    return useQPenalty;
}

void LocalCWLDForce::setUseQPenalty(bool enabled) {
    useQPenalty = enabled;
}

double LocalCWLDForce::getQPenaltyStrength() const {
    return qPenaltyStrength;
}

void LocalCWLDForce::setQPenaltyStrength(double value) {
    requireFinite(value, "q_penalty_strength");
    if (value < 0.0) {
        std::ostringstream message;
        message << "LocalCWLDForce: q_penalty_strength must be non-negative, got " << value;
        throw OpenMMException(message.str());
    }
    qPenaltyStrength = value;
}

LocalCWLDForce::NonbondedMethod LocalCWLDForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void LocalCWLDForce::setNonbondedMethod(NonbondedMethod method) {
    if (method != CutoffPeriodic) {
        std::ostringstream message;
        message << "LocalCWLDForce: v0.1 supports only CutoffPeriodic, got "
                << static_cast<int>(method);
        throw OpenMMException(message.str());
    }
    nonbondedMethod = method;
}

double LocalCWLDForce::getOne4PiEps0() const {
    return one4PiEps0;
}

void LocalCWLDForce::setOne4PiEps0(double value) {
    requirePositive(value, "ONE_4PI_EPS0");
    one4PiEps0 = value;
}

bool LocalCWLDForce::usesPeriodicBoundaryConditions() const {
    return true;
}

//-----------------------------------------------------------------------------
// Bridge to the ForceImpl (LCWLD-050). Both bodies stay one-liners on purpose:
// validation and kernel dispatch belong to LocalCWLDForceImpl, not here.
//-----------------------------------------------------------------------------

OpenMM::ForceImpl* LocalCWLDForce::createImpl() const {
    return new LocalCWLDForceImpl(*this);
}

void LocalCWLDForce::updateParametersInContext(OpenMM::Context& context) {
    dynamic_cast<LocalCWLDForceImpl&>(getImplInContext(context))
        .updateParametersInContext(getContextImpl(context));
}
