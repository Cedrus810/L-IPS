#ifndef OPENMM_LOCALCWLDFORCE_H_
#define OPENMM_LOCALCWLDFORCE_H_

#include <vector>

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "internal/windowsExportLocalCWLD.h"

namespace LocalCWLDPlugin {

/**
 * LocalCWLDForce -- local-environment charge response (CWLD).
 *
 * Each particle accumulates a local density from nearby "source" particles,
 * that density shifts its charge by a bounded amount, and the shifted charges
 * interact through a zero-multipole (ZMM) shifted pair potential. The force
 * therefore has a direct term and a chain term (dU/dQ * dQ/ddensity *
 * ddensity/dr); see plan sections 18.1-18.6 for the frozen mathematics.
 *
 * Scope of this class (ticket LCWLD-040): it is a pure parameter container.
 * It performs no classification, holds no topology, and touches no device
 * state. Deciding which particle is water, which is an ion, or what its
 * charge_mod should be happens in the Python builder before the Context
 * exists. Nothing here recomputes those per step.
 *
 * Units follow OpenMM throughout: nm, e, kJ/mol, kJ/mol/nm.
 *
 * Precision: the computation is single precision (DEC-004), matching what the
 * production CUDA platform already does. This API still exchanges doubles --
 * that is the OpenMM convention for Force parameters and says nothing about
 * the working precision of the kernel.
 */
class OPENMM_EXPORT_LOCALCWLD LocalCWLDForce : public OpenMM::Force {
public:
    /**
     * Supported nonbonded methods. v0.1 is periodic-with-cutoff only; the enum
     * exists so adding a method later does not change the call signature.
     */
    enum NonbondedMethod {
        CutoffPeriodic = 0
    };

    /** Construct a force with no particles and the plan section 17.1 defaults. */
    LocalCWLDForce();

    /**
     * Add a particle. Parameters are in the frozen order of plan section 17.2;
     * that order is shared by the C++ API, serialization, the Python wrapper,
     * the fixtures and the device arrays, so it must not be permuted anywhere.
     *
     * @param qbase              base charge, e
     * @param chargeMod          build-time density-source modifier
     * @param dpolar             response amplitude and sign
     * @param isPolar            response mask/weight
     * @param densSource         whether the particle contributes density
     * @param densSink           whether the particle receives density
     * @param sourceClassWeight  class weight used when acting as a source
     * @param staticPhase        response on/off
     * @param residueId          residue.index; used only to suppress
     *                           same-residue density, never to reclassify
     * @return the index of the new particle
     */
    int addParticle(double qbase, double chargeMod, double dpolar,
                    double isPolar, double densSource, double densSink,
                    double sourceClassWeight, double staticPhase,
                    int residueId);

    /** Number of particles. Must equal System::getNumParticles() at Context creation. */
    int getNumParticles() const;

    /** Read back one particle's parameters, in the section 17.2 order. */
    void getParticleParameters(int index, double& qbase, double& chargeMod,
                               double& dpolar, double& isPolar, double& densSource,
                               double& densSink, double& sourceClassWeight,
                               double& staticPhase, int& residueId) const;

    /**
     * Overwrite one particle's parameters. Call updateParametersInContext()
     * afterwards to push the change to a running Context.
     */
    void setParticleParameters(int index, double qbase, double chargeMod,
                               double dpolar, double isPolar, double densSource,
                               double densSink, double sourceClassWeight,
                               double staticPhase, int residueId);

    /**
     * Add an excluded pair. Order is normalized to (min, max), so (i, j) and
     * (j, i) are the same exclusion. Self-exclusions are rejected and
     * duplicates are rejected -- silently swallowing either would hide a
     * builder bug that is otherwise very hard to see in the results.
     *
     * @return the index of the new exclusion
     */
    int addExclusion(int particle1, int particle2);

    /** Number of exclusions. */
    int getNumExclusions() const;

    /** Read back one exclusion; always returned as (min, max). */
    void getExclusionParticles(int index, int& particle1, int& particle2) const;

    /** Replace one exclusion. Same normalization and duplicate rules as addExclusion(). */
    void setExclusionParticles(int index, int particle1, int particle2);

    /** Density support radius r_env, nm. Default 0.35. */
    double getEnvironmentCutoff() const;
    /** Set r_env. Must be finite, > 0, and <= the pair cutoff. */
    void setEnvironmentCutoff(double distance);

    /** Pair cutoff rc, nm. Default 1.2. */
    double getCutoffDistance() const;
    /** Set rc. Must be finite, > 0, and >= r_env. */
    void setCutoffDistance(double distance);

    /** ZMM closure order ell. Default 2; only 1, 2 and 3 are defined. */
    int getZMMOrder() const;
    /** Set the ZMM order. Must be 1, 2 or 3. */
    void setZMMOrder(int order);

    /** Density scale rho0. Default 13.5. */
    double getRho0() const;
    /** Set rho0. Must be finite and > 0. */
    void setRho0(double value);

    /** Response coefficient k_polar. Default 0.8. */
    double getKPolar() const;
    /** Set k_polar. Must be finite. */
    void setKPolar(double value);

    /** Soft bound on |Q - qbase|, e. Default 0.2. */
    double getChargeDeltaClamp() const;
    /** Set the charge-delta clamp. Must be finite and > 0. */
    void setChargeDeltaClamp(double value);

    /** Whether the quadratic Q penalty is active. Default false. */
    bool getUseQPenalty() const;
    /** Enable or disable the Q penalty. */
    void setUseQPenalty(bool enabled);

    /** Q penalty strength. Default 180.0; used only when the penalty is enabled. */
    double getQPenaltyStrength() const;
    /** Set the Q penalty strength. Must be finite and >= 0. */
    void setQPenaltyStrength(double value);

    /** Nonbonded method. v0.1 accepts only CutoffPeriodic. */
    NonbondedMethod getNonbondedMethod() const;
    /** Set the nonbonded method. Anything but CutoffPeriodic is rejected. */
    void setNonbondedMethod(NonbondedMethod method);

    /**
     * Electrostatic constant, kJ/mol nm/e^2. Frozen at 138.935458 for v0.1.
     *
     * The setter exists for testing only -- for example, checking a fixture
     * against an analytically simpler constant. It is NOT a production knob;
     * changing it changes the physics and every golden value with it.
     */
    double getOne4PiEps0() const;
    /** Test-only; see getOne4PiEps0(). Must be finite and > 0. */
    void setOne4PiEps0(double value);

    /**
     * Push updated per-particle parameters into an existing Context.
     *
     * v0.1 updates per-particle *values* only. The number of particles, the
     * exclusion list and every global parameter are fixed once the Context
     * exists; changing them requires a new Context. This restriction is
     * enforced by the ForceImpl (LCWLD-050), not silently ignored here.
     */
    void updateParametersInContext(OpenMM::Context& context);

    /** Always true: v0.1 is periodic-with-cutoff only. */
    bool usesPeriodicBoundaryConditions() const override;

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    class ParticleInfo;
    class ExclusionInfo;

    double environmentCutoff, cutoffDistance;
    double rho0, kPolar, chargeDeltaClamp;
    double qPenaltyStrength, one4PiEps0;
    int zmmOrder;
    bool useQPenalty;
    NonbondedMethod nonbondedMethod;
    std::vector<ParticleInfo> particles;
    std::vector<ExclusionInfo> exclusions;
};

/** Per-particle storage, in the plan section 17.2 order. Internal. */
class LocalCWLDForce::ParticleInfo {
public:
    double qbase, chargeMod, dpolar, isPolar;
    double densSource, densSink, sourceClassWeight, staticPhase;
    int residueId;
    ParticleInfo()
        : qbase(0.0), chargeMod(1.0), dpolar(0.0), isPolar(0.0),
          densSource(0.0), densSink(0.0), sourceClassWeight(0.0),
          staticPhase(0.0), residueId(0) {
    }
    ParticleInfo(double qbase, double chargeMod, double dpolar, double isPolar,
                 double densSource, double densSink, double sourceClassWeight,
                 double staticPhase, int residueId)
        : qbase(qbase), chargeMod(chargeMod), dpolar(dpolar), isPolar(isPolar),
          densSource(densSource), densSink(densSink),
          sourceClassWeight(sourceClassWeight), staticPhase(staticPhase),
          residueId(residueId) {
    }
};

/** One excluded pair, stored normalized as (min, max). Internal. */
class LocalCWLDForce::ExclusionInfo {
public:
    int particle1, particle2;
    ExclusionInfo() : particle1(-1), particle2(-1) {
    }
    ExclusionInfo(int particle1, int particle2)
        : particle1(particle1), particle2(particle2) {
    }
};

} // namespace LocalCWLDPlugin

#endif /* OPENMM_LOCALCWLDFORCE_H_ */
