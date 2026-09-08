/**
 * LCWLD-050 tests: ForceImpl behaviour, exercised through a fake kernel.
 *
 * A fake kernel rather than a real one is the whole point of this ticket: it
 * lets us assert *that* the ForceImpl calls initialize/execute/update, with
 * what arguments and how many times, without any numerics existing yet. The
 * real kernels arrive in LCWLD-070 and LCWLD-100.
 *
 * What is checked here is exactly the set of mistakes that would otherwise
 * produce a plausible-looking trajectory instead of an error:
 *   - a force group the caller did not ask for must contribute zero, and must
 *     not even reach the kernel;
 *   - includeForces / includeEnergy must be forwarded verbatim;
 *   - a particle-count or exclusion-topology change must be refused, not
 *     silently ignored against device buffers sized at initialize().
 */

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "openmm/Context.h"
#include "openmm/LocalCWLDForce.h"
#include "openmm/KernelFactory.h"
#include "openmm/LocalCWLDKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/internal/ContextImpl.h"

using namespace LocalCWLDPlugin;
using OpenMM::OpenMMException;

static int failures = 0;

static void check(bool condition, const std::string& what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what.c_str());
        failures++;
    }
}

template <typename Body>
static void checkThrows(Body body, const std::string& what) {
    try {
        body();
    } catch (const OpenMMException&) {
        return;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s (threw %s, not OpenMMException)\n", what.c_str(), e.what());
        failures++;
        return;
    }
    std::printf("FAIL: %s (did not throw)\n", what.c_str());
    failures++;
}

//--- Fake kernel: records calls, computes nothing --------------------------

struct CallLog {
    int initializeCalls = 0;
    int executeCalls = 0;
    int updateCalls = 0;
    bool lastIncludeForces = false;
    bool lastIncludeEnergy = false;
    int lastNumParticles = -1;
    double lastQbase0 = 0.0;
};

static CallLog log_;

class FakeLocalCWLDKernel : public CalcLocalCWLDForceKernel {
public:
    FakeLocalCWLDKernel(std::string name, const OpenMM::Platform& platform)
        : CalcLocalCWLDForceKernel(name, platform) {
    }
    void initialize(const OpenMM::System& system, const LocalCWLDForce& force) override {
        log_.initializeCalls++;
        log_.lastNumParticles = force.getNumParticles();
    }
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override {
        log_.executeCalls++;
        log_.lastIncludeForces = includeForces;
        log_.lastIncludeEnergy = includeEnergy;
        return 42.0;   // a value no real computation would produce by accident
    }
    void copyParametersToContext(OpenMM::ContextImpl& context,
                                 const LocalCWLDForce& force) override {
        log_.updateCalls++;
        double qbase, other;
        int residueId;
        force.getParticleParameters(0, qbase, other, other, other, other, other, other,
                                    other, residueId);
        log_.lastQbase0 = qbase;
    }
};

/**
 * Register the fake kernel into the Reference platform.
 *
 * This is how OpenMM plugins actually work -- a platform plugin calls
 * `Platform::registerKernelFactory` on an existing platform rather than
 * defining a new one. Trying to define a standalone Platform here failed the
 * way it should: a Platform must supply *every* required kernel (integrator,
 * state, forces...), not just the one under test.
 */
class FakeFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform,
                                         OpenMM::ContextImpl& context) const override {
        return new FakeLocalCWLDKernel(name, platform);
    }
};

//--- Helpers ---------------------------------------------------------------

/** A minimal periodic System with `n` particles in a box big enough for rc. */
static void buildSystem(OpenMM::System& system, int n, double boxSide = 4.0) {
    for (int i = 0; i < n; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(OpenMM::Vec3(boxSide, 0, 0),
                                        OpenMM::Vec3(0, boxSide, 0),
                                        OpenMM::Vec3(0, 0, boxSide));
}

static LocalCWLDForce* buildForce(int n) {
    LocalCWLDForce* force = new LocalCWLDForce();
    for (int i = 0; i < n; i++)
        force->addParticle(0.1 * i, 1.0, 0.012, 1.0, 1.0, 1.0, 1.0, 1.0, i);
    return force;
}

//--- Tests -----------------------------------------------------------------

static void testInitializeAndExecute(OpenMM::Platform& platform) {
    log_ = CallLog();
    OpenMM::System system;
    buildSystem(system, 4);
    LocalCWLDForce* force = buildForce(4);
    system.addForce(force);

    OpenMM::VerletIntegrator integrator(0.001);
    OpenMM::Context context(system, integrator, platform);
    context.setPositions(std::vector<OpenMM::Vec3>(4, OpenMM::Vec3(0, 0, 0)));

    check(log_.initializeCalls == 1, "kernel initialized exactly once");
    check(log_.lastNumParticles == 4, "kernel saw the right particle count");

    context.getState(OpenMM::State::Energy);
    check(log_.executeCalls == 1, "one execute for an energy request");
    check(log_.lastIncludeEnergy, "includeEnergy forwarded as true");

    context.getState(OpenMM::State::Forces);
    check(log_.lastIncludeForces, "includeForces forwarded as true");

    const double energy =
        context.getState(OpenMM::State::Energy).getPotentialEnergy();
    check(energy == 42.0, "the kernel's energy reaches the State unchanged");
}

static void testForceGroupIsRespected(OpenMM::Platform& platform) {
    log_ = CallLog();
    OpenMM::System system;
    buildSystem(system, 4);
    LocalCWLDForce* force = buildForce(4);
    force->setForceGroup(3);
    system.addForce(force);

    OpenMM::VerletIntegrator integrator(0.001);
    OpenMM::Context context(system, integrator, platform);
    context.setPositions(std::vector<OpenMM::Vec3>(4, OpenMM::Vec3(0, 0, 0)));

    const int before = log_.executeCalls;
    const double excluded =
        context.getState(OpenMM::State::Energy, false, 1 << 1).getPotentialEnergy();
    check(excluded == 0.0, "a group that was not requested contributes zero");
    check(log_.executeCalls == before, "the kernel is not even called for that group");

    const double included =
        context.getState(OpenMM::State::Energy, false, 1 << 3).getPotentialEnergy();
    check(included == 42.0, "the requested group does reach the kernel");
    check(log_.executeCalls == before + 1, "exactly one extra execute");
}

static void testUpdateParametersInContext(OpenMM::Platform& platform) {
    log_ = CallLog();
    OpenMM::System system;
    buildSystem(system, 4);
    LocalCWLDForce* force = buildForce(4);
    system.addForce(force);

    OpenMM::VerletIntegrator integrator(0.001);
    OpenMM::Context context(system, integrator, platform);
    context.setPositions(std::vector<OpenMM::Vec3>(4, OpenMM::Vec3(0, 0, 0)));

    force->setParticleParameters(0, -0.834, 1.0, -0.15, 1.0, 1.0, 1.0, 0.5, 1.0, 0);
    force->updateParametersInContext(context);
    check(log_.updateCalls == 1, "copyParametersToContext called once");
    check(log_.lastQbase0 == -0.834, "the new value reached the kernel");

    // Shape changes must be refused: device buffers were sized at initialize().
    force->addParticle(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4);
    checkThrows([&] { force->updateParametersInContext(context); },
                "changing the particle count is refused");
    check(log_.updateCalls == 1, "the refused update did not reach the kernel");
}

static void testExclusionTopologyIsFrozen(OpenMM::Platform& platform) {
    log_ = CallLog();
    OpenMM::System system;
    buildSystem(system, 4);
    LocalCWLDForce* force = buildForce(4);
    force->addExclusion(0, 1);
    system.addForce(force);

    OpenMM::VerletIntegrator integrator(0.001);
    OpenMM::Context context(system, integrator, platform);
    context.setPositions(std::vector<OpenMM::Vec3>(4, OpenMM::Vec3(0, 0, 0)));

    force->addExclusion(2, 3);
    checkThrows([&] { force->updateParametersInContext(context); },
                "changing the exclusion topology is refused");
    check(log_.updateCalls == 0, "the refused update did not reach the kernel");
}

static void testContextCreationValidation(OpenMM::Platform& platform) {
    // Particle count mismatch: the misalignment this prevents is silent.
    {
        OpenMM::System system;
        buildSystem(system, 4);
        system.addForce(buildForce(3));
        OpenMM::VerletIntegrator integrator(0.001);
        checkThrows([&] { OpenMM::Context context(system, integrator, platform); },
                    "particle-count mismatch is refused at Context creation");
    }
    // Box too small for the cutoff.
    {
        OpenMM::System system;
        buildSystem(system, 4, /*boxSide=*/1.0);   // rc defaults to 1.2 nm
        system.addForce(buildForce(4));
        OpenMM::VerletIntegrator integrator(0.001);
        checkThrows([&] { OpenMM::Context context(system, integrator, platform); },
                    "a box smaller than 2*rc is refused");
    }
    // Exclusion referring to a particle that does not exist.
    {
        OpenMM::System system;
        buildSystem(system, 4);
        LocalCWLDForce* force = buildForce(4);
        force->addExclusion(0, 9);
        system.addForce(force);
        OpenMM::VerletIntegrator integrator(0.001);
        checkThrows([&] { OpenMM::Context context(system, integrator, platform); },
                    "an out-of-range exclusion is refused at Context creation");
    }
}

int main() {
    try {
        OpenMM::Platform& reference = OpenMM::Platform::getPlatformByName("Reference");
        reference.registerKernelFactory(CalcLocalCWLDForceKernel::Name(), new FakeFactory());

        testInitializeAndExecute(reference);
        testForceGroupIsRespected(reference);
        testUpdateParametersInContext(reference);
        testExclusionTopologyIsFrozen(reference);
        testContextCreationValidation(reference);
    } catch (const std::exception& e) {
        std::printf("FAIL: unexpected exception escaped: %s\n", e.what());
        failures++;
    }
    if (failures == 0)
        std::printf("TestLocalCWLDForceImpl: all checks passed\n");
    else
        std::printf("TestLocalCWLDForceImpl: %d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
