/**
 * LCWLD-040 API tests: defaults, round trip, rejection of illegal input, and
 * copy/clone behaviour. No Context is created and no kernel is exercised --
 * this ticket's Force is a parameter container and nothing more.
 */

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include "openmm/LocalCWLDForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/System.h"

using namespace LocalCWLDPlugin;
using OpenMM::OpenMMException;

static int failures = 0;

static void check(bool condition, const std::string& what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what.c_str());
        failures++;
    }
}

static void checkClose(double actual, double expected, const std::string& what) {
    if (!(std::fabs(actual - expected) <= 1e-12 * (1.0 + std::fabs(expected)))) {
        std::printf("FAIL: %s (got %.17g, want %.17g)\n", what.c_str(), actual, expected);
        failures++;
    }
}

/** Run `body`, expecting an OpenMMException. */
template <typename Body>
static void checkThrows(Body body, const std::string& what) {
    try {
        body();
    } catch (const OpenMMException&) {
        return;
    } catch (...) {
        std::printf("FAIL: %s (threw, but not OpenMMException)\n", what.c_str());
        failures++;
        return;
    }
    std::printf("FAIL: %s (did not throw)\n", what.c_str());
    failures++;
}

static void testDefaults() {
    LocalCWLDForce force;
    // Plan section 17.1. If any of these drift, every golden fixture built on
    // top of them silently changes meaning.
    checkClose(force.getEnvironmentCutoff(), 0.35, "default r_env");
    checkClose(force.getCutoffDistance(), 1.2, "default rc");
    check(force.getZMMOrder() == 2, "default zmm_order");
    checkClose(force.getRho0(), 13.5, "default rho0");
    checkClose(force.getKPolar(), 0.8, "default k_polar");
    checkClose(force.getChargeDeltaClamp(), 0.2, "default q_delta_clamp");
    check(!force.getUseQPenalty(), "q penalty off by default");
    checkClose(force.getQPenaltyStrength(), 180.0, "default q_penalty_strength");
    checkClose(force.getOne4PiEps0(), 138.935458, "default ONE_4PI_EPS0");
    check(force.getNonbondedMethod() == LocalCWLDForce::CutoffPeriodic, "default method");
    check(force.getNumParticles() == 0, "starts with no particles");
    check(force.getNumExclusions() == 0, "starts with no exclusions");
    check(force.usesPeriodicBoundaryConditions(), "usesPeriodicBoundaryConditions is true");
}

static void testParticleRoundTrip() {
    LocalCWLDForce force;
    check(force.addParticle(-0.834, 1.25, -0.15, 1.0, 1.0, 1.0, 0.5, 1.0, 7) == 0,
          "addParticle returns index 0");
    check(force.addParticle(0.417, 0.98, 0.075, 1.0, 0.0, 1.0, 0.0, 1.0, 7) == 1,
          "addParticle returns consecutive indices");
    check(force.getNumParticles() == 2, "two particles stored");

    double qbase, chargeMod, dpolar, isPolar, densSource, densSink, weight, phase;
    int residueId;
    force.getParticleParameters(0, qbase, chargeMod, dpolar, isPolar, densSource,
                                densSink, weight, phase, residueId);
    // Field-by-field, because a permuted parameter order is the single most
    // likely way this API breaks and the values are all plausible doubles.
    checkClose(qbase, -0.834, "qbase round trip");
    checkClose(chargeMod, 1.25, "chargeMod round trip");
    checkClose(dpolar, -0.15, "dpolar round trip");
    checkClose(isPolar, 1.0, "isPolar round trip");
    checkClose(densSource, 1.0, "densSource round trip");
    checkClose(densSink, 1.0, "densSink round trip");
    checkClose(weight, 0.5, "sourceClassWeight round trip");
    checkClose(phase, 1.0, "staticPhase round trip");
    check(residueId == 7, "residueId round trip");

    force.setParticleParameters(0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9);
    force.getParticleParameters(0, qbase, chargeMod, dpolar, isPolar, densSource,
                                densSink, weight, phase, residueId);
    checkClose(qbase, 1.0, "set/get qbase");
    checkClose(chargeMod, 2.0, "set/get chargeMod");
    checkClose(dpolar, 3.0, "set/get dpolar");
    checkClose(isPolar, 4.0, "set/get isPolar");
    checkClose(densSource, 5.0, "set/get densSource");
    checkClose(densSink, 6.0, "set/get densSink");
    checkClose(weight, 7.0, "set/get sourceClassWeight");
    checkClose(phase, 8.0, "set/get staticPhase");
    check(residueId == 9, "set/get residueId");
}

static void testParticleRejection() {
    LocalCWLDForce force;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    checkThrows([&] { force.addParticle(nan, 1, 0, 0, 0, 0, 0, 0, 0); }, "NaN qbase rejected");
    checkThrows([&] { force.addParticle(0, inf, 0, 0, 0, 0, 0, 0, 0); }, "inf chargeMod rejected");
    checkThrows([&] { force.addParticle(0, 1, nan, 0, 0, 0, 0, 0, 0); }, "NaN dpolar rejected");
    checkThrows([&] { force.addParticle(0, 1, 0, 0, 0, 0, 0, 0, -1); }, "negative residueId rejected");
    check(force.getNumParticles() == 0, "rejected particles are not stored");

    checkThrows([&] {
        double a, b, c, d, e, f, g, h;
        int r;
        force.getParticleParameters(0, a, b, c, d, e, f, g, h, r);
    }, "get on empty force rejected");

    force.addParticle(0, 1, 0, 0, 0, 0, 0, 0, 0);
    checkThrows([&] { force.setParticleParameters(1, 0, 1, 0, 0, 0, 0, 0, 0, 0); },
                "out-of-range setParticleParameters rejected");
    checkThrows([&] { force.setParticleParameters(0, 0, 1, 0, 0, 0, 0, 0, nan, 0); },
                "NaN staticPhase rejected on set");
}

static void testExclusions() {
    LocalCWLDForce force;
    for (int i = 0; i < 5; i++)
        force.addParticle(0, 1, 0, 0, 0, 0, 0, 0, i);

    check(force.addExclusion(3, 1) == 0, "addExclusion returns index 0");
    int a, b;
    force.getExclusionParticles(0, a, b);
    check(a == 1 && b == 3, "exclusion normalized to (min, max)");

    checkThrows([&] { force.addExclusion(2, 2); }, "self exclusion rejected");
    checkThrows([&] { force.addExclusion(1, 3); }, "duplicate exclusion rejected");
    checkThrows([&] { force.addExclusion(3, 1); }, "duplicate in reversed order rejected");
    checkThrows([&] { force.addExclusion(-1, 2); }, "negative exclusion index rejected");
    check(force.getNumExclusions() == 1, "rejected exclusions are not stored");

    force.addExclusion(0, 4);
    force.setExclusionParticles(1, 4, 2);
    force.getExclusionParticles(1, a, b);
    check(a == 2 && b == 4, "setExclusionParticles normalizes");
    checkThrows([&] { force.setExclusionParticles(1, 3, 1); },
                "setExclusionParticles rejects a duplicate of another entry");
    // Rewriting an entry to the value it already holds must stay legal: the
    // duplicate scan has to skip the entry being replaced.
    force.setExclusionParticles(1, 2, 4);
    force.getExclusionParticles(1, a, b);
    check(a == 2 && b == 4, "self-rewrite of an exclusion is allowed");
    checkThrows([&] { force.setExclusionParticles(7, 0, 1); },
                "out-of-range setExclusionParticles rejected");
}

static void testGlobalValidation() {
    LocalCWLDForce force;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    checkThrows([&] { force.setCutoffDistance(0.0); }, "zero rc rejected");
    checkThrows([&] { force.setCutoffDistance(-1.0); }, "negative rc rejected");
    checkThrows([&] { force.setEnvironmentCutoff(nan); }, "NaN r_env rejected");
    checkThrows([&] { force.setZMMOrder(0); }, "zmm_order 0 rejected");
    checkThrows([&] { force.setZMMOrder(4); }, "zmm_order 4 rejected");
    checkThrows([&] { force.setRho0(0.0); }, "zero rho0 rejected");
    checkThrows([&] { force.setChargeDeltaClamp(0.0); }, "zero q_delta_clamp rejected");
    checkThrows([&] { force.setQPenaltyStrength(-1.0); }, "negative penalty strength rejected");
    checkThrows([&] { force.setOne4PiEps0(0.0); }, "zero ONE_4PI_EPS0 rejected");
    checkThrows([&] { force.setNonbondedMethod(static_cast<LocalCWLDForce::NonbondedMethod>(1)); },
                "unsupported nonbonded method rejected");

    // r_env <= rc is a joint constraint, so it must be enforced from both sides
    // and must not trap the caller: widening rc first has to work.
    checkThrows([&] { force.setEnvironmentCutoff(2.0); }, "r_env above rc rejected");
    checkThrows([&] { force.setCutoffDistance(0.1); }, "rc below r_env rejected");
    force.setCutoffDistance(3.0);
    force.setEnvironmentCutoff(2.0);
    checkClose(force.getEnvironmentCutoff(), 2.0, "r_env settable after widening rc");

    // k_polar may be zero or negative (response off / inverted); only
    // non-finite is illegal.
    force.setKPolar(0.0);
    checkClose(force.getKPolar(), 0.0, "zero k_polar allowed");
    force.setKPolar(-0.5);
    checkClose(force.getKPolar(), -0.5, "negative k_polar allowed");
    checkThrows([&] { force.setKPolar(nan); }, "NaN k_polar rejected");

    force.setUseQPenalty(true);
    check(force.getUseQPenalty(), "q penalty togglable");
    force.setQPenaltyStrength(0.0);
    checkClose(force.getQPenaltyStrength(), 0.0, "zero penalty strength allowed");
}

static void testCopyBehaviour() {
    LocalCWLDForce force;
    force.setZMMOrder(3);
    force.setUseQPenalty(true);
    force.addParticle(-0.834, 1.25, -0.15, 1.0, 1.0, 1.0, 0.5, 1.0, 0);
    force.addParticle(0.417, 0.98, 0.075, 1.0, 0.0, 1.0, 0.0, 1.0, 0);
    force.addExclusion(0, 1);

    LocalCWLDForce copy(force);
    check(copy.getNumParticles() == 2, "copy keeps particles");
    check(copy.getNumExclusions() == 1, "copy keeps exclusions");
    check(copy.getZMMOrder() == 3, "copy keeps globals");
    check(copy.getUseQPenalty(), "copy keeps penalty flag");

    // The copy must own its storage: editing it must not reach back.
    copy.setParticleParameters(0, 99.0, 1, 0, 0, 0, 0, 0, 0, 0);
    double qbase, other;
    int residueId;
    force.getParticleParameters(0, qbase, other, other, other, other, other, other,
                                other, residueId);
    checkClose(qbase, -0.834, "copy is deep, original unchanged");
}

static void testNoImplYet() {
    // LCWLD-050 is not implemented. Adding this force to a System that builds a
    // Context must fail with a clear message rather than crash. Serialization
    // is LCWLD-060 and is also expected to be unavailable.
    LocalCWLDForce force;
    force.addParticle(0, 1, 0, 0, 0, 0, 0, 0, 0);
    OpenMM::System system;
    system.addParticle(1.0);
    system.addForce(new LocalCWLDForce(force));
    check(system.getNumForces() == 1, "force is addable to a System");
}

int main() {
    try {
        testDefaults();
        testParticleRoundTrip();
        testParticleRejection();
        testExclusions();
        testGlobalValidation();
        testCopyBehaviour();
        testNoImplYet();
    } catch (const std::exception& e) {
        std::printf("FAIL: unexpected exception escaped: %s\n", e.what());
        failures++;
    }
    if (failures == 0)
        std::printf("TestLocalCWLDForceAPI: all checks passed\n");
    else
        std::printf("TestLocalCWLDForceAPI: %d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
