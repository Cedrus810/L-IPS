/**
 * Does LocalCWLDForce go non-finite where CustomGBForce stays finite?
 *
 * ---------------------------------------------------------------------------
 * Why this test exists
 * ---------------------------------------------------------------------------
 * The benchmark's `new` arm died inside LocalEnergyMinimizer. The mechanism is
 * in DEC-005's 2026-09-04 addendum: CommonMinimizeKernel falls back to a
 * CPU/Reference Context, which we cannot provide (LCWLD-070 skipped). But that
 * fallback fires on exactly one condition:
 *
 *     evaluateGpu(); if (!(fabs(energy) < FLT_MAX)) energy = evaluateCpu();
 *
 * So the crash also says the `new` arm produced a non-finite (or > FLT_MAX)
 * energy at some line-search geometry. Two readings, with opposite fixes:
 *
 *   (i)  Both forces blow up on overlapping atoms; L-BFGS line searches
 *        routinely try such steps. We just lack the fallback -> write LCWLD-070.
 *   (ii) Our kernel returns Inf/NaN where CustomGBForce returns a large finite
 *        number -> a real kernel bug. Writing LCWLD-070 would then HIDE it:
 *        the fallback would succeed silently and recompute the same Inf on the
 *        CPU, with nothing printed.
 *
 * ---------------------------------------------------------------------------
 * Criterion, fixed before the result (DEC-005 section 6)
 * ---------------------------------------------------------------------------
 *     If ANY configuration makes CustomGBForce finite and LocalCWLDForce
 *     non-finite, it is (ii): fix the kernel first.
 *
 * Both directions are reported, not just that one. "LocalCWLD finite where
 * CustomGB is not" is also worth knowing -- it would mean the two disagree
 * about where the model stops being defined.
 *
 * ---------------------------------------------------------------------------
 * Force-group isolation is not optional here
 * ---------------------------------------------------------------------------
 * At r = 1e-4 nm the NonbondedForce's own r^-12 overflows on its own. Compared
 * on total energy, BOTH arms would be non-finite and the test would answer
 * nothing. So the comparison is on force group 31 only -- the dump already
 * isolates CustomGBForce there (dec005_dump_1aay.py), and this test puts
 * LocalCWLDForce in the same group and ASSERTS the gb side really is 31 rather
 * than trusting it.
 *
 * Zero Integrator.step(): static getState() calls only.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <set>
#include <utility>
#include <algorithm>
#include <string>
#include <vector>

#include "openmm/Context.h"
#include "openmm/CustomGBForce.h"
#include "openmm/LocalCWLDForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/State.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/Vec3.h"
#include "openmm/serialization/XmlSerializer.h"

#include "LocalCWLDArms.h"

using OpenMM::Vec3;

/** The group dec005_dump_1aay.py isolates the reference force into. */
static const int ISOLATED_GROUP = 31;

/** Separations to probe, nm. 0.35 is r_env; 1e-5 is well past any real step. */
static const double SEPARATIONS[] = {1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.35};

struct Probe {
    double energy = 0;
    double maxForce = 0;
    bool energyFinite = false;
    bool forcesFinite = false;
    bool ok() const { return energyFinite && forcesFinite; }
};

/** Evaluate group 31 only, and say whether everything came back finite. */
static Probe probe(OpenMM::Context& context, const std::vector<Vec3>& positions) {
    context.setPositions(positions);
    const OpenMM::State state = context.getState(
        OpenMM::State::Energy | OpenMM::State::Forces, false, 1 << ISOLATED_GROUP);
    Probe p;
    p.energy = state.getPotentialEnergy();
    // OpenMM's fp32 accumulator can also return a finite value too large for
    // float; that is what CommonMinimizeKernel actually tests for, so test the
    // same thing rather than std::isfinite alone.
    p.energyFinite = std::isfinite(p.energy) &&
                     std::fabs(p.energy) < (double) std::numeric_limits<float>::max();
    p.forcesFinite = true;
    for (const Vec3& f : state.getForces()) {
        const double m = std::sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
        if (!std::isfinite(m)) { p.forcesFinite = false; break; }
        if (m > p.maxForce) p.maxForce = m;
    }
    return p;
}

/** Put LocalCWLDForce in the isolated group; check CustomGBForce already is. */
static void isolate(OpenMM::System& system, bool expectGB) {
    bool found = false;
    for (int i = 0; i < system.getNumForces(); i++) {
        OpenMM::Force& f = system.getForce(i);
        if (dynamic_cast<LocalCWLDPlugin::LocalCWLDForce*>(&f) != nullptr) {
            f.setForceGroup(ISOLATED_GROUP);
            found = true;
        } else if (dynamic_cast<OpenMM::CustomGBForce*>(&f) != nullptr) {
            if (f.getForceGroup() != ISOLATED_GROUP)
                throw OpenMM::OpenMMException(
                    "dumped CustomGBForce is not in the isolated group; the "
                    "comparison below would include the NonbondedForce blow-up");
            found = true;
        } else if (f.getForceGroup() == ISOLATED_GROUP) {
            throw OpenMM::OpenMMException("another force shares the isolated group");
        }
    }
    if (!found)
        throw OpenMM::OpenMMException(expectGB ? "no CustomGBForce" : "no LocalCWLDForce");
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::printf("usage: %s <plugin dir> <dump prefix>\n", argv[0]);
        return 1;
    }
    try {
        const char* env = std::getenv("OPENMM_PLUGIN_DIR");
        OpenMM::Platform::loadPluginsFromDirectory(
            env ? std::string(env) : OpenMM::Platform::getDefaultPluginsDirectory());
        OpenMM::Platform::loadPluginsFromDirectory(argv[1]);
        OpenMM::Platform& cuda = OpenMM::Platform::getPlatformByName("CUDA");

        const std::string prefix = argv[2];
        OpenMM::System* source = deserializeFile<OpenMM::System>(prefix + "_system.xml");
        OpenMM::State* refState = deserializeFile<OpenMM::State>(prefix + "_state_mixed.xml");
        const std::vector<Vec3> base = refState->getPositions();

        // Pick a pair that is NOT excluded and not bonded: two oxygens from
        // different water molecules. An excluded pair would be skipped by both
        // forces and prove nothing about the r -> 0 limit.
        const OpenMM::CustomGBForce* gbSrc = nullptr;
        for (int i = 0; i < source->getNumForces(); i++)
            if (const OpenMM::CustomGBForce* g =
                    dynamic_cast<const OpenMM::CustomGBForce*>(&source->getForce(i)))
                gbSrc = g;
        if (gbSrc == nullptr)
            throw OpenMM::OpenMMException("no CustomGBForce in the dumped System");
        std::set<std::pair<int, int>> excluded;
        for (int i = 0; i < gbSrc->getNumExclusions(); i++) {
            int p1, p2;
            gbSrc->getExclusionParticles(i, p1, p2);
            excluded.insert({std::min(p1, p2), std::max(p1, p2)});
        }
        int atomA = -1, atomB = -1;
        const int n = source->getNumParticles();
        for (int i = n / 2; i < n && atomB < 0; i++) {
            const double m = source->getParticleMass(i);
            if (m < 15.9 || m > 16.1) continue;      // water oxygen
            if (atomA < 0) { atomA = i; continue; }
            if (excluded.count({atomA, i}) == 0) atomB = i;
        }
        if (atomB < 0)
            throw OpenMM::OpenMMException("could not find a non-excluded O-O pair");
        std::printf("pair: atoms %d and %d (masses %.3f, %.3f), base separation %.4f nm\n",
                    atomA, atomB, source->getParticleMass(atomA),
                    source->getParticleMass(atomB),
                    std::sqrt((base[atomA] - base[atomB]).dot(base[atomA] - base[atomB])));

        OpenMM::System* gbSys  = buildArm(*source, ARM_GB);
        OpenMM::System* newSys = buildArm(*source, ARM_NEW);
        isolate(*gbSys, true);
        isolate(*newSys, false);

        OpenMM::VerletIntegrator gbInt(0.001), newInt(0.001);
        OpenMM::Context gbCtx(*gbSys, gbInt, cuda, {{"Precision", "mixed"}});
        OpenMM::Context newCtx(*newSys, newInt, cuda, {{"Precision", "mixed"}});

        std::printf("\n%10s  %18s %18s  %10s %10s  %s\n", "r (nm)",
                    "E_gb", "E_new", "maxF_gb", "maxF_new", "verdict");
        int bugRows = 0, bothBad = 0, gbOnlyBad = 0;
        for (const double r : SEPARATIONS) {
            std::vector<Vec3> pos = base;
            pos[atomB] = pos[atomA] + Vec3(r, 0, 0);
            const Probe g = probe(gbCtx, pos);
            const Probe c = probe(newCtx, pos);
            const char* verdict;
            if (g.ok() && !c.ok())      { verdict = "*** (ii) CWLD non-finite, GB finite"; bugRows++; }
            else if (!g.ok() && !c.ok()){ verdict = "(i) both non-finite";                 bothBad++; }
            else if (!g.ok() && c.ok()) { verdict = "GB non-finite, CWLD finite";          gbOnlyBad++; }
            else                        { verdict = "both finite"; }
            std::printf("%10.1e  %18.6g %18.6g  %10.3g %10.3g  %s\n",
                        r, g.energy, c.energy, g.maxForce, c.maxForce, verdict);
            std::fflush(stdout);
        }

        std::printf("\n");
        if (bugRows > 0) {
            std::printf("VERDICT (ii): %d configuration(s) finite for CustomGBForce and "
                        "non-finite for LocalCWLDForce.\n"
                        "Fix the kernel BEFORE writing LCWLD-070 -- the reference "
                        "fallback would hide this.\n", bugRows);
            return 1;
        }
        std::printf("VERDICT (i): no configuration separates them "
                    "(%d both non-finite, %d GB-only non-finite).\n"
                    "The minimiser failure is the missing reference fallback, "
                    "not a short-range kernel bug.\n", bothBad, gbOnlyBad);
        return 0;
    }
    catch (const std::exception& e) {
        std::printf("FAILED: %s\n", e.what());
        return 1;
    }
}
