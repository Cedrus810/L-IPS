/**
 * LCWLD-130: does the compact r_env list stay correct under dynamics?
 *
 * ---------------------------------------------------------------------------
 * Why a static test cannot answer this
 * ---------------------------------------------------------------------------
 * The DEC-005 acceptance harness evaluates forces ONCE, at fixed coordinates.
 * It therefore says nothing about the decision this plugin makes on every
 * single step: whether to rebuild the compact list. Both of the worst bugs in
 * this plugin so far were invisible to a static evaluation --
 *
 *   - `getStepsSinceReorder() == 0` as the rebuild trigger: that counter is
 *     maintained by the integrator, so it stayed 0 through every force-only
 *     evaluation and the list was rebuilt every step. Correct, just slow.
 *   - a stale maxTiles, which silently skipped the entire neighbour-list loop.
 *
 * and the rebuild decision has just moved onto the device (design doc section
 * 9), which is exactly the kind of change that a static test waves through.
 *
 * ---------------------------------------------------------------------------
 * The test
 * ---------------------------------------------------------------------------
 * Run dynamics with LocalCWLDForce. Take the coordinates. Evaluate the SAME
 * coordinates with the production CustomGBForce -- the DEC-005 oracle, which
 * has no compact list of any kind. The two must agree.
 *
 * The reference deliberately is NOT a second LocalCWLDForce Context. That was
 * the first version and it could not be trusted: LOCALCWLD_NEVER_REBUILD is a
 * process-wide environment variable, so making the list stale to check that the
 * test notices made the REFERENCE stale in the same way, and the two agreed to
 * 8e-2. The control was contaminated by the mutation it was supposed to detect.
 *
 * ⚠ The comparison is on FORCE GROUP 31 ONLY, which the dump already isolates
 * the CustomGBForce into. Comparing total forces looks like it should work --
 * the arms differ in nothing but this one force, so everything else ought to
 * cancel -- and it does not. The two Contexts order atoms differently
 * internally, so their NonbondedForce pair sums round differently in fp32; on
 * forces of order 10^3 kJ/mol/nm that leaves 0.1-2 per atom of pure ordering
 * noise. Measured on the first version of this test: 117 atoms over 1.0
 * kJ/mol/nm with a genuinely fresh list, and 102 with a deliberately stale one.
 * The signal was entirely buried.
 *
 * A list that goes stale -- never rebuilt, or rebuilt on the wrong steps --
 * fails this loudly: after a few hundred steps at 300 K, atoms have moved far
 * enough that pairs which drifted inside r_env are missing from the stale list,
 * dens is wrong for those atoms, and Q with it. That error is of the same order
 * as the entire modelled effect (coordinating-atom dF of 106-254 kJ/mol/nm),
 * not a rounding difference.
 *
 * ⚠ This test calls Integrator.step(). It is a correctness test, not a
 * benchmark, and it is registered with ctest deliberately: the property it
 * checks has no other guard.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "openmm/Context.h"
#include "openmm/CustomGBForce.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/LocalCWLDForce.h"
#include "openmm/LocalEnergyMinimizer.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/State.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include "openmm/serialization/XmlSerializer.h"

#include "LocalCWLDArms.h"

/** The group dec005_dump_1aay.py isolates the reference force into. */
static const int ISOLATED_GROUP = 31;

/** Put LocalCWLDForce in the isolated group; verify CustomGBForce already is. */
static void isolate(OpenMM::System& system) {
    for (int i = 0; i < system.getNumForces(); i++) {
        OpenMM::Force& f = system.getForce(i);
        if (dynamic_cast<LocalCWLDPlugin::LocalCWLDForce*>(&f) != nullptr)
            f.setForceGroup(ISOLATED_GROUP);
        else if (dynamic_cast<OpenMM::CustomGBForce*>(&f) != nullptr) {
            if (f.getForceGroup() != ISOLATED_GROUP)
                throw OpenMM::OpenMMException(
                    "dumped CustomGBForce is not in the isolated group");
        } else if (f.getForceGroup() == ISOLATED_GROUP)
            throw OpenMM::OpenMMException("another force shares the isolated group");
    }
}

using OpenMM::Vec3;

static const double TEMPERATURE = 300.0;
static const double FRICTION = 1.0;
static const double TIMESTEP = 0.002;

/**
 * DEC-005's per-particle force tolerance. Reused deliberately: the question
 * here is the same one -- are these forces the right forces -- so the bar
 * should be the same. It is 8.2x the fp32 noise floor and 1/106 of the weakest
 * real signal.
 */
static const double FORCE_ATOL = 1.0;

/**
 * The gate is p99 of |dF| against FORCE_ATOL -- both taken from DEC-005 rather
 * than invented here, so nothing is being fitted to the result.
 *
 * It cannot be the count of atoms over atol, because that count has a floor
 * that has nothing to do with this test: a pair sitting within fp32 epsilon of
 * the 1.2 nm cutoff is counted by one evaluation and not the other, worth about
 * 29 kJ/mol/nm on the two atoms involved. With ~11e6 pairs, the shell that thin
 * holds a few tens of pairs, so ~10^2 atoms land over atol no matter how
 * correct the list is. Measured floor: 120-140 atoms. DEC-005 already excludes
 * this population from its statistical threshold and requires the count be
 * reported instead; both are done below.
 *
 * The separation is not marginal. Measured at 400 steps:
 *
 *              p50        p99      p99.9   over atol
 *   correct   2.4e-4    3.5e-3      1.6       120
 *   stale     5.4e+0    1.9e+1     3.1e+1    31750
 *
 * p99 clears the bar by 285x when the list is right and misses it by 19x when
 * it is not, and p50 differs by four orders of magnitude. The stale figures are
 * from LOCALCWLD_NEVER_REBUILD=1 -- a probe that had to be re-implemented on
 * the device first, because moving the rebuild decision off the host had
 * quietly deleted it, and a mutation run against the dead version "passed"
 * while changing nothing at all.
 */

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::printf("usage: %s <plugin dir> <dump prefix>\n", argv[0]);
        std::printf("skipped: no dump provided\n");
        return 0;
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

        // Minimise on the CustomGBForce arm. A System containing
        // LocalCWLDForce cannot be minimised at all (no reference kernel for
        // the minimiser's CPU overflow fallback -- DEC-005 addendum), and
        // starting dynamics from unrelaxed coordinates would blow up for
        // reasons that have nothing to do with what is being tested.
        std::vector<Vec3> start;
        {
            OpenMM::System* gbSys = buildArm(*source, ARM_GB);
            OpenMM::LangevinMiddleIntegrator integrator(TEMPERATURE, FRICTION, TIMESTEP);
            OpenMM::Context context(*gbSys, integrator, cuda, {{"Precision", "mixed"}});
            context.setPositions(refState->getPositions());
            OpenMM::LocalEnergyMinimizer::minimize(context, 10.0, 200);
            start = context.getState(OpenMM::State::Positions).getPositions();
            delete gbSys;
        }

        // Long enough that atoms cross the 0.1 nm list padding many times over.
        // The platform rebuilds its own list roughly every 4 steps here, so 400
        // steps is about a hundred rebuild decisions.
        const int steps = (argc > 3) ? std::atoi(argv[3]) : 400;
        OpenMM::System* sys = buildArm(*source, ARM_NEW);
        isolate(*sys);
        OpenMM::LangevinMiddleIntegrator integrator(TEMPERATURE, FRICTION, TIMESTEP);
        OpenMM::Context context(*sys, integrator, cuda, {{"Precision", "mixed"}});
        context.setPositions(start);
        context.setVelocitiesToTemperature(TEMPERATURE, 12345);
        integrator.step(steps);
        const OpenMM::State evolved = context.getState(
            OpenMM::State::Positions | OpenMM::State::Forces, false, 1 << ISOLATED_GROUP);
        const std::vector<Vec3> pos = evolved.getPositions();
        const std::vector<Vec3> ran = evolved.getForces();

        // The oracle: production CustomGBForce at the same coordinates.
        OpenMM::System* fresh = buildArm(*source, ARM_GB);
        isolate(*fresh);
        OpenMM::LangevinMiddleIntegrator freshIntegrator(TEMPERATURE, FRICTION, TIMESTEP);
        OpenMM::Context freshContext(*fresh, freshIntegrator, cuda, {{"Precision", "mixed"}});
        freshContext.setPositions(pos);
        const std::vector<Vec3> reference =
            freshContext.getState(OpenMM::State::Forces, false,
                                  1 << ISOLATED_GROUP).getForces();

        double worst = 0, sum2 = 0;
        int worstAtom = -1, over = 0;
        for (size_t i = 0; i < ran.size(); i++) {
            const Vec3 d = ran[i] - reference[i];
            const double m = std::sqrt(d.dot(d));
            sum2 += m * m;
            if (m > FORCE_ATOL) over++;
            if (m > worst) { worst = m; worstAtom = (int) i; }
        }
        std::vector<double> mags;
        mags.reserve(ran.size());
        for (size_t i = 0; i < ran.size(); i++) {
            const Vec3 d = ran[i] - reference[i];
            mags.push_back(std::sqrt(d.dot(d)));
        }
        std::sort(mags.begin(), mags.end());
        const size_t n = mags.size();
        std::printf("after %d steps: %zu atoms, |dF| rms %.4e  max %.4e (atom %d)\n",
                    steps, ran.size(), std::sqrt(sum2 / ran.size()), worst, worstAtom);
        std::printf("  |dF| p50 %.3e  p99 %.3e  p99.9 %.3e\n",
                    mags[n/2], mags[(size_t)(0.99*n)], mags[(size_t)(0.999*n)]);
        std::printf("  atoms over atol %.1f kJ/mol/nm: %d (cutoff-boundary floor is "
                    "~120; reported, not gated on)\n", FORCE_ATOL, over);

        delete source; delete refState; delete sys; delete fresh;

        const double p99 = mags[(size_t)(0.99 * n)];
        if (!(p99 < FORCE_ATOL)) {
            std::printf("FAILED: p99 of |dF| is %.4e, over the DEC-005 per-particle\n"
                        "        tolerance of %.1f kJ/mol/nm. The bulk of the system\n"
                        "        disagrees with CustomGBForce, which is what a stale\n"
                        "        compact r_env list looks like.\n", p99, FORCE_ATOL);
            return 1;
        }
        std::printf("all checks passed\n");
        return 0;
    }
    catch (const std::exception& e) {
        std::printf("FAILED: %s\n", e.what());
        return 1;
    }
}
