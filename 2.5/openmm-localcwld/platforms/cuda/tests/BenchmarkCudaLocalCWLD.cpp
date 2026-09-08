/**
 * LCWLD-150 performance benchmark: does the kernel actually make ns/day?
 *
 * ⚠ THIS RUNS MD. It calls Integrator.step(), so it is outside the
 * zero-integration scope the assistant works under -- it is written here but
 * must be launched by the user. It is deliberately NOT registered with ctest.
 *
 * ---------------------------------------------------------------------------
 * Why all three arms live in one binary
 * ---------------------------------------------------------------------------
 * Every performance number in this project that turned out wrong was wrong for
 * the same reason: a ratio taken across two different harnesses, so the ratio
 * carried the harness difference with it. 32%/1.5x, ~90%/10x, s>=92%, and the
 * fixed-overhead model all failed that way.
 *
 * So: same binary, same integrator, same precision, same starting coordinates,
 * same warmup and measure counts for every arm. Do NOT compare a number printed
 * here against the Python `force_cost` numbers -- that comparison is exactly
 * the mistake this file exists to avoid. `s` is re-measured here too, for the
 * same reason, rather than reusing the 0.846 measured in Python.
 *
 * Protocol copied from `lips.engine.v26.profile_system_speed`
 * (`src/lips/engine/v26.py`), so the shape of the measurement matches what the
 * project already reports:
 *     LangevinMiddleIntegrator(300 K, 1/ps, 0.002 ps)
 *     minimizeEnergy(tolerance = 10, maxIterations = 200)
 *     setVelocitiesToTemperature(300 K)
 *     warmup 1000 steps, measure 4000 steps
 *
 * with one deliberate deviation: the minimisation runs ONCE, on the `gb` arm,
 * and every arm starts from its output. Minimising per arm -- which is what
 * this file did first -- has the arms start from different coordinates, which
 * is the same class of mistake as comparing across harnesses. It is also the
 * only form that works at all: see the comment at the minimisation site for
 * why a System containing LocalCWLDForce cannot be minimised today.
 *
 * ---------------------------------------------------------------------------
 * The three arms
 * ---------------------------------------------------------------------------
 *   gb      the dumped System unchanged -- production CustomGBForce
 *   new     CustomGBForce removed, LocalCWLDForce added with the same
 *           per-particle parameters (read BY NAME) and the same exclusions
 *   none    CustomGBForce removed, nothing added -- gives s, the share of step
 *           time the force is responsible for
 *
 * From the three:
 *     s   = 1 - cost(none)/cost(gb)          cost == 1/ns_per_day
 *     f   = (cost(new) - cost(none)) / (cost(gb) - cost(none))
 *     e2e = ns(new) / ns(gb)
 * and the Amdahl consistency check `e2e == 1/((1-s) + s*f)` must hold; if it
 * does not, the measurement is inconsistent and none of it should be quoted.
 *
 * Design projection to compare against (docs/reports/LCWLD-090-100-design.md):
 * f <= 0.419, end-to-end >= 1.97x. DEC-005 section 5 calls below 1.5x a
 * failure to meet target.
 *
 * ---------------------------------------------------------------------------
 * Two extra numbers, and why
 * ---------------------------------------------------------------------------
 * Run with LOCALCWLD_DEBUG=1 to also print, per neighbour-list rebuild, the
 * actual r_env pair count and the rebuild interval. If f comes out away from
 * 0.419 these two say immediately whether the cause is (a) the compact list
 * being bigger than the volume ratio predicted -- it already was, by more than
 * 2x, which is why the list needed a grow-and-retry -- or (b) rebuilds
 * happening more often than assumed, which would break the amortisation the
 * whole 2.36x design rests on. Without them the post-mortem is another round of
 * "two models both explain the data".
 *
 * Usage (user runs this):
 *     BenchmarkCudaLocalCWLD <plugin dir> <dump prefix> [warmup] [measure]
 */

#include <chrono>
#include <typeinfo>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "openmm/Context.h"
#include "openmm/CustomGBForce.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "openmm/LocalEnergyMinimizer.h"
#include "openmm/LocalCWLDForce.h"
#include "LocalCWLDArms.h"
#include "openmm/LocalCWLDVersion.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/State.h"
#include "openmm/System.h"
#include "openmm/serialization/XmlSerializer.h"

using namespace LocalCWLDPlugin;
using OpenMM::Vec3;

// Protocol constants, mirroring v26.py so the measurement shape matches the
// project's existing numbers. Overridable only to make a quick smoke run cheap;
// any quoted result must use the defaults.
static const double TEMPERATURE = 300.0;        // K
static const double FRICTION = 1.0;             // 1/ps
static const double TIMESTEP = 0.002;           // ps
static const int DEFAULT_WARMUP = 1000;
static const int DEFAULT_MEASURE = 4000;
/** Rounds of the three arms, interleaved. One round cannot show contention. */
static const int DEFAULT_REPS = 3;
/**
 * Spread of the RATIOS (f, end-to-end) above which no verdict is reported.
 * Not the spread of the rates: those drift with clocks and say nothing about
 * the ratio, which is what is being measured. See the gate for the numbers.
 */
static const double SPREAD_LIMIT = 0.03;

struct Result {
    double nsPerDay = 0;
    double seconds = 0;
};

/**
 * One arm, kept alive for the whole run so the arms can be interleaved.
 *
 * Serial arms (all of arm 1, then all of arm 2, ...) were the original shape
 * and it is not safe on a shared GPU. A run on 2026-09-04 came back with
 * gb = 24.64 ns/day against 59.09/59.21/59.43/59.79 in four other runs, while
 * that same run's `new` and `none` arms were both in their normal range: one
 * arm, and only one, had been starved by another process. Every ratio derived
 * from it (s = 0.938, f = 0.222, e2e = 3.70x) was an artifact, and nothing in
 * the output said so.
 *
 * Interleaving does not make contention go away -- it makes it land on all
 * arms instead of one, and it makes the per-arm spread visible so the run can
 * be thrown out on evidence rather than on someone noticing an odd number.
 */
struct ArmCtx {
    const char* label;
    OpenMM::System* system = nullptr;
    OpenMM::LangevinMiddleIntegrator* integrator = nullptr;
    OpenMM::Context* context = nullptr;
    std::vector<double> samples;          // ns/day, one per round

    double median() const {
        std::vector<double> v = samples;
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        return (n % 2) ? v[n/2] : 0.5*(v[n/2 - 1] + v[n/2]);
    }
    /** (max - min) / median, the number that says whether to trust the run. */
    double spread() const {
        const auto mm = std::minmax_element(samples.begin(), samples.end());
        return (*mm.second - *mm.first) / median();
    }
};

/** Time `measure` steps of one already-warmed arm and record the rate. */
static void timeRound(ArmCtx& arm, int measure) {
    arm.context->getState(0);                 // drain before starting the clock
    const auto t0 = std::chrono::steady_clock::now();
    arm.integrator->step(measure);
    arm.context->getState(0);                 // drain before stopping it
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const double nsPerDay = (measure * TIMESTEP * 1e-3) / seconds * 86400.0;
    arm.samples.push_back(nsPerDay);
    std::printf("[timing] %-22s %6.1f s  ->  %7.2f ns/day\n",
                arm.label, seconds, nsPerDay);
    std::fflush(stdout);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::printf("usage: %s <plugin dir> <dump prefix> [warmup] [measure] [reps]\n",
                    argv[0]);
        return 1;
    }
    const int warmup = (argc > 3) ? std::atoi(argv[3]) : DEFAULT_WARMUP;
    const int measure = (argc > 4) ? std::atoi(argv[4]) : DEFAULT_MEASURE;
    const int reps = (argc > 5) ? std::atoi(argv[5]) : DEFAULT_REPS;
    if (warmup != DEFAULT_WARMUP || measure != DEFAULT_MEASURE)
        std::printf("⚠ non-default step counts (%d/%d): fine for a smoke run, but do\n"
                    "  not quote the result -- the project's numbers use %d/%d.\n",
                    warmup, measure, DEFAULT_WARMUP, DEFAULT_MEASURE);
    try {
        const char* env = std::getenv("OPENMM_PLUGIN_DIR");
        OpenMM::Platform::loadPluginsFromDirectory(
            env ? std::string(env) : OpenMM::Platform::getDefaultPluginsDirectory());
        // Report the plugin load. The first run died with "Platform ... does
        // not support all required kernels" on the `new` arm and the log could
        // not say whether the plugin had failed to load or the factory had
        // failed to register -- two very different problems.
        const std::vector<std::string> ours =
            OpenMM::Platform::loadPluginsFromDirectory(argv[1]);
        bool foundOurs = false;
        for (const std::string& name : ours)
            if (name.find("LocalCWLDCUDA") != std::string::npos)
                foundOurs = true;
        std::printf("plugins: %zu from %s, LocalCWLDCUDA %s\n",
                    ours.size(), argv[1], foundOurs ? "LOADED" : "*** NOT LOADED ***");
        for (const std::string& e : OpenMM::Platform::getPluginLoadFailures())
            if (e.find("LocalCWLD") != std::string::npos)
                std::printf("  load failure: %s\n", e.c_str());

        OpenMM::Platform& cuda = OpenMM::Platform::getPlatformByName("CUDA");
        std::vector<std::string> needed(1, "CalcLocalCWLDForce");
        std::printf("plugins: CUDA supports CalcLocalCWLDForce = %s\n",
                    cuda.supportsKernels(needed) ? "yes" : "*** NO ***");
        std::fflush(stdout);
        std::printf("LocalCWLD %s, OpenMM %s\n",
                    LocalCWLDVersion::getVersion().c_str(),
                    OpenMM::Platform::getOpenMMVersion().c_str());

        const std::string prefix = argv[2];
        OpenMM::System* source = deserializeFile<OpenMM::System>(prefix + "_system.xml");
        OpenMM::State* refState = deserializeFile<OpenMM::State>(
            prefix + "_state_mixed.xml");
        const std::vector<Vec3> positions = refState->getPositions();
        std::printf("system: %d particles\n", source->getNumParticles());

        // Minimise ONCE, on the `gb` arm, and start every arm from the result.
        //
        // Two reasons, and the second one is the bug that forced this.
        //
        // (a) Per-arm minimisation made the arms start from DIFFERENT
        //     coordinates, so their ns/day were not comparable. That was wrong
        //     from the start and nobody noticed because nothing failed.
        // (b) LocalEnergyMinimizer cannot run on a System containing
        //     LocalCWLDForce. When the fp32 energy overflows -- at the starting
        //     point or mid line-search -- CommonMinimizeKernel::evaluateCpu()
        //     builds a second Context for the SAME System on the CPU/Reference
        //     platform (CommonMinimizeKernel.cpp:572-587). We have no reference
        //     kernel (LCWLD-070 was skipped), so that Context throws
        //     "Specified a Platform for a Context which does not support all
        //     required kernels" -- naming neither the kernel nor the platform.
        //     See docs/decisions/DEC-005 addendum.
        std::vector<Vec3> startPositions;
        {
            std::printf("\n[minim ] once, on the gb arm; all arms start here\n");
            std::fflush(stdout);
            OpenMM::System* sys = buildArm(*source, ARM_GB);
            OpenMM::LangevinMiddleIntegrator integrator(TEMPERATURE, FRICTION, TIMESTEP);
            OpenMM::Context context(*sys, integrator, cuda, {{"Precision", "mixed"}});
            context.setPositions(positions);
            std::printf("[minim ] E before = %.4f kJ/mol\n",
                        context.getState(OpenMM::State::Energy).getPotentialEnergy());
            std::fflush(stdout);
            OpenMM::LocalEnergyMinimizer::minimize(context, 10.0, 200);
            const OpenMM::State st = context.getState(OpenMM::State::Positions |
                                                      OpenMM::State::Energy);
            startPositions = st.getPositions();
            std::printf("[minim ] E after  = %.4f kJ/mol\n", st.getPotentialEnergy());
            std::fflush(stdout);
            delete sys;
        }

        // Build and time one arm at a time, and say which one is being built.
        // The first run failed with "Platform ... does not support all required
        // kernels" after the `new` arm's output, and the log could not say
        // whether that was the `new` arm's own Context or the next arm's --
        // three arms constructed up front made the failure ambiguous.
        const int NARMS = 4;
        ArmCtx arms[NARMS];
        arms[0].label = "gb   (CustomGBForce)";
        arms[1].label = "new  (LocalCWLDForce)";
        arms[2].label = "none (force removed)";
        arms[3].label = "bare (also no NB)";
        const Arm kinds[NARMS] = {ARM_GB, ARM_NEW, ARM_NONE, ARM_BARE};
        for (int a = 0; a < NARMS; a++) {
            ArmCtx& arm = arms[a];
            arm.system = buildArm(*source, kinds[a]);
            std::printf("[build ] %s: %d forces ->", arm.label, arm.system->getNumForces());
            for (int i = 0; i < arm.system->getNumForces(); i++)
                std::printf(" %s", typeid(arm.system->getForce(i)).name());
            std::printf("\n");
            std::fflush(stdout);
            arm.integrator = new OpenMM::LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP);
            arm.context = new OpenMM::Context(*arm.system, *arm.integrator, cuda,
                                              {{"Precision", "mixed"}});
            arm.context->setPositions(startPositions);
            arm.context->setVelocitiesToTemperature(TEMPERATURE);
            arm.integrator->step(warmup);     // absorbs kernel JIT and list build
            arm.context->getState(0);
            std::printf("[warmup] %s: %d steps\n", arm.label, warmup);
            std::fflush(stdout);
        }

        // Alternate the arm order every round.
        //
        // Clocks fall while a round runs -- this card cannot be locked (no
        // permission) and it throttles hard: five rounds took gb from 57.95 to
        // 46.96 ns/day. Inside a fixed order gb is always sampled first, at the
        // highest clock of the round, and new always later at a lower one, so
        // the drift lands entirely in the ratio. Reversing every other round
        // puts each arm early as often as late, and the linear part of the
        // drift cancels in the median.
        //
        // Locking clocks would be better and is not available:
        //   nvidia-smi -lgc -> "The current user does not have permission to
        //   change clocks for GPU 00000000:AF:00.0"
        for (int round = 0; round < reps; round++) {
            const bool reversed = (round % 2) == 1;
            std::printf("\n--- round %d/%d (%d steps per arm, %s) ---\n",
                        round + 1, reps, measure,
                        reversed ? "reversed order" : "forward order");
            std::fflush(stdout);
            for (int i = 0; i < NARMS; i++)
                timeRound(arms[reversed ? NARMS - 1 - i : i], measure);
        }
        if (reps % 2 == 1)
            std::printf("\n⚠ odd number of rounds: the forward/reverse pairing that\n"
                        "  cancels clock drift is incomplete. Use an even count.\n");

        std::printf("\n%-22s %10s %10s  %s\n", "arm", "median", "spread", "samples");
        double worstRateSpread = 0;
        for (int a = 0; a < NARMS; a++) {
            std::printf("%-22s %10.2f %9.1f%%  ", arms[a].label,
                        arms[a].median(), 100.0 * arms[a].spread());
            for (const double v : arms[a].samples) std::printf(" %.2f", v);
            std::printf("\n");
            worstRateSpread = std::max(worstRateSpread, arms[a].spread());
        }

        // Absolute rates drift; the RATIOS are what this benchmark measures,
        // and they do not. Measured on an idle 2080 Ti: rates fell 12.7%
        // monotonically across three rounds (thermal -- SM clock 1350 against a
        // 2175 MHz ceiling by the end) while f moved 1.0% and end-to-end moved
        // 0.23%. That is the whole point of interleaving: inside one round the
        // three arms are sampled at nearly the same clock, so the ratio is
        // clean even while throughput is not.
        //
        // The first version of this gate looked at the rate spread and refused
        // a verdict on a run whose ratios were stable to a quarter of a
        // percent. It was measuring the wrong quantity.
        std::vector<double> roundF, roundE2E, roundS;
        std::printf("\n%6s %9s %9s %9s   %8s %8s %8s\n",
                    "round", "gb", "new", "none", "s", "f", "e2e");
        for (int r = 0; r < reps; r++) {
            const double cg = 1.0/arms[0].samples[r], cn = 1.0/arms[1].samples[r],
                         c0 = 1.0/arms[2].samples[r];
            roundS.push_back(1.0 - c0/cg);
            roundF.push_back((cn - c0) / (cg - c0));
            roundE2E.push_back(arms[1].samples[r] / arms[0].samples[r]);
            std::printf("%6d %9.2f %9.2f %9.2f   %8.4f %8.4f %8.4f\n", r + 1,
                        arms[0].samples[r], arms[1].samples[r], arms[2].samples[r],
                        roundS.back(), roundF.back(), roundE2E.back());
        }
        auto med = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            const size_t n = v.size();
            return (n % 2) ? v[n/2] : 0.5*(v[n/2 - 1] + v[n/2]);
        };
        auto spreadOf = [&](const std::vector<double>& v) {
            const auto mm = std::minmax_element(v.begin(), v.end());
            return (*mm.second - *mm.first) / med(v);
        };
        const double ratioSpread = std::max(spreadOf(roundF), spreadOf(roundE2E));
        std::printf("%6s %9s %9s %9s   %8s %7.2f%% %7.2f%%\n", "spread", "", "", "",
                    "", 100.0*spreadOf(roundF), 100.0*spreadOf(roundE2E));

        // Same direction in every arm means the machine slowed down uniformly
        // (thermal). Contention hits one arm and leaves the others alone --
        // that is what a 2026-09-04 run looked like: gb at 24.64 against
        // 59.09/59.21/59.43/59.79 elsewhere, with `new` and `none` normal.
        // Only the two heavy arms. `none` runs for under two seconds a round,
        // where measurement noise swamps any drift, and it was that noise that
        // made a plainly thermal run report "suspect another process".
        // A step counts as a direction change only if it clears the noise
        // floor. Without that, LCWLD-161's run B -- `new` at
        // 178.78/178.83/177.17/177.50, a 0.9% wiggle -- flipped this to "NOT
        // monotonic" and the gate blamed a process that was not there.
        // ponytail: fixed 2%; make it per-arm if an arm ever gets noisier.
        const double NOISE = 0.02;
        bool allMonotonic = true;
        int direction = 0;
        for (int a = 0; a < 2 && allMonotonic; a++) {
            for (int r = 1; r < reps; r++) {
                const double rel = (arms[a].samples[r] - arms[a].samples[r-1]) /
                                   arms[a].median();
                if (std::fabs(rel) < NOISE) continue;
                const int d = (rel < 0) ? -1 : 1;
                if (direction == 0) direction = d;
                else if (d != direction) { allMonotonic = false; break; }
            }
        }

        // Interleaving only immunises the ratio while the arms drift TOGETHER.
        // Once one arm gets fast enough to stop heating the card it stops
        // drifting, the heavy arm keeps throttling, and the ratio inherits the
        // difference. LCWLD-161: gb fell 18.9% over four rounds while `new`
        // moved 0.9%, so e2e walked 3.085 -> 3.643 on a demonstrably idle GPU.
        // More rounds cannot fix that -- the drift is systematic, not noise.
        auto driftOf = [&](const ArmCtx& x) {
            return (x.samples[0] - x.samples[reps-1]) / x.median();
        };
        const bool armsDecoupled =
            reps >= 2 && allMonotonic &&
            std::fabs(driftOf(arms[0]) - driftOf(arms[1])) > 0.05;

        std::printf("  rate spread %.1f%% (%s); ratio spread %.2f%%\n",
                    100.0*worstRateSpread,
                    reps < 2 ? "one round, cannot tell"
                             : (!allMonotonic
                                    ? "NOT monotonic -> suspect another process on the GPU"
                                    : armsDecoupled
                                          ? "monotonic but the arms drift at different rates"
                                          : "monotonic in every arm -> clock drift, not contention"),
                    100.0*ratioSpread);
        if (armsDecoupled)
            std::printf("    gb drifts %.1f%% over %d rounds, new drifts %.1f%%. "
                        "Interleaving\n    cancels drift only while the arms drift "
                        "together; they no longer do.\n",
                        100.0*driftOf(arms[0]), reps, 100.0*driftOf(arms[1]));

        const Result gb = {med(std::vector<double>(arms[0].samples)), 0},
                     nw = {med(std::vector<double>(arms[1].samples)), 0},
                     nn = {med(std::vector<double>(arms[2].samples)), 0};

        // Costs, not rates: cost == 1/ns_per_day, and costs are what add up.
        const double cGB = 1.0/gb.nsPerDay, cNew = 1.0/nw.nsPerDay, cNone = 1.0/nn.nsPerDay;
        const double s = 1.0 - cNone/cGB;
        const double f = (cNew - cNone) / (cGB - cNone);
        const double e2e = nw.nsPerDay / gb.nsPerDay;
        const double amdahl = 1.0 / ((1.0 - s) + s*f);

        std::printf("\n================ LCWLD-150 result ================\n");
        std::printf("  ns/day    gb %8.2f   new %8.2f   none %8.2f\n",
                    gb.nsPerDay, nw.nsPerDay, nn.nsPerDay);
        std::printf("  s (CustomGBForce share of step time) = %.4f\n", s);
        std::printf("  f (new kernel cost / CustomGBForce)  = %.4f   [design <= 0.419]\n", f);
        std::printf("  end-to-end speedup                   = %.3fx  [design >= 1.97x]\n", e2e);
        std::printf("  Amdahl check 1/((1-s)+s*f)           = %.3fx  (must match e2e)\n",
                    amdahl);
        const double mismatch = std::fabs(amdahl - e2e) / e2e;
        std::printf("  consistency: %.1f%% %s\n", mismatch*100,
                    mismatch < 0.05 ? "ok"
                                    : "<<<< INCONSISTENT -- do not quote any of these");
        // The 1.5x line is decided on the median of interleaved rounds, and
        // only when the rounds agree. A single sample cannot separate "we are
        // at 1.52x" from "another process had the GPU for one arm" -- and the
        // margin over the line has been as small as 1.2%, well inside the 1.5%
        // run-to-run scatter seen in the production logs. Refusing to print a
        // verdict is the honest output here; DEC-005 section 6 fixes the
        // threshold before the result, which only means anything if the
        // measurement it is applied to is sound.
        if (reps < 2) {
            std::printf("  DEC-005 section 5: NO VERDICT -- one round cannot show scatter.\n");
        } else if (ratioSpread > SPREAD_LIMIT) {
            std::printf("  DEC-005 section 5: NO VERDICT -- ratio spread %.2f%% exceeds "
                        "%.2f%%.\n%s",
                        100.0*ratioSpread, 100.0*SPREAD_LIMIT,
                        armsDecoupled
                            ? "    The arms no longer drift together, so interleaving cannot\n"
                              "    cancel the clock drift. Adding rounds will NOT help. Report a\n"
                              "    range across rounds, or lock the clocks.\n"
                            : "    Rounds disagree about the ratio itself, which interleaving\n"
                              "    should have removed. Suspect another process on the GPU.\n");
        } else {
            const double margin = (e2e - 1.5) / 1.5;
            std::printf("  DEC-005 section 5: below 1.5x is a failure to meet target -> %s\n"
                        "    median of %d interleaved rounds; ratio spread %.2f%%, "
                        "margin over the line %+.2f%%\n",
                        e2e >= 1.5 ? "PASS" : "BELOW TARGET", reps,
                        100.0*ratioSpread, 100.0*margin);
            // A verdict whose margin is not clear of its own scatter is a coin
            // flip dressed up as a result. Say so on the same line as the word
            // PASS, where it cannot be read separately.
            if (std::fabs(margin) < 2.0*ratioSpread)
                std::printf("    ⚠ margin is under 2x the scatter -- this verdict is "
                            "not resolved. Add rounds.\n");
        }
        // 48.4 is a HARDCODED baseline measured on the 2080 Ti this project
        // develops on. Dividing it by an e2e measured elsewhere mixes two
        // machines: on the 3090 the CustomGBForce arm alone is 1.38x faster
        // (75.47 vs 54.78 ns/day), so the real baseline there is not 48.4.
        // Only the speedup transfers across hardware; the hour count does not.
        std::printf("  one 29x5ns matrix: 48.4 GPU-h -> %.1f h\n"
                    "    (48.4 is a 2080 Ti baseline; on other hardware only the\n"
                    "     speedup carries over, not the hour count)\n", 48.4/e2e);

        // Price one ordinary rc traversal, and say what the CWLD pair pass
        // costs in that unit. This is the number that says whether the kernel
        // is near its floor or nowhere near it.
        const double cBare = 1.0/arms[3].median();
        const double cNB = cNone - cBare;
        std::printf("\n  reference: NonbondedForce (CutoffPeriodic, same 1.2 nm,\n"
                    "  same neighbour list) costs %.4f of CustomGBForce -- that is\n"
                    "  ONE ordinary rc traversal.\n", cNB/(cGB - cNone));
        const double ourForce = (cNew - cNone) / (cGB - cNone);
        std::printf("  the CWLD force costs %.2fx that.  If it could be brought to\n"
                    "  1.0x, end-to-end would be %.3fx; at 1.5x, %.3fx.\n",
                    ourForce / (cNB/(cGB - cNone)),
                    1.0/((1.0 - s) + s*(cNB/(cGB - cNone))),
                    1.0/((1.0 - s) + s*1.5*(cNB/(cGB - cNone))));
        std::printf("\n  Do NOT compare these against the Python force_cost numbers.\n"
                    "  Cross-harness ratios are how every earlier estimate went wrong.\n");
        std::printf("==================================================\n");

        for (int a = 0; a < NARMS; a++) {
            delete arms[a].context;
            delete arms[a].integrator;
            delete arms[a].system;
        }
        delete source; delete refState;
    } catch (const std::exception& e) {
        std::printf("FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}
