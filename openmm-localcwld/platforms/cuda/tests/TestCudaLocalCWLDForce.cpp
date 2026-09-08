/**
 * LCWLD-110 smoke test: does the whole chain actually connect?
 *
 * This is not a physics test -- LCWLD-090's neighbour-list ingestion is still
 * open, so execute() deliberately throws rather than return plausible numbers
 * from a half-wired path. What this test establishes is everything *up to*
 * that point, which is the part that is expensive to debug later:
 *
 *   1. OpenMM finds and dlopen()s the plugin from a directory
 *   2. registerKernelFactories() runs and registers under the frozen name
 *   3. a CUDA Context can be built with a LocalCWLDForce in the System
 *   4. LocalCWLDForceImpl requests the kernel and gets ours
 *   5. **NVRTC compiles platforms/common/kernels/localCWLD.cc**
 *
 * Step 5 is the real prize. The device code is a string compiled at run time,
 * so a syntax error in it cannot be caught by the C++ build -- only by getting
 * this far. Reaching the "LCWLD-090 is incomplete" exception means the kernel
 * source compiled; an NVRTC error means it did not.
 */

#include <cmath>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "openmm/Context.h"
#include "openmm/LocalCWLDForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"

using namespace LocalCWLDPlugin;
using OpenMM::OpenMMException;

static int failures = 0;

static void check(bool condition, const std::string& what) {
    std::printf("%s  %s\n", condition ? "  ok  " : "  FAIL", what.c_str());
    if (!condition)
        failures++;
}

/**
 * Run the whole check at one system size.
 *
 * Size matters here, and not for the usual reason. A tile is 32 atoms, so
 * n = 8 is a SINGLE block: only the exclusion-tile loop runs and the
 * neighbour-list loop is never entered. A green run at n=8 says nothing at all
 * about half the traversal code. n must exceed 32 -- comfortably, so that the
 * neighbour list actually contains off-diagonal tiles.
 */
static void runChecks(OpenMM::Platform& cuda, int numAtoms, double boxSide,
                      double spacing, const char* label) {
    std::printf("\n  --- %s (n = %d, box = %.1f nm) ---\n", label, numAtoms, boxSide);
    try {
        // A small but non-degenerate system: two residues so that the
        // same-residue density rule has something to suppress, and a box
        // comfortably larger than 2*rc.
        OpenMM::System system;
        const int n = numAtoms;
        for (int i = 0; i < n; i++)
            system.addParticle(1.0);
        system.setDefaultPeriodicBoxVectors(OpenMM::Vec3(boxSide, 0, 0),
                                            OpenMM::Vec3(0, boxSide, 0),
                                            OpenMM::Vec3(0, 0, boxSide));

        LocalCWLDForce* force = new LocalCWLDForce();
        for (int i = 0; i < n; i++) {
            // qbase, chargeMod, dpolar, isPolar, densSource, densSink,
            // sourceClassWeight, staticPhase, residueId
            force->addParticle(i % 2 == 0 ? -0.834 : 0.417, 1.0,
                               i % 2 == 0 ? -0.15 : 0.075, 1.0,
                               i % 2 == 0 ? 1.0 : 0.0, 1.0, 0.5, 1.0, i / 4);
        }
        force->addExclusion(0, 1);
        system.addForce(force);

        OpenMM::VerletIntegrator integrator(0.001);
        OpenMM::Context context(system, integrator, cuda);
        check(true, "CUDA Context created with LocalCWLDForce in the System");

        // Fill the box on a lattice so a real neighbour list gets built,
        // rather than one dense clump that lands in a single tile.
        std::vector<OpenMM::Vec3> positions;
        // `spacing` is explicit, not boxSide/side: with 8 atoms in a 4 nm box a
        // space-filling lattice puts every pair beyond rc, and the whole run
        // silently evaluates to zero -- a green test that tested nothing.
        const int side = (int) std::ceil(std::cbrt((double) n));
        for (int i = 0; i < n; i++)
            positions.push_back(OpenMM::Vec3(
                spacing * (i % side) + 0.01 * (i % 7),
                spacing * ((i / side) % side) + 0.01 * (i % 5),
                spacing * (i / (side * side)) + 0.01 * (i % 3)));
        context.setPositions(positions);

        // Zero Integrator.step(). One static evaluation.
        const OpenMM::State state =
            context.getState(OpenMM::State::Forces | OpenMM::State::Energy);
        const double energy = state.getPotentialEnergy();
        const std::vector<OpenMM::Vec3>& forces = state.getForces();

        std::printf("  energy = %.6f kJ/mol\n", energy);
        for (int i = 0; i < 3; i++)
            std::printf("    F[%d] = (%+.4f, %+.4f, %+.4f)\n",
                        i, forces[i][0], forces[i][1], forces[i][2]);

        // These are sanity checks, not a physics verification. Verification is
        // DEC-005: per-particle comparison against production CustomGBForce on
        // real 1AAY, plus the bias/tile-clustering criteria. Nothing here says
        // the numbers are *right* -- only that they are not obviously broken.
        check(std::isfinite(energy), "energy is finite");
        bool allFinite = true, anyNonzero = false;
        OpenMM::Vec3 net(0, 0, 0);
        for (const OpenMM::Vec3& f : forces) {
            for (int k = 0; k < 3; k++) {
                if (!std::isfinite(f[k])) allFinite = false;
                if (f[k] != 0.0) anyNonzero = true;
            }
            net += f;
        }
        check(allFinite, "all forces are finite");
        check(anyNonzero, "forces are not all zero (the kernel did something)");

        // Newton's third law.
        //
        // ⚠ WEAK CRITERION -- read what it does NOT prove. The pair force is
        // scattered as +f / -f into 64-bit fixed-point accumulators, so integer
        // addition cancels it exactly whether or not the physics is right. In
        // particular **dropping a pair does not break it**: a missing pair
        // removes equal and opposite contributions from both i and j, and the
        // net stays zero. So this is blind to the single most dangerous failure
        // mode here (a tile read twice or not at all).
        //
        // What it does prove: the scatter sign and the i/j pairing are not
        // reversed. Useful, cheap, and nothing more. Completeness is tested by
        // the brute-force comparison below at small N, and at full scale by
        // DEC-005 (residual tile clustering + the HIE149 single-point probe).
        const double netMagnitude = std::sqrt(net[0]*net[0] + net[1]*net[1] + net[2]*net[2]);
        std::printf("  net force |sum F| = %.3e kJ/mol/nm\n", netMagnitude);
        check(netMagnitude < 1e-3, "net force cancels (Newton's third law)");

        // ---- Completeness: brute-force O(N^2) reference at N=8 -------------
        //
        // THIS is the check that a missing tile fails. Every pair here is
        // enumerated directly, with no tiles and no neighbour list, so any pair
        // the kernel skipped shows up as an energy discrepancy.
        //
        // The formula below is a test-only transcription of the frozen pair
        // term (plan section 18.4) -- same role as the LCWLD-030 golden
        // fixtures, and deliberately written in the most obvious way rather
        // than the fastest. It is not a production path and must never be
        // reused as one.
        {
            const double rc = force->getCutoffDistance();
            const double rEnv = force->getEnvironmentCutoff();
            const double k = force->getOne4PiEps0();
            const double rho0 = force->getRho0(), kPolar = force->getKPolar();
            const double clamp = force->getChargeDeltaClamp();
            double c0, c2, c4, c6;
            switch (force->getZMMOrder()) {
                case 1: c0 = -1.5; c2 = 0.5; c4 = 0; c6 = 0; break;
                case 2: c0 = -15.0/8; c2 = 5.0/4; c4 = -3.0/8; c6 = 0; break;
                default: c0 = -35.0/16; c2 = 35.0/16; c4 = -21.0/16; c6 = 5.0/16;
            }

            std::vector<double> qb(n), src(n), sink(n), amp(n);
            std::vector<int> res(n);
            for (int i = 0; i < n; i++) {
                double q, cm, dp, ip, ds, dk, w, ph;
                int r;
                force->getParticleParameters(i, q, cm, dp, ip, ds, dk, w, ph, r);
                qb[i] = q; src[i] = ds*w*cm; sink[i] = dk; amp[i] = ph*ip*dp; res[i] = r;
            }
            std::vector<std::pair<int,int> > excl;
            for (int e = 0; e < force->getNumExclusions(); e++) {
                int p1, p2;
                force->getExclusionParticles(e, p1, p2);
                excl.push_back(std::make_pair(p1, p2));
            }
            auto excluded = [&](int i, int j) {
                for (const auto& e : excl)
                    if ((e.first == i && e.second == j) || (e.first == j && e.second == i))
                        return true;
                return false;
            };
            auto dist2 = [&](int i, int j) {
                double d[3];
                for (int c = 0; c < 3; c++) {
                    d[c] = positions[j][c] - positions[i][c];
                    d[c] -= boxSide * std::round(d[c] / boxSide);
                }
                return d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
            };

            std::vector<double> dens(n, 0.0);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    if (i == j || res[i] == res[j] || excluded(i, j))
                        continue;
                    const double r2 = dist2(i, j);
                    if (r2 >= rEnv*rEnv)
                        continue;
                    const double t = 1 - r2/(rEnv*rEnv);
                    dens[i] += sink[i] * src[j] * t * t;
                }

            std::vector<double> q(n);
            for (int i = 0; i < n; i++) {
                const double inner = std::tanh(kPolar * dens[i] / rho0);
                q[i] = qb[i] + (amp[i] == 0 ? 0.0
                                            : clamp * std::tanh(amp[i]*inner/clamp));
            }

            double expected = 0.0;
            for (int i = 0; i < n; i++)
                for (int j = i+1; j < n; j++) {
                    if (excluded(i, j))
                        continue;
                    const double r2 = dist2(i, j);
                    if (r2 >= rc*rc)
                        continue;
                    const double r = std::sqrt(r2), x2 = r2/(rc*rc);
                    const double poly = (c0 + x2*(c2 + x2*(c4 + x2*c6)))/rc;
                    const double diff = qb[i]*(q[j]-qb[j]) + qb[j]*(q[i]-qb[i])
                                      + (q[i]-qb[i])*(q[j]-qb[j]);
                    expected += k * (diff/r + q[i]*q[j]*poly + qb[i]*qb[j]/rc);
                }

            std::printf("  brute force energy = %.6f kJ/mol   (kernel %.6f, diff %.2e)\n",
                        expected, energy, std::fabs(energy - expected));
            check(std::fabs(energy - expected) < 1e-3 * (1 + std::fabs(expected)),
                  "energy matches the brute-force reference (no pairs missing)");
        }
    } catch (const std::exception& e) {
        std::printf("  FAIL  unexpected exception: %s\n", e.what());
        failures++;
    }
}

int main(int argc, char* argv[]) {
    try {
        const char* openmmPlugins = std::getenv("OPENMM_PLUGIN_DIR");
        const std::string openmmPluginDir =
            openmmPlugins != nullptr ? std::string(openmmPlugins)
                                     : OpenMM::Platform::getDefaultPluginsDirectory();
        const std::vector<std::string> base =
            OpenMM::Platform::loadPluginsFromDirectory(openmmPluginDir);
        std::printf("  loaded %zu OpenMM plugin(s) from %s\n",
                    base.size(), openmmPluginDir.c_str());
        check(!base.empty(), "OpenMM's own plugins loaded");
        if (argc > 1) {
            const std::vector<std::string> loaded =
                OpenMM::Platform::loadPluginsFromDirectory(argv[1]);
            bool foundOurs = false;
            for (const std::string& name : loaded)
                if (name.find("LocalCWLDCUDA") != std::string::npos)
                    foundOurs = true;
            check(foundOurs, "libOpenMMLocalCWLDCUDA.so was loaded");
        }
        OpenMM::Platform& cuda = OpenMM::Platform::getPlatformByName("CUDA");
        check(true, "CUDA platform available");

        // n=8: one block, exercises the exclusion-tile loop only.
        // n=200: 7 blocks, so the neighbour-list loop is actually entered.
        // n=8, tight cluster: one block, so only the exclusion-tile loop runs.
        // n=200 on a 0.55 nm lattice: 7 blocks and a real neighbour list, so the
        // second loop is actually entered. A green n=8 alone would say nothing
        // about half the traversal code.
        runChecks(cuda, 8, 4.0, 0.30, "single block (exclusion-tile loop only)");
        runChecks(cuda, 200, 4.0, 0.55, "multi block (neighbour-list loop too)");
    } catch (const std::exception& e) {
        std::printf("  FAIL  setup failed: %s\n", e.what());
        failures++;
    }

    if (failures == 0)
        std::printf("TestCudaLocalCWLDForce: all checks passed\n");
    else
        std::printf("TestCudaLocalCWLDForce: %d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
