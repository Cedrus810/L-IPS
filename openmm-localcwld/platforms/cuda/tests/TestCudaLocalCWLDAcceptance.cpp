/**
 * DEC-005 acceptance: LocalCWLDForce vs production CustomGBForce on real 1AAY.
 *
 * Inputs come from `docs/reports/dec005_dump_1aay.py`, all through OpenMM's own
 * serialiser:
 *
 *   <tag>_system.xml        the CWLD System; the CustomGBForce inside it carries
 *                           the nine per-particle parameters, their NAMES, and
 *                           every exclusion
 *   <tag>_state_mixed.xml   positions, box, and the ISOLATED CustomGBForce
 *                           reference forces (production precision)
 *   <tag>_groups.txt        integer index lists for the DEC-005 comparison
 *                           groups (no floats, no units, so no silent drift)
 *
 * There is no hand-written parser for the numerical data on purpose. A
 * hand-rolled dump format fails the same way a hand-rolled parameter mapping
 * does -- column order, units and precision drift silently and everything still
 * looks right. OpenMM is the sole authority on the format.
 *
 * Parameters are read BY NAME, never by index: if someone adds an
 * `addPerParticleParameter` in v26, a positional read would silently shift
 * every column. Same failure class as the DIVALENT_METAL_SPECIES /
 * species_of() mismatch that hid Zn for a whole system.
 *
 * ⚠ The reference forces are the isolated CustomGBForce, not the system total.
 * Comparing against the total would differ by orders of magnitude and read as
 * "the kernel is completely wrong".
 *
 * Zero Integrator.step(): one static evaluation.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
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
#include "openmm/serialization/XmlSerializer.h"

using namespace LocalCWLDPlugin;
using OpenMM::Vec3;

// DEC-005 section 3.4: defined against the SIGNAL, not the noise floor.
// 1.0 kJ/mol/nm is 8.2x the measured fp32 floor and 1/106 of the weakest real
// signal. Fixed before this kernel existed; see DEC-005 section 6.
static const double FORCE_ATOL = 1.0;      // kJ/mol/nm
static const double FORCE_RTOL = 2.5e-4;
static const double BIAS_T_MAX = 3.0;

static int failures = 0;

static void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    if (!ok) failures++;
}

/** XmlSerializer::deserialize takes an istream, so hand it the file directly. */
template <typename T>
static T* deserializeFile(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw OpenMM::OpenMMException("cannot open " + path);
    return OpenMM::XmlSerializer::deserialize<T>(in);
}

static double norm(const Vec3& v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::printf("  skipped: usage %s <plugin dir> <dump prefix>\n", argv[0]);
        return 0;                       // the dump is generated on demand
    }
    const std::string pluginDir = argv[1], prefix = argv[2];
    try {
        const char* env = std::getenv("OPENMM_PLUGIN_DIR");
        OpenMM::Platform::loadPluginsFromDirectory(
            env ? std::string(env) : OpenMM::Platform::getDefaultPluginsDirectory());
        OpenMM::Platform::loadPluginsFromDirectory(pluginDir);
        OpenMM::Platform& cuda = OpenMM::Platform::getPlatformByName("CUDA");

        OpenMM::System* refSystem = deserializeFile<OpenMM::System>(prefix + "_system.xml");
        OpenMM::State* refStatePtr = deserializeFile<OpenMM::State>(
            prefix + "_state_mixed.xml");
        const OpenMM::State& refState = *refStatePtr;
        const int n = refSystem->getNumParticles();
        const std::vector<Vec3>& positions = refState.getPositions();
        const std::vector<Vec3>& refForces = refState.getForces();
        std::printf("  %d atoms, reference energy %.4f kJ/mol\n",
                    n, refState.getPotentialEnergy());

        const OpenMM::CustomGBForce* gb = nullptr;
        for (int i = 0; i < refSystem->getNumForces(); i++)
            if ((gb = dynamic_cast<const OpenMM::CustomGBForce*>(
                     &refSystem->getForce(i))) != nullptr)
                break;
        if (gb == nullptr)
            throw OpenMM::OpenMMException("no CustomGBForce in the dumped System");

        // By name. A positional read here is the bug this comment exists to
        // prevent.
        std::map<std::string, int> column;
        for (int i = 0; i < gb->getNumPerParticleParameters(); i++)
            column[gb->getPerParticleParameterName(i)] = i;
        for (const char* need : {"qbase", "charge_mod", "dpolar", "is_polar",
                                 "dens_source", "dens_sink", "source_class_weight",
                                 "static_phase", "mol_id"})
            if (column.find(need) == column.end())
                throw OpenMM::OpenMMException(
                    std::string("dumped CustomGBForce has no parameter '") + need + "'");

        OpenMM::System system;
        for (int i = 0; i < n; i++)
            system.addParticle(refSystem->getParticleMass(i));
        Vec3 a, b, c;
        refSystem->getDefaultPeriodicBoxVectors(a, b, c);
        system.setDefaultPeriodicBoxVectors(a, b, c);

        LocalCWLDForce* force = new LocalCWLDForce();
        for (int i = 0; i < n; i++) {
            std::vector<double> p;
            gb->getParticleParameters(i, p);
            force->addParticle(p[column["qbase"]], p[column["charge_mod"]],
                               p[column["dpolar"]], p[column["is_polar"]],
                               p[column["dens_source"]], p[column["dens_sink"]],
                               p[column["source_class_weight"]],
                               p[column["static_phase"]],
                               (int) std::lround(p[column["mol_id"]]));
        }
        for (int i = 0; i < gb->getNumExclusions(); i++) {
            int p1, p2;
            gb->getExclusionParticles(i, p1, p2);
            force->addExclusion(p1, p2);
        }
        std::printf("  built LocalCWLDForce: %d particles, %d exclusions\n",
                    force->getNumParticles(), force->getNumExclusions());
        system.addForce(force);

        OpenMM::VerletIntegrator integrator(0.001);
        OpenMM::Context context(system, integrator, cuda, {{"Precision", "single"}});
        context.setPositions(positions);
        const OpenMM::State state =
            context.getState(OpenMM::State::Forces | OpenMM::State::Energy);
        const std::vector<Vec3>& got = state.getForces();

        const double eRel = std::fabs(state.getPotentialEnergy()
                                      - refState.getPotentialEnergy())
                            / std::fabs(refState.getPotentialEnergy());
        std::printf("  energy: kernel %.4f  ref %.4f  rel %.3e\n",
                    state.getPotentialEnergy(), refState.getPotentialEnergy(), eRel);
        check(eRel < 1e-4, "energy within the coarse screen (DEC-005: not a criterion)");

        // Diagnostic: is the residual consistent with a global sign flip or a
        // constant scale? A dot-product ratio separates "wrong sign" from
        // "wrong magnitude" from "wrong everywhere", which staring at |dF|
        // cannot.
        {
            double dot = 0, refSq = 0, gotSq = 0;
            for (int i = 0; i < n; i++) {
                dot += got[i][0]*refForces[i][0] + got[i][1]*refForces[i][1]
                     + got[i][2]*refForces[i][2];
                refSq += refForces[i][0]*refForces[i][0] + refForces[i][1]*refForces[i][1]
                       + refForces[i][2]*refForces[i][2];
                gotSq += got[i][0]*got[i][0] + got[i][1]*got[i][1] + got[i][2]*got[i][2];
            }
            std::printf("    diag: <got,ref>/|ref|^2 = %+.4f   |got|/|ref| = %.4f\n",
                        dot/refSq, std::sqrt(gotSq/refSq));
        }

        // ---- Groups: every atom individually -------------------------------
        std::ifstream gin(prefix + "_groups.txt");
        if (!gin)
            throw OpenMM::OpenMMException("cannot open " + prefix + "_groups.txt");
        std::string line;
        std::vector<double> devs(n), projs(n);
        for (int i = 0; i < n; i++) {
            const Vec3 d(got[i][0]-refForces[i][0], got[i][1]-refForces[i][1],
                         got[i][2]-refForces[i][2]);
            devs[i] = norm(d);
            const double m = norm(refForces[i]);
            projs[i] = (m < 1e-12) ? 0.0
                     : (d[0]*refForces[i][0] + d[1]*refForces[i][1]
                        + d[2]*refForces[i][2]) / m;
        }
        while (std::getline(gin, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string name; int count;
            ss >> name >> count;
            double worst = 0; int worstAtom = -1;
            for (int j = 0; j < count; j++) {
                int i; ss >> i;
                const double allowed = FORCE_ATOL + FORCE_RTOL * norm(refForces[i]);
                if (devs[i] / allowed > worst) { worst = devs[i]/allowed; worstAtom = i; }
            }
            std::printf("    %-16s n=%3d  worst %.3fx allowance (atom %d, |dF|=%.3e)\n",
                        name.c_str(), count, worst, worstAtom,
                        worstAtom >= 0 ? devs[worstAtom] : 0.0);
            check(worst <= 1.0, name + ": every atom within DEC-005 tolerance");
        }

        // ---- Group E: whole-system quantiles --------------------------------
        std::vector<double> sorted = devs;
        std::sort(sorted.begin(), sorted.end());
        std::printf("    whole system     |dF| p50 %.3e  p99 %.3e  max %.3e\n",
                    sorted[n/2], sorted[(int)(0.99*n)], sorted.back());
        check(sorted[(int)(0.99*n)] <= FORCE_ATOL, "E: p99 deviation within atol");

        // ---- Bias, per component (DEC-005 3.7) ------------------------------
        // Per-component baseline is ~0 and does not depend on the accumulation
        // design, so it is a usable criterion. The projection's baseline DOES
        // depend on it, so the projection is reported only.
        for (int cc = 0; cc < 3; cc++) {
            double sum = 0, sum2 = 0;
            for (int i = 0; i < n; i++) {
                const double v = got[i][cc] - refForces[i][cc];
                sum += v; sum2 += v*v;
            }
            const double mean = sum/n, var = sum2/n - mean*mean;
            const double t = mean / std::sqrt(std::max(var, 1e-30)/n);
            std::printf("    bias t[%c] = %+7.2f\n", "xyz"[cc], t);
            check(std::fabs(t) < BIAS_T_MAX,
                  std::string("bias: ") + "xyz"[cc] + " residual mean compatible with zero");
        }

        // ---- Tile clustering (DEC-005 3.7) ----------------------------------
        // ⚠ WEAKENED: blocks on USER atom order. DEC-005 requires OpenMM's
        // INTERNAL order (CudaContext reorders spatially and maps forces back),
        // which needs CudaContext::getAtomIndex() and is not reachable from a
        // test that only sees the public API. A missing tile in internal order
        // is therefore smeared across user-order blocks rather than isolated.
        // Reported, not asserted -- an assertion here would imply a guarantee
        // this version cannot make.
        {
            const int block = 32, nBlocks = (n + block - 1) / block;
            double worstT = 0; int worstBlock = -1;
            for (int bIdx = 0; bIdx < nBlocks; bIdx++) {
                const int lo = bIdx*block, hi = std::min(lo+block, n), m = hi-lo;
                double sum = 0, sum2 = 0;
                for (int i = lo; i < hi; i++) { sum += projs[i]; sum2 += projs[i]*projs[i]; }
                const double mean = sum/m, var = sum2/m - mean*mean;
                const double t = mean / std::sqrt(std::max(var, 1e-30)/m);
                if (std::fabs(t) > std::fabs(worstT)) { worstT = t; worstBlock = bIdx; }
            }
            std::printf("    tile clustering (USER order, weakened): "
                        "max |t_b| = %.2f at block %d/%d\n",
                        std::fabs(worstT), worstBlock, nBlocks);
        }
        delete refSystem;
        delete refStatePtr;
    } catch (const std::exception& e) {
        std::printf("  FAIL  %s\n", e.what());
        failures++;
    }
    if (failures == 0)
        std::printf("TestCudaLocalCWLDAcceptance: all checks passed\n");
    else
        std::printf("TestCudaLocalCWLDAcceptance: %d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
