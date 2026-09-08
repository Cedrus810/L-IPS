#ifndef LOCALCWLD_TEST_ARMS_H_
#define LOCALCWLD_TEST_ARMS_H_

/**
 * The three comparison arms, shared by the benchmark and the pathological-
 * geometry test.
 *
 * This lives in a header rather than being written twice because a
 * hand-copied second version of `buildArm` is exactly the failure mode the
 * per-particle-parameter lookup below is written to avoid: a transcription
 * that looks right, drifts silently, and is never checked. Both consumers
 * must build byte-identical Systems or neither comparison means anything.
 */

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "openmm/CustomGBForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/LocalCWLDForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include "openmm/serialization/XmlSerializer.h"

// Test-only header: both consumers are single-file programs that already pull
// these names in unqualified.
using OpenMM::Vec3;
using LocalCWLDPlugin::LocalCWLDForce;

template <typename T>
static T* deserializeFile(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw OpenMM::OpenMMException("cannot open " + path);
    return OpenMM::XmlSerializer::deserialize<T>(in);
}

/** Copy the dumped System, optionally swapping CustomGBForce for ours. */
enum Arm { ARM_GB, ARM_NEW, ARM_NONE, ARM_BARE };

static OpenMM::System* buildArm(const OpenMM::System& source, Arm arm) {
    OpenMM::System* system = new OpenMM::System();
    for (int i = 0; i < source.getNumParticles(); i++)
        system->addParticle(source.getParticleMass(i));
    Vec3 a, b, c;
    source.getDefaultPeriodicBoxVectors(a, b, c);
    system->setDefaultPeriodicBoxVectors(a, b, c);
    for (int i = 0; i < source.getNumConstraints(); i++) {
        int p1, p2; double d;
        source.getConstraintParameters(i, p1, p2, d);
        system->addConstraint(p1, p2, d);
    }

    const OpenMM::CustomGBForce* gb = nullptr;
    for (int i = 0; i < source.getNumForces(); i++) {
        const OpenMM::Force& f = source.getForce(i);
        const OpenMM::CustomGBForce* asGB = dynamic_cast<const OpenMM::CustomGBForce*>(&f);
        if (asGB != nullptr) {
            gb = asGB;
            // Only the `gb` arm keeps it. Cloning through the serialiser keeps
            // every setting; hand-copying a CustomGBForce would be another
            // unverified transcription.
            if (arm == ARM_GB)
                system->addForce(OpenMM::XmlSerializer::clone(f));
        } else if (arm == ARM_BARE &&
                   dynamic_cast<const OpenMM::NonbondedForce*>(&f) != nullptr) {
            // ARM_BARE also drops the NonbondedForce. It is the only other
            // full-rc traversal in the System -- CutoffPeriodic at the same
            // 1.2 nm on the same neighbour list, and one of OpenMM's most
            // optimised kernels. Subtracting it from ARM_NONE prices one
            // ordinary rc traversal, which is the unit the CWLD pair pass
            // should be judged in. Without that unit "our pair pass costs 1.56
            // traversal-equivalents" is measured against CustomGBForce/3,
            // which is an average over three passes that are not equal.
            //
            // The arm is not physical (no electrostatics, no LJ). Timing only.
        } else {
            system->addForce(OpenMM::XmlSerializer::clone(f));
        }
    }
    if (gb == nullptr)
        throw OpenMM::OpenMMException("no CustomGBForce in the dumped System");

    if (arm == ARM_NEW) {
        std::map<std::string, int> column;
        for (int i = 0; i < gb->getNumPerParticleParameters(); i++)
            column[gb->getPerParticleParameterName(i)] = i;
        LocalCWLDForce* force = new LocalCWLDForce();
        for (int i = 0; i < source.getNumParticles(); i++) {
            std::vector<double> p;
            gb->getParticleParameters(i, p);
            // By name, never by index -- see TestCudaLocalCWLDAcceptance.
            force->addParticle(p[column.at("qbase")], p[column.at("charge_mod")],
                               p[column.at("dpolar")], p[column.at("is_polar")],
                               p[column.at("dens_source")], p[column.at("dens_sink")],
                               p[column.at("source_class_weight")],
                               p[column.at("static_phase")],
                               (int) std::lround(p[column.at("mol_id")]));
        }
        for (int i = 0; i < gb->getNumExclusions(); i++) {
            int p1, p2;
            gb->getExclusionParticles(i, p1, p2);
            force->addExclusion(p1, p2);
        }
        // Probe A (LCWLD-150 attribution, peer's proposal). Shrinking r_env
        // empties the compact list, so computeDensity and computeChainForce
        // degenerate to launch overhead while the full-rc pair pass is
        // untouched. The physics is WRONG under this setting -- it is a timing
        // probe, never an acceptance configuration.
        //
        // Note what this does NOT remove: buildEnvPairs still walks the whole
        // rc neighbour list, it just writes almost nothing. So the probe leaves
        // "pair pass + list build" standing, and against the measured f = 0.598
        // the readings are:
        //
        //   f_A ~ 0.41  -> the two r_env passes account for ~0.19, i.e. ~0.29 of
        //                  an rc pass each -- 2.2x the 0.129 measured with a
        //                  TILED r_env traversal. The flat compact list is the
        //                  problem; reorganise it.
        //   f_A ~ 0.55  -> the r_env passes are already cheap and the pair pass
        //                  itself is where the cost is. Optimising the r_env
        //                  side would be wasted.
        if (const char* probe = std::getenv("LOCALCWLD_PROBE_ENV_CUTOFF")) {
            const double r = std::atof(probe);
            force->setEnvironmentCutoff(r);
            std::printf("\n*** PROBE A ACTIVE: r_env = %g nm (production 0.35).\n"
                        "*** The physics is wrong. This is a timing probe.\n"
                        "*** Do NOT quote the resulting f or ns/day as a result.\n\n", r);
            std::fflush(stdout);
        }
        system->addForce(force);
    }
    return system;
}

#endif /* LOCALCWLD_TEST_ARMS_H_ */
