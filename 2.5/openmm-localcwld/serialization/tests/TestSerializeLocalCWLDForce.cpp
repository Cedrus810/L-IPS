/**
 * LCWLD-060: XML round trip for LocalCWLDForce.
 *
 * The failure this guards against is not "serialization crashes" -- that is
 * loud. It is a field that silently does not survive the round trip, so a
 * checkpointed run resumes with different physics than it stopped with. So
 * every global is set to a NON-DEFAULT value before serializing: a field that
 * is dropped by the proxy would come back as its default, and comparing
 * defaults against defaults would pass.
 *
 * No Context, no Platform, no GPU.
 */
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>

#include "openmm/LocalCWLDForce.h"
#include "openmm/System.h"
#include "openmm/serialization/XmlSerializer.h"

using namespace LocalCWLDPlugin;

static int failures = 0;

static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("  FAIL: %s\n", what.c_str());
        failures++;
    }
}

static void checkClose(double got, double want, const std::string& what) {
    // Exact equality is the right test: XmlSerializer writes doubles with
    // enough digits to round trip, so anything less than exact means a field
    // went through a float somewhere.
    if (got != want) {
        std::printf("  FAIL: %s: got %.17g want %.17g\n", what.c_str(), got, want);
        failures++;
    }
}

/** Deliberately non-default everywhere. */
static LocalCWLDForce* buildReference() {
    LocalCWLDForce* force = new LocalCWLDForce();
    force->setEnvironmentCutoff(0.41);
    force->setCutoffDistance(1.37);
    force->setZMMOrder(3);
    force->setRho0(11.25);
    force->setKPolar(0.63);
    force->setChargeDeltaClamp(0.17);
    force->setUseQPenalty(true);
    force->setQPenaltyStrength(207.5);
    force->setOne4PiEps0(138.5);
    force->setForceGroup(7);
    force->setName("cwld-under-test");
    for (int i = 0; i < 5; i++) {
        const double d = 0.1 * i;
        force->addParticle(-0.834 + d, 0.021 + d, -0.15 + d, (i % 2) ? 1.0 : 0.0,
                           0.5 + d, 0.25 + d, 1.5 + d, 0.75 + d, 100 + i);
    }
    force->addExclusion(0, 1);
    force->addExclusion(3, 1);        // stored normalized as (1, 3)
    force->addExclusion(2, 4);
    return force;
}

static void compare(const LocalCWLDForce& a, const LocalCWLDForce& b,
                    const std::string& tag) {
    checkClose(b.getEnvironmentCutoff(), a.getEnvironmentCutoff(), tag + " environmentCutoff");
    checkClose(b.getCutoffDistance(),    a.getCutoffDistance(),    tag + " cutoffDistance");
    checkClose(b.getRho0(),              a.getRho0(),              tag + " rho0");
    checkClose(b.getKPolar(),            a.getKPolar(),            tag + " kPolar");
    checkClose(b.getChargeDeltaClamp(),  a.getChargeDeltaClamp(),  tag + " chargeDeltaClamp");
    checkClose(b.getQPenaltyStrength(),  a.getQPenaltyStrength(),  tag + " qPenaltyStrength");
    checkClose(b.getOne4PiEps0(),        a.getOne4PiEps0(),        tag + " one4PiEps0");
    check(b.getZMMOrder()        == a.getZMMOrder(),        tag + " zmmOrder");
    check(b.getUseQPenalty()     == a.getUseQPenalty(),     tag + " useQPenalty");
    check(b.getNonbondedMethod() == a.getNonbondedMethod(), tag + " method");
    check(b.getForceGroup()      == a.getForceGroup(),      tag + " forceGroup");
    check(b.getName()            == a.getName(),            tag + " name");

    check(b.getNumParticles() == a.getNumParticles(), tag + " particle count");
    for (int i = 0; i < a.getNumParticles() && i < b.getNumParticles(); i++) {
        double q1, c1, d1, p1, s1, k1, w1, f1, q2, c2, d2, p2, s2, k2, w2, f2;
        int r1, r2;
        a.getParticleParameters(i, q1, c1, d1, p1, s1, k1, w1, f1, r1);
        b.getParticleParameters(i, q2, c2, d2, p2, s2, k2, w2, f2, r2);
        const std::string at = tag + " particle " + std::to_string(i);
        checkClose(q2, q1, at + " qbase");
        checkClose(c2, c1, at + " chargeMod");
        checkClose(d2, d1, at + " dpolar");
        checkClose(p2, p1, at + " isPolar");
        checkClose(s2, s1, at + " densSource");
        checkClose(k2, k1, at + " densSink");
        checkClose(w2, w1, at + " sourceClassWeight");
        checkClose(f2, f1, at + " staticPhase");
        check(r2 == r1, at + " residueId");
    }

    check(b.getNumExclusions() == a.getNumExclusions(), tag + " exclusion count");
    for (int i = 0; i < a.getNumExclusions() && i < b.getNumExclusions(); i++) {
        int a1, a2, b1, b2;
        a.getExclusionParticles(i, a1, a2);
        b.getExclusionParticles(i, b1, b2);
        check(a1 == b1 && a2 == b2, tag + " exclusion " + std::to_string(i));
    }
}

int main() {
    LocalCWLDForce* reference = buildReference();

    // 1. Force on its own.
    {
        std::stringstream ss;
        OpenMM::XmlSerializer::serialize<LocalCWLDForce>(reference, "Force", ss);
        LocalCWLDForce* copy = OpenMM::XmlSerializer::deserialize<LocalCWLDForce>(ss);
        compare(*reference, *copy, "force");
        delete copy;
    }

    // 2. Inside a System -- the production path. This is what checkpointing
    //    and XmlSerializer::clone() actually go through, and it exercises the
    //    proxy lookup by type name rather than by static type.
    {
        OpenMM::System system;
        for (int i = 0; i < 5; i++)
            system.addParticle(1.0);
        system.addForce(buildReference());
        std::stringstream ss;
        OpenMM::XmlSerializer::serialize<OpenMM::System>(&system, "System", ss);
        OpenMM::System* copy = OpenMM::XmlSerializer::deserialize<OpenMM::System>(ss);
        check(copy->getNumForces() == 1, "system force count");
        const LocalCWLDForce* got =
            dynamic_cast<const LocalCWLDForce*>(&copy->getForce(0));
        check(got != nullptr, "system round trip produced a LocalCWLDForce");
        if (got != nullptr)
            compare(*reference, *got, "system");
        delete copy;
    }

    // 3. A file from a newer build must be refused, not read approximately.
    {
        std::stringstream ss;
        OpenMM::XmlSerializer::serialize<LocalCWLDForce>(reference, "Force", ss);
        std::string xml = ss.str();
        const std::string from = "version=\"1\"";
        const size_t at = xml.find(from);
        check(at != std::string::npos, "found the version attribute to bump");
        if (at != std::string::npos) {
            xml.replace(at, from.size(), "version=\"99\"");
            std::stringstream newer(xml);
            bool threw = false;
            try {
                delete OpenMM::XmlSerializer::deserialize<LocalCWLDForce>(newer);
            } catch (const std::exception&) {
                threw = true;
            }
            check(threw, "a newer serialization version is refused");
        }
    }

    delete reference;
    if (failures == 0)
        std::printf("TestSerializeLocalCWLDForce: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
