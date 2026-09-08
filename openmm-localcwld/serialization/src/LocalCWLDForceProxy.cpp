#include "openmm/serialization/LocalCWLDForceProxy.h"

#include "openmm/LocalCWLDForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/serialization/SerializationNode.h"

using namespace LocalCWLDPlugin;
using namespace OpenMM;

/**
 * Bump when the format changes. deserialize() refuses anything newer than it
 * knows, rather than silently reading a file it does not understand -- a
 * mis-read parameter here is a physics change that no test would catch.
 */
static const int SERIALIZATION_VERSION = 1;

LocalCWLDForceProxy::LocalCWLDForceProxy()
    : SerializationProxy("LocalCWLDForce") {
}

void LocalCWLDForceProxy::serialize(const void* object, SerializationNode& node) const {
    const LocalCWLDForce& force = *reinterpret_cast<const LocalCWLDForce*>(object);
    node.setIntProperty("version", SERIALIZATION_VERSION);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());

    node.setIntProperty("method", (int) force.getNonbondedMethod());
    node.setDoubleProperty("environmentCutoff", force.getEnvironmentCutoff());
    node.setDoubleProperty("cutoffDistance", force.getCutoffDistance());
    node.setIntProperty("zmmOrder", force.getZMMOrder());
    node.setDoubleProperty("rho0", force.getRho0());
    node.setDoubleProperty("kPolar", force.getKPolar());
    node.setDoubleProperty("chargeDeltaClamp", force.getChargeDeltaClamp());
    node.setBoolProperty("useQPenalty", force.getUseQPenalty());
    node.setDoubleProperty("qPenaltyStrength", force.getQPenaltyStrength());
    node.setDoubleProperty("one4PiEps0", force.getOne4PiEps0());

    SerializationNode& particles = node.createChildNode("Particles");
    for (int i = 0; i < force.getNumParticles(); i++) {
        double qbase, chargeMod, dpolar, isPolar, densSource, densSink,
               sourceClassWeight, staticPhase;
        int residueId;
        force.getParticleParameters(i, qbase, chargeMod, dpolar, isPolar, densSource,
                                    densSink, sourceClassWeight, staticPhase, residueId);
        // Written BY NAME, never positionally. The same rule the acceptance
        // test follows for the CustomGBForce columns, for the same reason:
        // a reordering must break loudly, not shift the physics silently.
        particles.createChildNode("Particle")
            .setDoubleProperty("qbase", qbase)
            .setDoubleProperty("chargeMod", chargeMod)
            .setDoubleProperty("dpolar", dpolar)
            .setDoubleProperty("isPolar", isPolar)
            .setDoubleProperty("densSource", densSource)
            .setDoubleProperty("densSink", densSink)
            .setDoubleProperty("sourceClassWeight", sourceClassWeight)
            .setDoubleProperty("staticPhase", staticPhase)
            .setIntProperty("residueId", residueId);
    }

    SerializationNode& exclusions = node.createChildNode("Exclusions");
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int p1, p2;
        force.getExclusionParticles(i, p1, p2);
        exclusions.createChildNode("Exclusion")
            .setIntProperty("p1", p1)
            .setIntProperty("p2", p2);
    }
}

void* LocalCWLDForceProxy::deserialize(const SerializationNode& node) const {
    const int version = node.getIntProperty("version");
    if (version > SERIALIZATION_VERSION)
        throw OpenMMException("LocalCWLDForce: serialized with a newer format (version " +
                              std::to_string(version) + "); this build understands up to " +
                              std::to_string(SERIALIZATION_VERSION));
    LocalCWLDForce* force = new LocalCWLDForce();
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));

        // Every setter validates, so a corrupted file fails here rather than
        // at the first force evaluation.
        force->setNonbondedMethod((LocalCWLDForce::NonbondedMethod)
                                  node.getIntProperty("method"));
        force->setEnvironmentCutoff(node.getDoubleProperty("environmentCutoff"));
        force->setCutoffDistance(node.getDoubleProperty("cutoffDistance"));
        force->setZMMOrder(node.getIntProperty("zmmOrder"));
        force->setRho0(node.getDoubleProperty("rho0"));
        force->setKPolar(node.getDoubleProperty("kPolar"));
        force->setChargeDeltaClamp(node.getDoubleProperty("chargeDeltaClamp"));
        force->setUseQPenalty(node.getBoolProperty("useQPenalty"));
        force->setQPenaltyStrength(node.getDoubleProperty("qPenaltyStrength"));
        force->setOne4PiEps0(node.getDoubleProperty("one4PiEps0"));

        for (const SerializationNode& p : node.getChildNode("Particles").getChildren())
            force->addParticle(p.getDoubleProperty("qbase"),
                               p.getDoubleProperty("chargeMod"),
                               p.getDoubleProperty("dpolar"),
                               p.getDoubleProperty("isPolar"),
                               p.getDoubleProperty("densSource"),
                               p.getDoubleProperty("densSink"),
                               p.getDoubleProperty("sourceClassWeight"),
                               p.getDoubleProperty("staticPhase"),
                               p.getIntProperty("residueId"));

        for (const SerializationNode& e : node.getChildNode("Exclusions").getChildren())
            force->addExclusion(e.getIntProperty("p1"), e.getIntProperty("p2"));
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
