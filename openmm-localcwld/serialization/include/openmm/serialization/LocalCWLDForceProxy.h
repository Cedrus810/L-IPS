#ifndef OPENMM_LOCALCWLDFORCEPROXY_H_
#define OPENMM_LOCALCWLDFORCEPROXY_H_

#include "openmm/internal/windowsExport.h"
#include "openmm/serialization/SerializationProxy.h"

namespace LocalCWLDPlugin {

/**
 * XML serialization for LocalCWLDForce (LCWLD-060).
 *
 * Without this, `XmlSerializer::serialize()` throws on any System containing a
 * LocalCWLDForce -- which takes out checkpointing, State saving, and OpenMM's
 * own `XmlSerializer::clone()`. It is not an optional convenience.
 *
 * It is also the whole Python story for now. There are no SWIG bindings
 * (LCWLD-080) and no SWIG, pybind11 or Cython in the environment, so Python
 * cannot construct a LocalCWLDForce directly. With this proxy registered it
 * does not have to: `XmlSerializer.deserialize()` builds the object in C++ and
 * hands Python back a generic `Force` proxy, and `System.addForce()` only ever
 * needed a `Force*`.
 */
class LocalCWLDForceProxy : public OpenMM::SerializationProxy {
public:
    LocalCWLDForceProxy();
    void serialize(const void* object, OpenMM::SerializationNode& node) const override;
    void* deserialize(const OpenMM::SerializationNode& node) const override;
};

} // namespace LocalCWLDPlugin

#endif /* OPENMM_LOCALCWLDFORCEPROXY_H_ */
