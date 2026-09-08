#ifndef OPENMM_LOCALCWLDVERSION_H_
#define OPENMM_LOCALCWLDVERSION_H_

#include <string>
#include "internal/windowsExportLocalCWLD.h"

namespace LocalCWLDPlugin {

/**
 * Build identification for the LocalCWLD plugin (ticket LCWLD-010).
 *
 * This is the placeholder content of the OpenMMLocalCWLD API library: it lets
 * the skeleton link and install before LCWLD-040 adds LocalCWLDForce. It
 * carries no physics and must never acquire any.
 */
class OPENMM_EXPORT_LOCALCWLD LocalCWLDVersion {
public:
    /** Semantic version of this plugin, e.g. "0.1.0". */
    static std::string getVersion();
    /** Frozen physics-semantics revision from plan section 17 ("v0.1"). */
    static std::string getPhysicsSemanticsVersion();
};

} // namespace LocalCWLDPlugin

#endif /* OPENMM_LOCALCWLDVERSION_H_ */
