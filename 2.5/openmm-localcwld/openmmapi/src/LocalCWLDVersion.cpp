#include "openmm/LocalCWLDVersion.h"

namespace LocalCWLDPlugin {

std::string LocalCWLDVersion::getVersion() {
    return LOCALCWLD_VERSION_STRING;
}

std::string LocalCWLDVersion::getPhysicsSemanticsVersion() {
    // Plan section 17 froze the v0.1 physics semantics table. Bump this only
    // together with that table, never to mark a code refactor.
    return "v0.1";
}

} // namespace LocalCWLDPlugin
