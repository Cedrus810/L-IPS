/**
 * Register the serialization proxy when this library is loaded.
 *
 * The constructor attribute matters for the Python path: nothing on the Python
 * side ever calls a registration function. `Platform.loadPluginsFromDirectory`
 * dlopen()s libOpenMMLocalCWLDCUDA.so, which links this library, which runs
 * this on load. By the time Python can reach a LocalCWLDForce at all, the
 * proxy is already registered.
 */
#include <typeinfo>

#include "openmm/LocalCWLDForce.h"
#include "openmm/serialization/LocalCWLDForceProxy.h"
#include "openmm/serialization/SerializationProxy.h"

extern "C" void registerLocalCWLDSerializationProxies();

#if defined(_MSC_VER)
#include <windows.h>
BOOL WINAPI DllMain(HANDLE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH)
        registerLocalCWLDSerializationProxies();
    return TRUE;
}
#else
extern "C" __attribute__((constructor)) void registerLocalCWLDSerializationProxiesOnLoad() {
    registerLocalCWLDSerializationProxies();
}
#endif

extern "C" void registerLocalCWLDSerializationProxies() {
    OpenMM::SerializationProxy::registerProxy(
        typeid(LocalCWLDPlugin::LocalCWLDForce),
        new LocalCWLDPlugin::LocalCWLDForceProxy());
}
