#ifndef OPENMM_WINDOWSEXPORT_LOCALCWLD_H_
#define OPENMM_WINDOWSEXPORT_LOCALCWLD_H_

/*
 * Shared-library visibility macro for the LocalCWLD public API
 * (ticket LCWLD-010). DEC-003 scopes v0.1 to Linux; the Windows branch exists
 * so the public headers stay portable, not because Windows is supported.
 */

#if defined(_MSC_VER)
    #if defined(LOCALCWLD_BUILDING_STATIC_LIBRARY)
        #define OPENMM_EXPORT_LOCALCWLD
    #elif defined(LOCALCWLD_BUILDING_SHARED_LIBRARY)
        #define OPENMM_EXPORT_LOCALCWLD __declspec(dllexport)
    #else
        #define OPENMM_EXPORT_LOCALCWLD __declspec(dllimport)
    #endif
#else
    #define OPENMM_EXPORT_LOCALCWLD __attribute__((visibility("default")))
#endif

#endif /* OPENMM_WINDOWSEXPORT_LOCALCWLD_H_ */
