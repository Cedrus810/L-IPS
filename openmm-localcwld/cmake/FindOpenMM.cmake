#---------------------------------------------------------------------------
# FindOpenMM -- locate an OpenMM installation (ticket LCWLD-010).
#
# The conda-forge `openmm` package does NOT ship OpenMMConfig.cmake (verified
# in DEC-001 section 2), so a hand-written finder is required rather than a
# find_package(OpenMM CONFIG).
#
# Input:
#   OPENMM_DIR (variable or environment) -- installation prefix.
# Output:
#   OPENMM_FOUND, OPENMM_ROOT_DIR, OPENMM_INCLUDE_DIR, OPENMM_LIBRARY,
#   OPENMM_PLUGIN_DIR and imported target OpenMM::OpenMM.
#---------------------------------------------------------------------------

if(NOT OPENMM_DIR AND DEFINED ENV{OPENMM_DIR})
    set(OPENMM_DIR "$ENV{OPENMM_DIR}")
endif()
if(NOT OPENMM_DIR AND DEFINED ENV{OPENMM_PREFIX})
    set(OPENMM_DIR "$ENV{OPENMM_PREFIX}")
endif()
if(NOT OPENMM_DIR AND DEFINED ENV{CONDA_PREFIX})
    # Convenience only: an activated conda env that happens to contain OpenMM.
    if(EXISTS "$ENV{CONDA_PREFIX}/include/OpenMM.h")
        set(OPENMM_DIR "$ENV{CONDA_PREFIX}")
    endif()
endif()

find_path(OPENMM_INCLUDE_DIR
    NAMES OpenMM.h
    HINTS ${OPENMM_DIR}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)
find_path(OPENMM_INCLUDE_DIR NAMES OpenMM.h PATH_SUFFIXES include)

find_library(OPENMM_LIBRARY
    NAMES OpenMM
    HINTS ${OPENMM_DIR}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH)
find_library(OPENMM_LIBRARY NAMES OpenMM PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenMM
    REQUIRED_VARS OPENMM_INCLUDE_DIR OPENMM_LIBRARY
    REASON_FAILURE_MESSAGE
        "Could not locate OpenMM.h / libOpenMM. Pass the installation prefix \
explicitly, e.g. -DOPENMM_DIR=/path/to/envs/openmm_dev (the prefix that \
contains include/OpenMM.h and lib/libOpenMM.so).")

if(OPENMM_FOUND)
    get_filename_component(OPENMM_ROOT_DIR "${OPENMM_INCLUDE_DIR}" DIRECTORY)
    get_filename_component(_openmm_libdir "${OPENMM_LIBRARY}" DIRECTORY)
    set(OPENMM_PLUGIN_DIR "${_openmm_libdir}/plugins"
        CACHE PATH "OpenMM runtime plugin directory")

    # Version is not exported in a header macro by every build; read it from
    # the shared library soname area only if a header happens to carry it.
    if(NOT TARGET OpenMM::OpenMM)
        add_library(OpenMM::OpenMM UNKNOWN IMPORTED)
        set_target_properties(OpenMM::OpenMM PROPERTIES
            IMPORTED_LOCATION "${OPENMM_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OPENMM_INCLUDE_DIR}")
    endif()
endif()

mark_as_advanced(OPENMM_INCLUDE_DIR OPENMM_LIBRARY)
