#--- SWIG Python wrapper (LCWLD-080) ----------------------------------------
# Produces `localcwld.py` + `_localcwld.so` in the build tree. The XML route
# (LCWLD-060, python/openmm_localcwld/xml_bridge.py) stays supported: it needs
# no toolchain at all, and the two are cross-checked against each other.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/localcwld.i")
    message(STATUS "  python wrapper      : localcwld.i missing (LCWLD-080 pending) -- target skipped")
    return()
endif()

find_package(SWIG REQUIRED)
include(UseSWIG)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "  swig                : ${SWIG_EXECUTABLE} ${SWIG_VERSION}")
message(STATUS "  python              : ${Python_EXECUTABLE} ${Python_VERSION}")

# SWIG needs OpenMM's own .i files (swig/OpenMMSwigHeaders.i, swig/typemaps.i);
# they ship in the OpenMM include tree, not with SWIG.
if(NOT EXISTS "${OPENMM_ROOT_DIR}/include/swig/OpenMMSwigHeaders.i")
    message(FATAL_ERROR
        "OpenMM's SWIG inputs are missing from ${OPENMM_ROOT_DIR}/include/swig. "
        "The wrapper cannot be generated without them.")
endif()

set_property(SOURCE localcwld.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE localcwld.i PROPERTY INCLUDE_DIRECTORIES
    "${OPENMM_ROOT_DIR}/include")

swig_add_library(localcwld_python
    LANGUAGE python
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
    SOURCES localcwld.i)

set_target_properties(localcwld_python PROPERTIES
    OUTPUT_NAME "_localcwld"
    PREFIX ""
    # RPATH for the INSTALLED copy only. BUILD_WITH_INSTALL_RPATH stays OFF so
    # the build-tree extension keeps CMake's own build RPATH -- with it ON, the
    # import fails with "libOpenMMLocalCWLD.so.0: cannot open shared object
    # file", because that library sits in build-cuda/openmmapi, not $ORIGIN.
    INSTALL_RPATH "$ORIGIN;${OPENMM_ROOT_DIR}/lib"
    BUILD_WITH_INSTALL_RPATH OFF)

target_include_directories(localcwld_python PRIVATE "${OPENMM_ROOT_DIR}/include")
target_link_libraries(localcwld_python PRIVATE
    OpenMMLocalCWLD OpenMM::OpenMM Python::Module)

install(TARGETS localcwld_python LIBRARY DESTINATION .)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/localcwld.py" DESTINATION .)
