#--- CUDA platform plugin (implemented by LCWLD-110) ------------------------
# No nvcc, no .cu files, no enable_language(CUDA): the device kernel is a
# string compiled at run time by the NVRTC that OpenMM is linked against
# (NVRTC 12.9 on this installation). All this target needs is the CUDA driver
# API. See the CUDA block in the root CMakeLists.txt for the reasoning.

find_package(CUDADriverAPI REQUIRED)

# The common-compute layer (ComputeContext, ComputeForceInfo, ComputeArray...)
# lives inside OpenMM's own CUDA *platform plugin*, not in libOpenMM. Linking
# only libOpenMM builds fine and then fails at dlopen() with
# `undefined symbol: _ZTIN6OpenMM16ComputeForceInfoE` -- a link error deferred
# to run time, which is exactly the kind that wastes an afternoon.
find_library(OPENMM_CUDA_LIBRARY
    NAMES OpenMMCUDA
    HINTS "${OPENMM_ROOT_DIR}/lib/plugins" "${OPENMM_ROOT_DIR}/lib"
    NO_DEFAULT_PATH)
if(NOT OPENMM_CUDA_LIBRARY)
    message(FATAL_ERROR
        "Could not find libOpenMMCUDA in ${OPENMM_ROOT_DIR}/lib/plugins. "
        "It provides OpenMM's common-compute layer, which this plugin links against.")
endif()
message(STATUS "  openmm cuda plugin  : ${OPENMM_CUDA_LIBRARY}")

file(GLOB LOCALCWLD_CUDA_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

if(NOT LOCALCWLD_CUDA_SOURCES)
    message(STATUS "  cuda plugin         : no sources yet (LCWLD-110 pending) -- target skipped")
    return()
endif()

add_library(OpenMMLocalCWLDCUDA SHARED ${LOCALCWLD_CUDA_SOURCES})
target_link_libraries(OpenMMLocalCWLDCUDA
    PRIVATE OpenMMLocalCWLDCommon OpenMMLocalCWLD OpenMM::OpenMM
            ${OPENMM_CUDA_LIBRARY} CUDA::driver_api)
# OpenMM's CUDA platform headers live under include/openmm/cuda and include
# their siblings unqualified, so that directory has to be on the search path.
target_include_directories(OpenMMLocalCWLDCUDA PRIVATE
    ${OPENMM_INCLUDE_DIR}
    ${OPENMM_INCLUDE_DIR}/openmm/cuda
    ${OPENMM_INCLUDE_DIR}/openmm/common)
# CUDA 13 headers deprecate double4, which OpenMM's own CudaContext.h returns
# from getPeriodicBoxSize(). Not our header and not fixable here, so silence it
# rather than let a CUDA 13 build drown in warnings we cannot act on. Verified:
# with CUDA 13.3 headers the build has zero errors and the binary runs.
if(CUDA_DRIVER_API_VERSION_STRING AND CUDA_DRIVER_API_VERSION_STRING VERSION_GREATER_EQUAL 13.0)
    target_compile_options(OpenMMLocalCWLDCUDA PRIVATE -Wno-deprecated-declarations)
endif()

set_target_properties(OpenMMLocalCWLDCUDA PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ORIGIN/..;${OPENMM_ROOT_DIR}/lib")
install(TARGETS OpenMMLocalCWLDCUDA LIBRARY DESTINATION ${LOCALCWLD_PLUGIN_INSTALL_DIR})

# Benchmarks are built but deliberately NOT registered with ctest: they call
# Integrator.step(), i.e. they run MD, and must be launched deliberately rather
# than swept up by a test run. Same glob directory, different prefix.
file(GLOB LOCALCWLD_CUDA_BENCHMARKS CONFIGURE_DEPENDS "tests/Benchmark*.cpp")
foreach(BENCH_FILE ${LOCALCWLD_CUDA_BENCHMARKS})
    get_filename_component(BENCH_NAME "${BENCH_FILE}" NAME_WE)
    add_executable(${BENCH_NAME} "${BENCH_FILE}")
    target_link_libraries(${BENCH_NAME} PRIVATE OpenMMLocalCWLD OpenMM::OpenMM)
endforeach()

if(BUILD_TESTING)
    file(GLOB LOCALCWLD_CUDA_TESTS CONFIGURE_DEPENDS "tests/Test*.cpp")
    foreach(TEST_FILE ${LOCALCWLD_CUDA_TESTS})
        get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)
        add_executable(${TEST_NAME} "${TEST_FILE}")
        # Deliberately NOT linking libOpenMMCUDA here. The test only uses the
        # public API (LocalCWLDForce, Context, Platform); the CUDA platform
        # reaches it by being dlopen()ed at run time. Linking it directly *and*
        # having OpenMM dlopen it gets the library initialised twice and the
        # process dies with `double free or corruption` at exit -- after all
        # assertions have passed, which makes it look like a teardown mystery
        # rather than a link-line mistake. Only the plugin target links it.
        target_link_libraries(${TEST_NAME} PRIVATE OpenMMLocalCWLD OpenMM::OpenMM)
        # Two things the test cannot discover on its own: where OpenMM's own
        # plugins live (the path compiled into libOpenMM points at
        # /usr/local/openmm, which does not exist in a conda env), and where our
        # freshly built plugin is.
        # The acceptance test needs a dump prefix as well. Without it the test
        # returns 0 and ctest reports a pass -- a green light for a test that
        # never ran, which is worse than a red one. Pass the prefix when the
        # dump exists so the run is real; when it does not, the test says
        # "skipped" on its own line rather than pretending.
        set(LOCALCWLD_DUMP_PREFIX
            "${CMAKE_SOURCE_DIR}/docs/reports/dec005_1aay_dpolar")
        if(TEST_NAME MATCHES "TestCudaLocalCWLD(Acceptance|Pathological|Dynamics)"
           AND EXISTS "${LOCALCWLD_DUMP_PREFIX}_system.xml")
            add_test(NAME ${TEST_NAME}
                     COMMAND ${TEST_NAME} "$<TARGET_FILE_DIR:OpenMMLocalCWLDCUDA>"
                                          "${LOCALCWLD_DUMP_PREFIX}")
        else()
            add_test(NAME ${TEST_NAME}
                     COMMAND ${TEST_NAME} "$<TARGET_FILE_DIR:OpenMMLocalCWLDCUDA>")
        endif()
        set_tests_properties(${TEST_NAME} PROPERTIES
            ENVIRONMENT "OPENMM_PLUGIN_DIR=${OPENMM_ROOT_DIR}/lib/plugins")
    endforeach()
endif()
