#--- OpenMMLocalCWLD: platform-independent public API library ---------------
# LCWLD-010 creates this target with placeholder content only. LCWLD-040 adds
# LocalCWLDForce.{h,cpp} and LCWLD-050 adds the ForceImpl and kernel contract;
# both are picked up by the CONFIGURE_DEPENDS glob without editing this file.

file(GLOB_RECURSE LOCALCWLD_API_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE LOCALCWLD_API_HEADERS CONFIGURE_DEPENDS "include/openmm/*.h")

add_library(OpenMMLocalCWLD SHARED ${LOCALCWLD_API_SOURCES})
add_library(OpenMMLocalCWLD::OpenMMLocalCWLD ALIAS OpenMMLocalCWLD)

target_include_directories(OpenMMLocalCWLD PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(OpenMMLocalCWLD PUBLIC OpenMM::OpenMM)

target_compile_definitions(OpenMMLocalCWLD
    PRIVATE LOCALCWLD_BUILDING_SHARED_LIBRARY
            LOCALCWLD_VERSION_STRING="${PROJECT_VERSION}")

set_target_properties(OpenMMLocalCWLD PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    INSTALL_RPATH "$ORIGIN;${OPENMM_ROOT_DIR}/lib"
    BUILD_WITH_INSTALL_RPATH OFF)

install(TARGETS OpenMMLocalCWLD LIBRARY DESTINATION lib)
install(DIRECTORY include/openmm/ DESTINATION include/openmm
        FILES_MATCHING PATTERN "*.h")

if(BUILD_TESTING)
    file(GLOB LOCALCWLD_API_TESTS CONFIGURE_DEPENDS "tests/Test*.cpp")
    foreach(TEST_FILE ${LOCALCWLD_API_TESTS})
        get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)
        add_executable(${TEST_NAME} "${TEST_FILE}")
        target_link_libraries(${TEST_NAME} PRIVATE OpenMMLocalCWLD OpenMM::OpenMM)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()
