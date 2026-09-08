#--- Reference platform plugin (implemented by LCWLD-070) -------------------
# LCWLD-010 only wires up the target so `-DLOCALCWLD_BUILD_REFERENCE=ON`
# configures cleanly. No kernel sources exist yet.

file(GLOB LOCALCWLD_REFERENCE_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

if(NOT LOCALCWLD_REFERENCE_SOURCES)
    message(STATUS "  reference plugin    : no sources yet (LCWLD-070 pending) -- target skipped")
    return()
endif()

add_library(OpenMMLocalCWLDReference SHARED ${LOCALCWLD_REFERENCE_SOURCES})
target_link_libraries(OpenMMLocalCWLDReference PRIVATE OpenMMLocalCWLD OpenMM::OpenMM)
target_include_directories(OpenMMLocalCWLDReference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${OPENMM_INCLUDE_DIR})
set_target_properties(OpenMMLocalCWLDReference PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ORIGIN/..;${OPENMM_ROOT_DIR}/lib")
install(TARGETS OpenMMLocalCWLDReference LIBRARY DESTINATION ${LOCALCWLD_PLUGIN_INSTALL_DIR})

if(BUILD_TESTING)
    file(GLOB LOCALCWLD_REFERENCE_TESTS CONFIGURE_DEPENDS "tests/Test*.cpp")
    foreach(TEST_FILE ${LOCALCWLD_REFERENCE_TESTS})
        get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)
        add_executable(${TEST_NAME} "${TEST_FILE}")
        target_link_libraries(${TEST_NAME} PRIVATE OpenMMLocalCWLD OpenMM::OpenMM)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()
