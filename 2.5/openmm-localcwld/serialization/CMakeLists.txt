#--- XML serialization proxy (implemented by LCWLD-060) ---------------------
file(GLOB LOCALCWLD_SERIALIZATION_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

if(NOT LOCALCWLD_SERIALIZATION_SOURCES)
    message(STATUS "  serialization       : no sources yet (LCWLD-060 pending) -- target skipped")
    return()
endif()

target_sources(OpenMMLocalCWLD PRIVATE ${LOCALCWLD_SERIALIZATION_SOURCES})
target_include_directories(OpenMMLocalCWLD PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
install(DIRECTORY include/openmm/ DESTINATION include/openmm
        FILES_MATCHING PATTERN "*.h")

if(NOT BUILD_TESTING)
    return()
endif()

file(GLOB LOCALCWLD_SERIALIZATION_TESTS CONFIGURE_DEPENDS "tests/Test*.cpp")
foreach(TEST_FILE ${LOCALCWLD_SERIALIZATION_TESTS})
    get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)
    add_executable(${TEST_NAME} "${TEST_FILE}")
    target_link_libraries(${TEST_NAME} PRIVATE OpenMMLocalCWLD OpenMM::OpenMM)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
