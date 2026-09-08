#--- Common-compute host layer + encoded device kernels ---------------------
# Implemented by LCWLD-090 (host) and LCWLD-100 (kernels). LCWLD-010 only
# provides the encode step so the .cc sources become a linkable string table.

include(EncodeKernelFiles)

file(GLOB LOCALCWLD_COMMON_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB LOCALCWLD_DEVICE_KERNELS CONFIGURE_DEPENDS "kernels/*.cc")

if(NOT LOCALCWLD_COMMON_SOURCES)
    message(STATUS "  common host layer   : no sources yet (LCWLD-090 pending) -- target skipped")
    return()
endif()

if(LOCALCWLD_DEVICE_KERNELS)
    localcwld_encode_kernels(LOCALCWLD_KERNELS_H LOCALCWLD_KERNELS_CPP
                             ${LOCALCWLD_DEVICE_KERNELS})
else()
    set(LOCALCWLD_KERNELS_CPP "")
endif()

add_library(OpenMMLocalCWLDCommon STATIC
    ${LOCALCWLD_COMMON_SOURCES} ${LOCALCWLD_KERNELS_CPP})
target_include_directories(OpenMMLocalCWLDCommon PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(OpenMMLocalCWLDCommon PUBLIC OpenMMLocalCWLD OpenMM::OpenMM)
