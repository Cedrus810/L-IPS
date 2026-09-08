#---------------------------------------------------------------------------
# OpenMM LocalCWLDForce plugin -- root build script (ticket LCWLD-010).
#
# This file owns layout, options and installation only. No physics lives here
# and none may be added here; the numerical kernels belong to LCWLD-070/100.
#
# Toolchain (DEC-001): this plugin MUST be compiled with a C++ toolchain that
# is libstdc++-ABI compatible with the OpenMM build it links against. For the
# conda `openmm_dev` environment that means the conda-forge GCC 14.x shipped
# inside the environment, NOT the system compiler. No absolute path to any
# particular user's home directory appears in this file; select the compiler
# from the command line, e.g.
#
#   export OPENMM_PREFIX=/path/to/envs/openmm_dev
#   cmake -S . -B build \
#     -DOPENMM_DIR=$OPENMM_PREFIX \
#     -DCMAKE_C_COMPILER=$OPENMM_PREFIX/bin/x86_64-conda-linux-gnu-gcc \
#     -DCMAKE_CXX_COMPILER=$OPENMM_PREFIX/bin/x86_64-conda-linux-gnu-g++
#---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)

project(OpenMMLocalCWLD VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

#--- Options ---------------------------------------------------------------
option(LOCALCWLD_BUILD_REFERENCE "Build the Reference platform plugin" OFF)
option(LOCALCWLD_BUILD_CUDA      "Build the CUDA platform plugin"      OFF)
option(LOCALCWLD_BUILD_PYTHON    "Build the SWIG Python wrapper"       OFF)
option(BUILD_TESTING             "Build the test executables"          OFF)

#--- OpenMM ----------------------------------------------------------------
find_package(OpenMM REQUIRED)

# Plugins are dlopen()ed by OpenMM from this directory. Keep it derived from
# the OpenMM installation so a staged install and a real install agree.
set(LOCALCWLD_PLUGIN_INSTALL_DIR "lib/plugins" CACHE STRING
    "Directory (relative to CMAKE_INSTALL_PREFIX) for platform plugins")

#--- Compiler-vs-OpenMM ABI sanity check -----------------------------------
# DEC-001 recorded a real hazard on this machine: the system GCC is several
# major versions newer than the one OpenMM was built with. Mismatched
# libstdc++ ABI shows up as a link error at best and a runtime crash at worst,
# so warn loudly at configure time instead of at 3am inside a production run.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND OPENMM_ROOT_DIR)
    if(EXISTS "${OPENMM_ROOT_DIR}/bin/x86_64-conda-linux-gnu-g++")
        get_filename_component(_lcwld_cxx_real "${CMAKE_CXX_COMPILER}" REALPATH)
        get_filename_component(_lcwld_conda_cxx
            "${OPENMM_ROOT_DIR}/bin/x86_64-conda-linux-gnu-g++" REALPATH)
        if(NOT _lcwld_cxx_real STREQUAL _lcwld_conda_cxx)
            message(WARNING
                "OPENMM_DIR ships its own g++ (${_lcwld_conda_cxx}) but this build "
                "uses ${CMAKE_CXX_COMPILER} (GCC ${CMAKE_CXX_COMPILER_VERSION}). "
                "OpenMM was built with the former; mixing libstdc++ ABIs can fail "
                "at link time or crash at runtime. Pass -DCMAKE_CXX_COMPILER to "
                "the compiler inside OPENMM_DIR/bin unless you know they match.")
        endif()
    endif()
endif()

# enable_testing() must run before any add_test() in a subdirectory, so it
# cannot live down with add_subdirectory(serialization).
if(BUILD_TESTING)
    enable_testing()
endif()

#--- API library ------------------------------------------------------------
add_subdirectory(openmmapi)

if(LOCALCWLD_BUILD_REFERENCE)
    add_subdirectory(platforms/reference)
endif()

if(LOCALCWLD_BUILD_CUDA)
    # Deliberately no enable_language(CUDA) and no nvcc.
    #
    # This plugin compiles no .cu file. Per plan section 21 everything under
    # platforms/cuda/src is host C++, and the device kernel is
    # platforms/common/kernels/localCWLD.cc -- a *string* that OpenMM compiles
    # at run time with the NVRTC it is linked against. On this installation
    # that is NVRTC 12.9 (libOpenMMCUDA.so -> libnvrtc.so.12), and no build
    # setting here can change which NVRTC does that job.
    #
    # So all we need is the CUDA driver API: <cuda.h> to compile against and
    # libcuda to link. The driver is CUDA 13.0 capable and is backward
    # compatible with the 12.x headers OpenMM's own CUDA plugin uses.
    #
    # Consequence worth knowing: pointing this build at a CUDA 13 toolkit
    # would NOT move the device kernels to CUDA 13. Doing that means installing
    # a CUDA-13 build of OpenMM itself. See docs/decisions/DEC-001-toolchain.md.
    find_package(CUDAToolkit QUIET)
    add_subdirectory(platforms/common)
    add_subdirectory(platforms/cuda)
endif()

if(LOCALCWLD_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# NOT gated on BUILD_TESTING. The proxy is production code: without it
# XmlSerializer throws on any System containing a LocalCWLDForce, which takes
# out checkpointing and State saving. Only its tests are gated, inside.
add_subdirectory(serialization)

#--- Summary ---------------------------------------------------------------
message(STATUS "")
message(STATUS "OpenMMLocalCWLD ${PROJECT_VERSION}")
message(STATUS "  OpenMM              : ${OPENMM_ROOT_DIR}")
message(STATUS "  build type          : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler        : ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
message(STATUS "  reference plugin    : ${LOCALCWLD_BUILD_REFERENCE}")
message(STATUS "  cuda plugin         : ${LOCALCWLD_BUILD_CUDA}")
message(STATUS "  python wrapper      : ${LOCALCWLD_BUILD_PYTHON}")
message(STATUS "  tests               : ${BUILD_TESTING}")
message(STATUS "  install prefix      : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  plugin install dir  : ${CMAKE_INSTALL_PREFIX}/${LOCALCWLD_PLUGIN_INSTALL_DIR}")
message(STATUS "")
