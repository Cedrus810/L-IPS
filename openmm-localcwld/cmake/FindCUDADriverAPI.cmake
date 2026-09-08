#---------------------------------------------------------------------------
# FindCUDADriverAPI -- locate <cuda.h> and libcuda for the CUDA driver API.
#
# This plugin never invokes nvcc (see the CUDA block in the root
# CMakeLists.txt), so a full CUDAToolkit is not required. What IS required is
# the driver API header/library pair that OpenMM's own CudaContext.h uses:
#
#     openmm/cuda/CudaContext.h  ->  #include <cuda.h>
#
# Prefer the headers that ship next to the OpenMM installation. Mixing a
# different CUDA major version's cuda.h with OpenMM's CUDA-12 build buys
# nothing and risks type drift, since the runtime NVRTC is fixed by OpenMM.
#
# Output: CUDADriverAPI_FOUND and imported target CUDA::driver_api.
#---------------------------------------------------------------------------

# Which cuda.h to compile against. Default: the one shipped next to OpenMM,
# i.e. the CUDA major version OpenMM itself was built with (12.9 here).
#
# Staying on 12.9 is the current choice, but the CUDA 13 path is deliberately
# kept open and has been verified: compiling OpenMM's CudaContext.h against
# /opt/cuda/include (CUDA 13.3) succeeds with zero errors and the resulting
# binary runs. Point LOCALCWLD_CUDA_INCLUDE_DIR at a CUDA 13 include directory
# to build that way:
#
#     cmake ... -DLOCALCWLD_CUDA_INCLUDE_DIR=/opt/cuda/include
#
# Caveat, measured: CUDA 13 headers deprecate `double4`, which OpenMM's own
# CudaContext.h returns from getPeriodicBoxSize(). That is OpenMM's header,
# not ours, so the warning is silenced on the plugin target rather than
# "fixed". It is a warning, not an error.
set(LOCALCWLD_CUDA_INCLUDE_DIR "" CACHE PATH
    "Directory containing cuda.h. Empty = use the one shipped with OPENMM_DIR.")

set(_lcwld_header_hints "")
if(LOCALCWLD_CUDA_INCLUDE_DIR)
    list(APPEND _lcwld_header_hints "${LOCALCWLD_CUDA_INCLUDE_DIR}")
endif()
if(OPENMM_ROOT_DIR)
    list(APPEND _lcwld_header_hints
        "${OPENMM_ROOT_DIR}/targets/x86_64-linux/include"
        "${OPENMM_ROOT_DIR}/include")
endif()
if(CUDAToolkit_FOUND)
    list(APPEND _lcwld_header_hints "${CUDAToolkit_INCLUDE_DIRS}")
endif()

find_path(CUDA_DRIVER_API_INCLUDE_DIR
    NAMES cuda.h
    HINTS ${_lcwld_header_hints}
    NO_DEFAULT_PATH)
find_path(CUDA_DRIVER_API_INCLUDE_DIR NAMES cuda.h)

# libcuda comes from the NVIDIA driver, not from a toolkit. Prefer the real
# driver library over a toolkit stub: on this machine the stub belongs to a
# CUDA 13.3 toolkit while the headers above are 12.9, and there is no reason to
# link a third CUDA version into the picture when the actual driver is right
# there. A stub is only the fallback, for build hosts with no driver installed.
find_library(CUDA_DRIVER_API_LIBRARY
    NAMES cuda
    HINTS /usr/lib /usr/lib64 /lib/x86_64-linux-gnu
    NO_DEFAULT_PATH)
find_library(CUDA_DRIVER_API_LIBRARY
    NAMES cuda
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES stubs)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDADriverAPI
    REQUIRED_VARS CUDA_DRIVER_API_INCLUDE_DIR CUDA_DRIVER_API_LIBRARY
    REASON_FAILURE_MESSAGE
        "Need the CUDA driver API (cuda.h + libcuda). On a conda OpenMM these \
live in OPENMM_DIR/targets/x86_64-linux/include and /usr/lib respectively.")

if(CUDADriverAPI_FOUND AND NOT TARGET CUDA::driver_api)
    add_library(CUDA::driver_api UNKNOWN IMPORTED)
    set_target_properties(CUDA::driver_api PROPERTIES
        IMPORTED_LOCATION "${CUDA_DRIVER_API_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDA_DRIVER_API_INCLUDE_DIR}")
endif()

mark_as_advanced(CUDA_DRIVER_API_INCLUDE_DIR CUDA_DRIVER_API_LIBRARY)

# Report which CUDA the headers actually are. This is the number that decides
# what the plugin compiles against, and it is NOT necessarily the driver's
# version (nvidia-smi reports the driver, which is newer and backward
# compatible) nor whichever nvcc happens to be first on PATH.
if(CUDADriverAPI_FOUND AND EXISTS "${CUDA_DRIVER_API_INCLUDE_DIR}/cuda.h")
    file(STRINGS "${CUDA_DRIVER_API_INCLUDE_DIR}/cuda.h" _lcwld_cuda_version_line
         REGEX "^#define[ \t]+CUDA_VERSION[ \t]+[0-9]+")
    if(_lcwld_cuda_version_line)
        string(REGEX MATCH "[0-9]+" CUDA_DRIVER_API_VERSION "${_lcwld_cuda_version_line}")
        math(EXPR _lcwld_cuda_major "${CUDA_DRIVER_API_VERSION} / 1000")
        math(EXPR _lcwld_cuda_minor "(${CUDA_DRIVER_API_VERSION} % 1000) / 10")
        set(CUDA_DRIVER_API_VERSION_STRING "${_lcwld_cuda_major}.${_lcwld_cuda_minor}"
            CACHE STRING "CUDA version of the headers used to build the plugin")
        message(STATUS "  cuda headers        : ${CUDA_DRIVER_API_VERSION_STRING} "
                       "(${CUDA_DRIVER_API_INCLUDE_DIR})")
    endif()
endif()
