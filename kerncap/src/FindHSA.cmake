# FindHSA.cmake — Locate the HSA runtime in a ROCm installation
#
# This module defines:
#   HSA_FOUND        - True if HSA was found
#   HSA_INCLUDE_DIRS - HSA header directories
#   HSA_LIBRARIES    - HSA runtime library
#   hsa::hsa         - Imported target

# Check ROCM_PATH, then /opt/rocm, then standard paths
set(_HSA_SEARCH_PATHS)
if(DEFINED ENV{ROCM_PATH})
    list(APPEND _HSA_SEARCH_PATHS "$ENV{ROCM_PATH}")
endif()
list(APPEND _HSA_SEARCH_PATHS "/opt/rocm")
list(APPEND _HSA_SEARCH_PATHS "/usr/local")
list(APPEND _HSA_SEARCH_PATHS "/usr")

find_path(HSA_INCLUDE_DIR
    NAMES hsa/hsa.h
    PATHS ${_HSA_SEARCH_PATHS}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)
# Fallback to default search
find_path(HSA_INCLUDE_DIR NAMES hsa/hsa.h)

find_library(HSA_LIBRARY
    NAMES hsa-runtime64
    PATHS ${_HSA_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)
find_library(HSA_LIBRARY NAMES hsa-runtime64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HSA
    REQUIRED_VARS HSA_LIBRARY HSA_INCLUDE_DIR
    FAIL_MESSAGE "Could not find HSA runtime. Set ROCM_PATH or ensure ROCm is installed at /opt/rocm."
)

if(HSA_FOUND)
    set(HSA_INCLUDE_DIRS ${HSA_INCLUDE_DIR})
    set(HSA_LIBRARIES ${HSA_LIBRARY})

    if(NOT TARGET hsa::hsa)
        add_library(hsa::hsa SHARED IMPORTED)
        set_target_properties(hsa::hsa PROPERTIES
            IMPORTED_LOCATION "${HSA_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${HSA_INCLUDE_DIR}"
            INTERFACE_COMPILE_DEFINITIONS "AMD_INTERNAL_BUILD"
        )
    endif()
endif()

mark_as_advanced(HSA_INCLUDE_DIR HSA_LIBRARY)
