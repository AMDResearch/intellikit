find_package(hsa-runtime64 REQUIRED HINTS ${ROCM_PATH})
find_package(hip REQUIRED HINTS ${ROCM_PATH})
find_package(rocprofiler-sdk REQUIRED HINTS ${ROCM_PATH})
find_package(amd_comgr REQUIRED HINTS ${ROCM_PATH})

# Vendored header-only dependencies
set(VENDOR_DIR "${CMAKE_SOURCE_DIR}/vendor")

add_library(kerncap SHARED
    kerncap.hip
    kernarg_metadata.cpp
)

set_target_properties(kerncap PROPERTIES
    OUTPUT_NAME "kerncap"
    PREFIX "lib"
    CXX_STANDARD 20
    HIP_STANDARD 20
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
)

target_include_directories(kerncap PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${VENDOR_DIR}"
)

target_link_libraries(kerncap PRIVATE
    hsa-runtime64::hsa-runtime64
    rocprofiler-sdk::rocprofiler-sdk
    amd_comgr
)

target_compile_definitions(kerncap PRIVATE AMD_INTERNAL_BUILD)

#
# ROCm 10 added hsa_amd_queue_create and HIP creates its compute queue through
# it. The slot is absent from earlier headers, so probe for the AmdExtTable
# member we actually write rather than keying off a ROCm version.
#
# Two probes, not one. Asking only "does the member exist" answers no both when
# the header lacks it and when the header could not be included at all, and the
# second is a broken configuration wearing the first one's clothes.
#
include(CheckCXXSourceCompiles)
get_target_property(_kerncap_hsa_inc hsa-runtime64::hsa-runtime64
                    INTERFACE_INCLUDE_DIRECTORIES)
set(CMAKE_REQUIRED_INCLUDES ${_kerncap_hsa_inc})
set(CMAKE_REQUIRED_DEFINITIONS -DAMD_INTERNAL_BUILD)

check_cxx_source_compiles("
#include <hsa/hsa_api_trace.h>
int main() { AmdExtTable t{}; (void)t; return 0; }
" KERNCAP_HSA_API_TRACE_USABLE)

if(NOT KERNCAP_HSA_API_TRACE_USABLE)
    message(FATAL_ERROR
        "kerncap: cannot compile against <hsa/hsa_api_trace.h> from "
        "'${_kerncap_hsa_inc}'. Queue interception cannot be configured. "
        "Set ROCM_PATH to the ROCm tree you intend to build against.")
endif()

check_cxx_source_compiles("
#include <hsa/hsa_api_trace.h>
int main() {
  AmdExtTable t{};
  return t.hsa_amd_queue_create_fn == nullptr ? 0 : 1;
}
" KERNCAP_HAVE_HSA_AMD_QUEUE_CREATE)

unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_DEFINITIONS)
unset(_kerncap_hsa_inc)

if(KERNCAP_HAVE_HSA_AMD_QUEUE_CREATE)
    target_compile_definitions(kerncap PRIVATE KERNCAP_HAVE_HSA_AMD_QUEUE_CREATE)
    message(STATUS "kerncap: hooking hsa_amd_queue_create (ROCm 10+ compute queue path)")
else()
    message(STATUS "kerncap: hsa_amd_queue_create absent, hooking hsa_queue_create only")
endif()

target_compile_options(kerncap PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Debug>:-O0 -ggdb>
    $<$<CONFIG:Release>:-O2>
)

install(TARGETS kerncap
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

# kerncap-replay: VA-faithful HSA kernel replay binary
add_executable(kerncap-replay replay.cpp)

set_target_properties(kerncap-replay PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(kerncap-replay PRIVATE
    "${VENDOR_DIR}"
)

target_link_libraries(kerncap-replay PRIVATE
    hsa-runtime64::hsa-runtime64
    hip::host
)

target_compile_definitions(kerncap-replay PRIVATE AMD_INTERNAL_BUILD)

# Allow unresolved symbols from hsa-runtime64's transitive dependencies
# (drm, elf, numa) which are resolved at runtime but may not be in the
# build-time linker's search path (e.g., conda's compiler_compat ld).
target_link_options(kerncap-replay PRIVATE "LINKER:--allow-shlib-undefined")

target_compile_options(kerncap-replay PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Debug>:-O0 -ggdb>
    $<$<CONFIG:Release>:-O2>
)

install(TARGETS kerncap-replay RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
