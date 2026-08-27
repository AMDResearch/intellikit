################################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

include(${PROJECT_SOURCE_DIR}/cmake/AccordoCompilerOptions.cmake)
message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
include(FetchContent)

#
# dependencies
#

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG        v1.11.0
)
FetchContent_MakeAvailable(spdlog)

include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)
CPMAddPackage("gh:fmtlib/fmt#7.1.3")
CPMAddPackage("gh:nlohmann/json@3.11.3")

find_package(HSA REQUIRED)

#
# ROCm 10 added hsa_amd_queue_create and HIP moved its compute queue onto it.
# The slot is absent from earlier headers, so probe for the AmdExtTable member
# we actually write rather than keying off a ROCm version.
#
# Two probes, not one. A single "does the member exist" test answers no both
# when the header lacks the member and when the header could not be included at
# all -- and the second case is a broken configuration wearing the first case's
# clothes. Separating them is the difference between compiling out a hook we do
# not need and silently compiling out one we do.
#
include(CheckCXXSourceCompiles)
get_target_property(_accordo_hsa_includes hsa::hsa INTERFACE_INCLUDE_DIRECTORIES)
set(CMAKE_REQUIRED_INCLUDES ${_accordo_hsa_includes})
set(CMAKE_REQUIRED_DEFINITIONS -DAMD_INTERNAL_BUILD)

check_cxx_source_compiles("
#include <hsa/hsa_api_trace.h>
int main() { AmdExtTable table{}; (void)table; return 0; }
" ACCORDO_HSA_API_TRACE_USABLE)

if(NOT ACCORDO_HSA_API_TRACE_USABLE)
    message(FATAL_ERROR
        "accordo: cannot compile against <hsa/hsa_api_trace.h> from "
        "'${_accordo_hsa_includes}'. Queue interception cannot be configured. "
        "Set ROCM_PATH to the ROCm tree you intend to build against.")
endif()

check_cxx_source_compiles("
#include <hsa/hsa_api_trace.h>
int main() {
  AmdExtTable table{};
  return table.hsa_amd_queue_create_fn == nullptr ? 0 : 1;
}
" ACCORDO_HAVE_HSA_AMD_QUEUE_CREATE)

unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_DEFINITIONS)
unset(_accordo_hsa_includes)

#
# accordo target
#


set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_HIP_FLAGS_DEBUG "-O0 -ggdb")


target_include_directories(accordo
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/accordo>"
)


target_sources(accordo
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/log.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/accordo.hpp>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/accordo.hip
)

intelliperf_compiler_warnings(accordo)
intelliperf_compiler_options(accordo)

if(ACCORDO_HAVE_HSA_AMD_QUEUE_CREATE)
    target_compile_definitions(accordo PRIVATE ACCORDO_HAVE_HSA_AMD_QUEUE_CREATE)
    message(STATUS "accordo: hooking hsa_amd_queue_create (ROCm 10+ compute queue path)")
else()
    message(STATUS "accordo: hsa_amd_queue_create absent, hooking hsa_queue_create only")
endif()

target_link_libraries(accordo
    PUBLIC
        spdlog::spdlog
        fmt::fmt
        hsa::hsa
        nlohmann_json::nlohmann_json
    PRIVATE
)
