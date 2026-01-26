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

include(${PROJECT_SOURCE_DIR}/cmake/NexusCompilerOptions.cmake)
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

# Disable install rules for kernelDB dependency
set(KERNELDB_SKIP_INSTALL ON CACHE BOOL "Skip kernelDB install rules" FORCE)

CPMAddPackage(
    NAME kernelDB
    GIT_REPOSITORY https://github.com/AMDResearch/kerneldb.git
    GIT_TAG bb5d9d1bdc7433d81e04b194adba2c0158520a38
    EXCLUDE_FROM_ALL YES
)

CPMAddPackage("gh:nlohmann/json@3.11.3")

find_package(HSA REQUIRED)

#
# nexus target
#


set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_HIP_FLAGS_DEBUG "-O0 -ggdb")


target_include_directories(nexus
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/csrc/include>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>"
)


target_sources(nexus
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/csrc/include/nexus/log.hpp>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/csrc/include/nexus/nexus.hpp>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/nexus.cpp
)

nexus_compiler_warnings(nexus)
nexus_compiler_options(nexus)

target_link_libraries(nexus
    PUBLIC
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        kernelDB64
        fmt::fmt
        hsa::hsa
    PRIVATE
)
