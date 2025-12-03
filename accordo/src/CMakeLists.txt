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


find_package(HSA REQUIRED)

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

target_link_libraries(accordo
    PUBLIC
        spdlog::spdlog
        fmt::fmt
        hsa::hsa
    PRIVATE
)
