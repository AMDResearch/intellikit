# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.21)

project(intellikit
    VERSION 0.1.0
    DESCRIPTION "IntelliKit: LLM-ready GPU profiling and analysis toolkit"
    LANGUAGES CXX
)

# Build options - can be controlled via environment or cmake flags
option(BUILD_NEXUS "Build Nexus tool" ON)

# Add subdirectories based on what's enabled
if(BUILD_NEXUS)
    message(STATUS "Building Nexus (C++ library)")
    add_subdirectory(nexus)
endif()

# Accordo: Pure Python with runtime C++ compilation
message(STATUS "Accordo: Pure Python package (C++ compiled at runtime)")

# Metrix: Pure Python
message(STATUS "Metrix: Pure Python package")

