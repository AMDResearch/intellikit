################################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
################################################################################

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(
    proboscis
    VERSION       0.1.0
    DESCRIPTION   "Agent-driven GPU kernel instrumentation"
    LANGUAGES     CXX C
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Disable in-source builds
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not supported.")
endif()

# Default build type
set(DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}'")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Build type" FORCE)
endif()

# ROCm path
if (NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation path")
endif()

# Find HSA runtime
find_library(HSA_RUNTIME_LIB hsa-runtime64
    PATHS ${ROCM_PATH}/lib
    REQUIRED
)
find_path(HSA_INCLUDE_DIR hsa/hsa.h
    PATHS ${ROCM_PATH}/include
    REQUIRED
)

# Build libproboscis.so
add_library(proboscis SHARED)
add_library(proboscis::proboscis ALIAS proboscis)

target_include_directories(proboscis
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc/include>"
        "$<BUILD_INTERFACE:${HSA_INCLUDE_DIR}>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/src"
)

target_link_libraries(proboscis
    PRIVATE
        ${HSA_RUNTIME_LIB}
)

set_target_properties(proboscis
    PROPERTIES
        CXX_STANDARD                17
        CXX_STANDARD_REQUIRED       ON
        CXX_EXTENSIONS              OFF
        CXX_VISIBILITY_PRESET       hidden
        VISIBILITY_INLINES_HIDDEN   ON
        POSITION_INDEPENDENT_CODE   ON
        VERSION                     "${PROJECT_VERSION}"
        SOVERSION                   "${PROJECT_VERSION_MAJOR}"
        OUTPUT_NAME                 proboscis
        SKIP_BUILD_RPATH            FALSE
        BUILD_WITH_INSTALL_RPATH    TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
        BUILD_RPATH                 "\$ORIGIN"
        INSTALL_RPATH               "\$ORIGIN"
        ARCHIVE_OUTPUT_DIRECTORY    ${PROJECT_BINARY_DIR}/lib
        LIBRARY_OUTPUT_DIRECTORY    ${PROJECT_BINARY_DIR}/lib
        RUNTIME_OUTPUT_DIRECTORY    ${PROJECT_BINARY_DIR}/bin
)

# Install into Python package directory for scikit-build-core
if(DEFINED SKBUILD_PLATLIB_DIR)
    set(PROBOSCIS_PKG_DEST "${SKBUILD_PLATLIB_DIR}/proboscis")
else()
    set(PROBOSCIS_PKG_DEST "proboscis")
endif()
install(TARGETS proboscis
    LIBRARY DESTINATION ${PROBOSCIS_PKG_DEST}
    RUNTIME DESTINATION ${PROBOSCIS_PKG_DEST}
)

# Sources
add_subdirectory(csrc/src)
