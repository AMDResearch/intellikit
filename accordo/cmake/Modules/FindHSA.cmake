################################################################################
# MIT License

# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

# FindHSA.cmake - Locate the HSA runtime library and set up the target

# Search for the HSA include directory with priority given to /opt/rocm

file(GLOB ROCM_PATHS "/opt/rocm*/include")

find_path(HSA_INCLUDE_DIR
    NAMES hsa/hsa.h
    PATHS ${ROCM_PATHS}
    /usr/local/include
    /usr/include
    DOC "Path to HSA include directory"
)

# Search for the HSA library with priority given to /opt/rocm
find_library(HSA_LIBRARY
    NAMES hsa-runtime64
    PATHS /opt/rocm/lib
    /opt/rocm/lib64
    /usr/local/lib
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    DOC "Path to HSA runtime library"
)

message("HSA_INCLUDE_DIR: ${HSA_INCLUDE_DIR}")
message("HSA_LIBRARY: ${HSA_LIBRARY}")

# Check if both the include directory and library were found
if(HSA_INCLUDE_DIR AND HSA_LIBRARY)
    # Create the imported target hsa::hsa
    add_library(hsa::hsa INTERFACE IMPORTED)
    set_target_properties(hsa::hsa PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HSA_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${HSA_LIBRARY}"
        INTERFACE_COMPILE_DEFINITIONS "AMD_INTERNAL_BUILD"
    )

    # Print status messages
    message(STATUS "HSA include directory: ${HSA_INCLUDE_DIR}")
    message(STATUS "HSA library: ${HSA_LIBRARY}")
    set(HSA_FOUND TRUE)
else()
    # Handle errors if HSA is not found
    if(NOT HSA_INCLUDE_DIR)
        message(WARNING "HSA include directory not found.")
    endif()

    if(NOT HSA_LIBRARY)
        message(WARNING "HSA library not found.")
    endif()

    set(HSA_FOUND FALSE)
endif()

# Provide a variable to indicate whether HSA was found
mark_as_advanced(HSA_INCLUDE_DIR HSA_LIBRARY)
