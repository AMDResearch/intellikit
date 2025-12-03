#!/bin/bash
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

################################################################################
# Run script for testing Nexus with direct C++ library usage
#
# This demonstrates the environment variable usage mode (without Python API)
################################################################################

# Select which test to run
kernel="vector_add"
# kernel="vector_add_inline"
# kernel="vector_add_template"
# kernel="vector_add_thrust"

output="${kernel}_trace.json"

echo "=========================================="
echo "Running Nexus with C++ library mode"
echo "=========================================="
echo "Test kernel: $kernel"
echo "Output file: $output"
echo ""

# Get script directory and project root
script_dir=$(cd $(dirname $0) && pwd)
project_root=$(cd $script_dir/.. && pwd)

# Set up environment variables for Nexus
export HSA_TOOLS_LIB=$project_root/build/lib/libnexus.so
export NEXUS_OUTPUT_FILE=$output
export NEXUS_LOG_LEVEL=1

# Check if library exists
if [ ! -f "$HSA_TOOLS_LIB" ]; then
    echo "ERROR: libnexus.so not found at $HSA_TOOLS_LIB"
    echo "Run scripts/rebuild.sh first to build the library"
    exit 1
fi

# Run the test binary
binary=$project_root/test/$kernel

if [ ! -f "$binary" ]; then
    echo "ERROR: Test binary not found: $binary"
    echo "Available tests:"
    ls -1 $project_root/test/vector_add* 2>/dev/null || echo "  None found"
    exit 1
fi

echo "Running: $binary"
echo ""

$binary

if [ -f "$output" ]; then
    echo ""
    echo "=========================================="
    echo "Trace captured successfully!"
    echo "Output file: $output"
    echo ""
    echo "View with: cat $output | python3 -m json.tool"
    echo "Or use Python API: Nexus.load('$output')"
    echo "=========================================="
else
    echo ""
    echo "WARNING: No trace file generated (no GPU kernels executed?)"
fi
