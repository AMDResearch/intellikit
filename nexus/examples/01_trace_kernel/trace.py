#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Nexus Example: Kernel Tracing

This example writes a HIP kernel to /tmp, compiles it,
and uses Nexus to extract the assembly and source code.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from nexus import Nexus

# Simple HIP kernel for tracing
HIP_KERNEL = """
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vector_add(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    hipLaunchKernelGGL(vector_add, dim3(4), dim3(256), 0, 0,
                       d_a, d_b, d_c, N);
    hipDeviceSynchronize();

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    printf("Kernel executed successfully\\n");
    return 0;
}
"""


def main():
    print("=" * 80)
    print("Nexus Example: Kernel Tracing")
    print("=" * 80)
    print()

    with tempfile.TemporaryDirectory(prefix="nexus_trace_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Temp directory: {tmp_dir}")
        print()

        # Write kernel to file
        print("Step 1: Writing kernel to /tmp...")
        kernel_file = tmp_path / "vector_add.hip"
        kernel_file.write_text(HIP_KERNEL)
        print(f"  Wrote: {kernel_file}")
        print()

        # Compile kernel
        print("Step 2: Compiling kernel...")
        binary_file = tmp_path / "vector_add"
        cmd = ["hipcc", str(kernel_file), "-g", "-o", str(binary_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Compilation failed:\n{result.stderr}")
            return 1
        print(f"  Compiled: {binary_file}")
        print()

        # Trace with Nexus
        print("Step 3: Tracing with Nexus...")
        nexus = Nexus(log_level=1)

        try:
            trace = nexus.run([str(binary_file)])
            print(f"  Captured {len(trace)} kernel(s)")
            print()

            # Display trace results
            print("=" * 80)
            print("KERNEL TRACE RESULTS")
            print("=" * 80)

            for kernel in trace:
                print()
                print(f"Kernel: {kernel.name}")
                print(f"Signature: {kernel.signature}")
                print()

                print(f"Assembly ({len(kernel.assembly)} instructions):")
                for i, asm in enumerate(kernel.assembly[:10], 1):
                    print(f"  {i:2d}. {asm}")
                if len(kernel.assembly) > 10:
                    print(f"  ... ({len(kernel.assembly) - 10} more instructions)")
                print()

                print(f"HIP Source ({len(kernel.hip)} lines):")
                for i, hip_line in enumerate(kernel.hip[:10], 1):
                    line_no = kernel.lines[i - 1] if kernel.lines and i <= len(kernel.lines) else i
                    print(f"  {line_no:3d}. {hip_line}")
                if len(kernel.hip) > 10:
                    print(f"  ... ({len(kernel.hip) - 10} more lines)")
                print()

            print("=" * 80)
            print("Tracing completed successfully")
            print("=" * 80)

        except Exception as e:
            print(f"  Tracing failed: {e}")
            return 1

        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
