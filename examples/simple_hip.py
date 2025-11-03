#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Simple HIP Kernel Tracing Example

This example demonstrates how to use Nexus to trace a simple HIP kernel.
It compiles a basic vector addition kernel and captures its assembly and source code.
"""

from nexus import Nexus
import subprocess
import tempfile
import os
from pathlib import Path

def main():
    print("Tracing a simple HIP kernel...")

    # 1. Create a simple HIP kernel source
    hip_source = """
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vector_add(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    // Launch kernel
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    hipLaunchKernelGGL(vector_add, grid, block, 0, 0, d_a, d_b, d_c, N);
    hipDeviceSynchronize();

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    printf("HIP kernel executed successfully!\\n");
    return 0;
}
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix=".hip", delete=False) as f:
        f.write(hip_source)
        hip_file = Path(f.name)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        exe_file = Path(f.name)

    try:
        # 2. Compile the HIP kernel
        compile_cmd = ["hipcc", str(hip_file), "-g", "-o", str(exe_file)]
        print(f"Compiling HIP kernel: {' '.join(compile_cmd)}")
        compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if compile_result.returncode != 0:
            print(f"Compilation failed: {compile_result.stderr}")
            return 1
        print("Compilation successful.")

        # 3. Use Nexus to run and trace the compiled HIP binary
        nexus = Nexus(log_level=1)  # Set log_level to 1 for info messages
        print("\nRunning compiled HIP binary with Nexus...")
        trace = nexus.run([str(exe_file)])

        # 4. Process and display the trace
        print(f"\nCaptured {len(trace)} kernel(s):")

        for kernel in trace:
            print(f"\n{'='*80}")
            print(f"Kernel: {kernel.name}")
            print(f"Signature: {kernel.signature}")
            print(f"{'='*80}")

            print(f"\nAssembly ({len(kernel.assembly)} instructions):")
            for i, asm_line in enumerate(kernel.assembly, 1):
                print(f"  {i:3d}. {asm_line}")

            print(f"\nHIP Source ({len(kernel.hip)} lines):")
            if kernel.hip:
                # Use actual source line numbers if available, otherwise enumerate
                if kernel.lines and len(kernel.lines) == len(kernel.hip):
                    for line_no, hip_line in zip(kernel.lines, kernel.hip):
                        print(f"  {line_no:3d}. {hip_line}")
                else:
                    for i, hip_line in enumerate(kernel.hip, 1):
                        print(f"  {i:3d}. {hip_line}")
            else:
                print("  (No HIP source captured)")

        print(f"\n{'='*80}")
        print("Example completed!")

        # Also save to JSON for reference
        trace.save("hip_trace.json")
        print(f"Trace saved to hip_trace.json")
        return 0

    finally:
        # Clean up temporary files
        if hip_file.exists():
            os.remove(hip_file)
        if exe_file.exists():
            os.remove(exe_file)

if __name__ == "__main__":
    exit(main())
