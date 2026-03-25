#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Samplex Example: Basic PC Sampling

This example writes a simple vector addition kernel to /tmp,
compiles it, and uses Samplex to find instruction-level hotspots.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

# Simple vector addition kernel
VECTOR_ADD_KERNEL = """
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vector_add(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024 * 1024;  // 1M elements
    const size_t bytes = N * sizeof(float);

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Run multiple iterations for more samples
    for (int i = 0; i < 100; i++) {
        hipLaunchKernelGGL(vector_add, dim3(gridSize), dim3(blockSize), 0, 0,
                           d_a, d_b, d_c, N);
    }
    hipDeviceSynchronize();

    printf("Vector add completed\\n");

    free(h_a);
    free(h_b);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    return 0;
}
"""


def main():
    with tempfile.TemporaryDirectory(prefix="samplex_example_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Write and compile kernel
        kernel_file = tmp_path / "vector_add.hip"
        kernel_file.write_text(VECTOR_ADD_KERNEL)

        binary_file = tmp_path / "vector_add"
        result = subprocess.run(
            ["hipcc", str(kernel_file), "-o", str(binary_file), "-O2"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}")
            return 1

        # Profile with samplex
        from samplex import Samplex
        from samplex.cli.main import format_text_output

        sampler = Samplex()
        results = sampler.sample(
            command=str(binary_file),
            kernel_filter="vector_add",
            top_n=10,
        )

        print(format_text_output(results))
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
