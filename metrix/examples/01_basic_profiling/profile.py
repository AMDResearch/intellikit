#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Metrix Example: Basic GPU Kernel Profiling

This example writes a simple vector addition kernel to /tmp,
compiles it, and uses Metrix to profile its performance.
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

    hipLaunchKernelGGL(vector_add, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_a, d_b, d_c, N);
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
    print("=" * 80)
    print("Metrix Example: Basic GPU Kernel Profiling")
    print("=" * 80)
    print()

    with tempfile.TemporaryDirectory(prefix="metrix_example_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Temp directory: {tmp_dir}")
        print()

        # Write kernel to file
        print("Step 1: Writing kernel to /tmp...")
        kernel_file = tmp_path / "vector_add.hip"
        kernel_file.write_text(VECTOR_ADD_KERNEL)
        print(f"  Wrote: {kernel_file}")
        print()

        # Compile kernel
        print("Step 2: Compiling kernel...")
        binary_file = tmp_path / "vector_add"
        cmd = ["hipcc", str(kernel_file), "-o", str(binary_file), "-O2"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Compilation failed:\n{result.stderr}")
            return 1
        print(f"  Compiled: {binary_file}")
        print()

        # Profile with metrix
        print("Step 3: Profiling with Metrix...")

        try:
            from metrix import Metrix

            # Initialize profiler (auto-detects GPU architecture)
            profiler = Metrix()

            # Select a few key metrics to display
            metrics_to_collect = [
                "memory.hbm_bandwidth_utilization",
                "memory.l2_hit_rate",
                "memory.coalescing_efficiency",
                "compute.total_flops",
            ]

            print(f"  Running: {binary_file}")
            results = profiler.profile(
                command=str(binary_file), metrics=metrics_to_collect, cwd=str(tmp_path)
            )

            print()
            print("=" * 80)
            print("GPU PERFORMANCE METRICS")
            print("=" * 80)

            for kernel in results.kernels:
                print(f"\nKernel: {kernel.name}")
                print(f"  Duration: {kernel.duration_us.avg:.2f} μs")

                # Display metrics
                for metric_name, stats in kernel.metrics.items():
                    print(f"  {metric_name}: {stats.avg:.2f}")

            print("=" * 80)

        except Exception as e:
            print(f"  Metrix profiling failed: {e}")
            print("  Running kernel directly to verify compilation...")
            result = subprocess.run([str(binary_file)], capture_output=True, text=True)
            print(result.stdout)
            if result.returncode == 0:
                print("  Kernel executed successfully")

        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
