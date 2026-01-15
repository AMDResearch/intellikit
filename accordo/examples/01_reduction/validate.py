#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Accordo Example: Reduction Kernel Validation

This example creates baseline and optimized reduction kernels in /tmp,
compiles them, and uses Accordo to validate they produce identical results.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from accordo import Accordo

# Baseline reduction kernel - simple atomic approach
BASELINE_KERNEL = """
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void reduce_sum(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(output, input[idx]);
    }
}

int main() {
    const int N = 1048576;  // 1M elements

    float *d_input, *d_output;
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, sizeof(float));

    float* h_input = (float*)malloc(N * sizeof(float));
    float h_output = 0.0f;
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_output, &h_output, sizeof(float), hipMemcpyHostToDevice);

    int gridSize = (N + 255) / 256;
    hipLaunchKernelGGL(reduce_sum, dim3(gridSize), dim3(256), 0, 0,
                       d_input, d_output, N);
    hipDeviceSynchronize();

    hipMemcpy(&h_output, d_output, sizeof(float), hipMemcpyDeviceToHost);
    printf("Baseline: %.2f\\n", h_output);

    free(h_input);
    hipFree(d_input);
    hipFree(d_output);
    return 0;
}
"""

# Optimized reduction kernel - uses shared memory
OPTIMIZED_KERNEL = """
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void reduce_sum(const float* input, float* output, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

int main() {
    const int N = 1048576;  // 1M elements

    float *d_input, *d_output;
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, sizeof(float));

    float* h_input = (float*)malloc(N * sizeof(float));
    float h_output = 0.0f;
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_output, &h_output, sizeof(float), hipMemcpyHostToDevice);

    int gridSize = (N + 255) / 256;
    hipLaunchKernelGGL(reduce_sum, dim3(gridSize), dim3(256), 0, 0,
                       d_input, d_output, N);
    hipDeviceSynchronize();

    hipMemcpy(&h_output, d_output, sizeof(float), hipMemcpyDeviceToHost);
    printf("Optimized: %.2f\\n", h_output);

    free(h_input);
    hipFree(d_input);
    hipFree(d_output);
    return 0;
}
"""


def compile_kernel_from_string(kernel_code, name, tmp_dir):
    """Write kernel to /tmp, compile it, return binary path."""
    source_file = tmp_dir / f"{name}.hip"
    binary_file = tmp_dir / name

    source_file.write_text(kernel_code)
    print(f"Compiling {name}...", end=" ")

    # Important: -g flag enables debug symbols for kernelDB auto-extraction
    cmd = ["hipcc", str(source_file), "-o", str(binary_file), "-O2", "-g"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("FAILED")
        print(result.stderr)
        sys.exit(1)

    print("OK")
    return binary_file


def main():
    print("=" * 80)
    print("Accordo Example: Reduction Kernel Validation")
    print("=" * 80)
    print()

    # Create temp directory for kernels
    with tempfile.TemporaryDirectory(prefix="accordo_reduction_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Temp directory: {tmp_dir}")
        print()

        # Write and compile kernels
        print("Step 1: Compiling kernels in /tmp...")
        baseline_bin = compile_kernel_from_string(BASELINE_KERNEL, "baseline", tmp_path)
        optimized_bin = compile_kernel_from_string(OPTIMIZED_KERNEL, "optimized", tmp_path)
        print()

        # Initialize Accordo validator for the reduce_sum kernel
        print("Step 2: Initializing Accordo validator...")
        print(f"  Binary: {baseline_bin}")
        print(f"  Kernel: reduce_sum")
        try:
            validator = Accordo(
                binary=str(baseline_bin),
                kernel_name="reduce_sum",
                working_directory=str(tmp_path)
            )
            print(f"  Kernel arguments: {[f'{name}:{type}' for name, type in validator.kernel_args]}")
            print("  Validator ready")
        except Exception as e:
            print(f"Failed: {e}")
            return 1
        print()

        # Capture baseline
        print("Step 3: Capturing baseline snapshot...")
        try:
            baseline_snap = validator.capture_snapshot(
                binary=str(baseline_bin),
                timeout_seconds=30
            )
            print(f"  Captured: {len(baseline_snap.arrays)} arrays in {baseline_snap.execution_time_ms:.2f}ms")
        except Exception as e:
            print(f"Failed: {e}")
            return 1
        print()

        # Capture optimized
        print("Step 4: Capturing optimized snapshot...")
        try:
            optimized_snap = validator.capture_snapshot(
                binary=str(optimized_bin),
                timeout_seconds=30
            )
            print(f"  Captured: {len(optimized_snap.arrays)} arrays in {optimized_snap.execution_time_ms:.2f}ms")
        except Exception as e:
            print(f"Failed: {e}")
            return 1
        print()

        # Validate
        print("Step 5: Validating correctness...")
        result = validator.compare_snapshots(baseline_snap, optimized_snap, tolerance=1e-4)

        print("=" * 80)
        if result.is_valid:
            print("VALIDATION PASSED")
            print(f"  Arrays matched: {result.num_arrays_validated}")
            print(f"  Success rate: {result.success_rate:.1f}%")

            baseline_time = baseline_snap.execution_time_ms
            optimized_time = optimized_snap.execution_time_ms
            speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0

            print()
            print("Performance:")
            print(f"  Baseline:  {baseline_time:.2f}ms")
            print(f"  Optimized: {optimized_time:.2f}ms")
            print(f"  Speedup:   {speedup:.2f}x")
        else:
            print("VALIDATION FAILED")
            print(result.summary())
            return 1

        print("=" * 80)
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
