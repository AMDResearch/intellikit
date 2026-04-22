#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Accordo Example 03: Element-wise add (HIP baseline vs optimized)

Validates two HIP implementations of the same kernel name ``elemwise_add``.
Style matches Example 01: banners, numbered steps, temp ``hipcc -g`` build.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from accordo import Accordo

# One thread per element (simple)
BASELINE_KERNEL = r"""
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void elemwise_add(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 4096;
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, N * sizeof(float));
    hipMalloc(&d_b, N * sizeof(float));
    hipMalloc(&d_c, N * sizeof(float));

    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_a[i] = 0.001f * (float)i;
        h_b[i] = 0.002f * (float)(N - 1 - i);
    }
    hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, N * sizeof(float), hipMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    hipLaunchKernelGGL(elemwise_add, grid, block, 0, 0, d_a, d_b, d_c, N);
    hipDeviceSynchronize();

    free(h_a);
    free(h_b);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    printf("baseline ok\n");
    return 0;
}
"""

# Grid-stride loop, same kernel name, same numerical result
OPTIMIZED_KERNEL = r"""
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void elemwise_add(const float* a, const float* b, float* c, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 4096;
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, N * sizeof(float));
    hipMalloc(&d_b, N * sizeof(float));
    hipMalloc(&d_c, N * sizeof(float));

    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_a[i] = 0.001f * (float)i;
        h_b[i] = 0.002f * (float)(N - 1 - i);
    }
    hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, N * sizeof(float), hipMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / (2 * block.x));
    if (grid.x < 1) grid.x = 1;
    hipLaunchKernelGGL(elemwise_add, grid, block, 0, 0, d_a, d_b, d_c, N);
    hipDeviceSynchronize();

    free(h_a);
    free(h_b);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    printf("optimized ok\n");
    return 0;
}
"""


def compile_kernel_from_string(kernel_code: str, name: str, tmp_dir: Path) -> Path:
    source_file = tmp_dir / f"{name}.hip"
    binary_file = tmp_dir / name
    source_file.write_text(kernel_code)
    print(f"Compiling {name}...", end=" ", flush=True)
    cmd = ["hipcc", str(source_file), "-o", str(binary_file), "-O2", "-g"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FAILED")
        print(result.stderr)
        sys.exit(1)
    print("OK")
    return binary_file


def main() -> int:
    print("=" * 80)
    print("Accordo Example 03: Element-wise add (HIP baseline vs optimized)")
    print("=" * 80)
    print()

    with tempfile.TemporaryDirectory(prefix="accordo_elemwise_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Temp directory: {tmp_dir}\n")

        print("Step 1: Compiling HIP binaries...")
        baseline_bin = compile_kernel_from_string(BASELINE_KERNEL, "baseline", tmp_path)
        optimized_bin = compile_kernel_from_string(OPTIMIZED_KERNEL, "optimized", tmp_path)
        print()

        print("Step 2: Initializing Accordo validator (kernel: elemwise_add)...")
        try:
            validator = Accordo(
                binary=str(baseline_bin),
                kernel_name="elemwise_add",
                working_directory=str(tmp_path),
            )
            print(f"  Kernel arguments: {[f'{n}:{t}' for n, t in validator.kernel_args]}")
            print("  Validator ready\n")
        except Exception as e:
            print(f"Failed: {e}")
            return 1

        print("Step 3: Capturing baseline snapshot...")
        try:
            baseline_snap = validator.capture_snapshot(binary=str(baseline_bin), timeout_seconds=60)
            print(
                f"  Captured: {len(baseline_snap.arrays)} arrays in "
                f"{baseline_snap.execution_time_ms:.2f}ms\n"
            )
        except Exception as e:
            print(f"Failed: {e}")
            return 1

        print("Step 4: Capturing optimized snapshot...")
        try:
            optimized_snap = validator.capture_snapshot(
                binary=str(optimized_bin), timeout_seconds=60
            )
            print(
                f"  Captured: {len(optimized_snap.arrays)} arrays in "
                f"{optimized_snap.execution_time_ms:.2f}ms\n"
            )
        except Exception as e:
            print(f"Failed: {e}")
            return 1

        print("Step 5: Comparing snapshots...")
        result = validator.compare_snapshots(baseline_snap, optimized_snap, atol=1e-4, rtol=1e-5)

        print("=" * 80)
        if result.is_valid:
            print("VALIDATION PASSED")
            print(f"  Arrays matched: {result.num_arrays_validated}")
            print(f"  Success rate: {result.success_rate:.1f}%")
            bt = baseline_snap.execution_time_ms
            ot = optimized_snap.execution_time_ms
            if ot > 0:
                print("\nPerformance:")
                print(f"  Baseline:  {bt:.2f}ms")
                print(f"  Optimized: {ot:.2f}ms")
                print(f"  Speedup:   {bt / ot:.2f}x")
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
