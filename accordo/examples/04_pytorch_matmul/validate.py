#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Accordo Example 04: Small square matmul (HIP baseline vs optimized)

Two HIP programs share the kernel name ``matmul_nn`` (naïve k-loop vs 2-unrolled
inner k). Same host data layout N=32 row-major A, B, C.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from accordo import Accordo

BASELINE_KERNEL = r"""
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matmul_nn(const float* A, const float* B, float* C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;
    float sum = 0.f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

int main() {
    const int N = 32;
    const size_t sz = N * N * sizeof(float);
    float *dA, *dB, *dC;
    hipMalloc(&dA, sz);
    hipMalloc(&dB, sz);
    hipMalloc(&dC, sz);

    float* hA = (float*)malloc(sz);
    float* hB = (float*)malloc(sz);
    for (int i = 0; i < N * N; i++) {
        hA[i] = 0.01f * (float)(i % 17);
        hB[i] = 0.02f * (float)(i % 13);
    }
    hipMemcpy(dA, hA, sz, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, sz, hipMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    hipLaunchKernelGGL(matmul_nn, grid, block, 0, 0, dA, dB, dC, N);
    hipDeviceSynchronize();

    free(hA);
    free(hB);
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    printf("baseline matmul ok\n");
    return 0;
}
"""

OPTIMIZED_KERNEL = r"""
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matmul_nn(const float* A, const float* B, float* C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;
    float sum = 0.f;
    int k = 0;
    for (; k + 1 < N; k += 2) {
        sum += A[row * N + k] * B[k * N + col];
        sum += A[row * N + k + 1] * B[(k + 1) * N + col];
    }
    for (; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

int main() {
    const int N = 32;
    const size_t sz = N * N * sizeof(float);
    float *dA, *dB, *dC;
    hipMalloc(&dA, sz);
    hipMalloc(&dB, sz);
    hipMalloc(&dC, sz);

    float* hA = (float*)malloc(sz);
    float* hB = (float*)malloc(sz);
    for (int i = 0; i < N * N; i++) {
        hA[i] = 0.01f * (float)(i % 17);
        hB[i] = 0.02f * (float)(i % 13);
    }
    hipMemcpy(dA, hA, sz, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, sz, hipMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    hipLaunchKernelGGL(matmul_nn, grid, block, 0, 0, dA, dB, dC, N);
    hipDeviceSynchronize();

    free(hA);
    free(hB);
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    printf("optimized matmul ok\n");
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
    print("Accordo Example 04: Square matmul (HIP baseline vs optimized)")
    print("=" * 80)
    print()

    with tempfile.TemporaryDirectory(prefix="accordo_matmul_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Temp directory: {tmp_dir}\n")

        print("Step 1: Compiling HIP binaries...")
        baseline_bin = compile_kernel_from_string(BASELINE_KERNEL, "baseline", tmp_path)
        optimized_bin = compile_kernel_from_string(OPTIMIZED_KERNEL, "optimized", tmp_path)
        print()

        print("Step 2: Initializing Accordo validator (kernel: matmul_nn)...")
        try:
            validator = Accordo(
                binary=str(baseline_bin),
                kernel_name="matmul_nn",
                working_directory=str(tmp_path),
            )
            print(
                f"  Kernel arguments: {[f'{n}:{t}' for n, t in validator.kernel_args]}"
            )
            print("  Validator ready\n")
        except Exception as e:
            print(f"Failed: {e}")
            return 1

        print("Step 3: Capturing baseline snapshot...")
        try:
            baseline_snap = validator.capture_snapshot(
                binary=str(baseline_bin), timeout_seconds=60
            )
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
        result = validator.compare_snapshots(
            baseline_snap, optimized_snap, atol=1e-3, rtol=1e-4
        )

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
