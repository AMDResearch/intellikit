# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Accordo: validate HIP matmul example (baseline vs 2-unrolled matmul_nn kernel)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from accordo import Accordo

VALIDATE = (
    Path(__file__).resolve().parents[1] / "examples" / "04_pytorch_matmul" / "validate.py"
)

BASELINE_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matmul_nn(const float* A, const float* B, float* C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;
    float sum = 0.f;
    for (int k = 0; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}

int main() {
    const int N = 32;
    const size_t sz = N * N * sizeof(float);
    float *dA, *dB, *dC;
    hipMalloc(&dA, sz); hipMalloc(&dB, sz); hipMalloc(&dC, sz);
    float* hA = (float*)malloc(sz);
    float* hB = (float*)malloc(sz);
    for (int i = 0; i < N * N; i++) { hA[i] = 0.01f * (i % 17); hB[i] = 0.02f * (i % 13); }
    hipMemcpy(dA, hA, sz, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, sz, hipMemcpyHostToDevice);
    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);
    hipLaunchKernelGGL(matmul_nn, grid, block, 0, 0, dA, dB, dC, N);
    hipDeviceSynchronize();
    free(hA); free(hB); hipFree(dA); hipFree(dB); hipFree(dC);
    return 0;
}
"""

OPTIMIZED_SRC = r"""
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
    for (; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}

int main() {
    const int N = 32;
    const size_t sz = N * N * sizeof(float);
    float *dA, *dB, *dC;
    hipMalloc(&dA, sz); hipMalloc(&dB, sz); hipMalloc(&dC, sz);
    float* hA = (float*)malloc(sz);
    float* hB = (float*)malloc(sz);
    for (int i = 0; i < N * N; i++) { hA[i] = 0.01f * (i % 17); hB[i] = 0.02f * (i % 13); }
    hipMemcpy(dA, hA, sz, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, sz, hipMemcpyHostToDevice);
    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);
    hipLaunchKernelGGL(matmul_nn, grid, block, 0, 0, dA, dB, dC, N);
    hipDeviceSynchronize();
    free(hA); free(hB); hipFree(dA); hipFree(dB); hipFree(dC);
    return 0;
}
"""


def _compile(src: str, name: str, tmp: Path) -> Path:
    src_file = tmp / f"{name}.hip"
    bin_file = tmp / name
    src_file.write_text(src)
    r = subprocess.run(
        ["hipcc", str(src_file), "-o", str(bin_file), "-O2", "-g"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"hipcc compile failed: {r.stderr[:200]}")
    return bin_file


@pytest.fixture(scope="module")
def require_hipcc_and_gpu():
    if shutil.which("hipcc") is None:
        pytest.skip("hipcc not on PATH")
    if shutil.which("rocprofv3") is None:
        pytest.skip("rocprofv3 not on PATH")


@pytest.fixture(scope="module")
def require_validate_script():
    if not VALIDATE.is_file():
        pytest.skip(f"Example script missing: {VALIDATE}")


def test_accordo_matmul_nn_validation(require_hipcc_and_gpu):
    """Accordo validate: baseline vs 2-unrolled matmul_nn produce identical outputs."""
    with tempfile.TemporaryDirectory(prefix="accordo_test_matmul_") as tmp_dir:
        tmp = Path(tmp_dir)
        baseline = _compile(BASELINE_SRC, "baseline", tmp)
        optimized = _compile(OPTIMIZED_SRC, "optimized", tmp)

        validator = Accordo(
            binary=str(baseline),
            kernel_name="matmul_nn",
            working_directory=str(tmp),
        )
        baseline_snap = validator.capture_snapshot(binary=str(baseline), timeout_seconds=60)
        optimized_snap = validator.capture_snapshot(binary=str(optimized), timeout_seconds=60)
        result = validator.compare_snapshots(baseline_snap, optimized_snap, atol=1e-3, rtol=1e-4)

    assert result.is_valid, f"Validation failed:\n{result.summary()}"
    assert result.num_arrays_validated >= 1


def test_accordo_matmul_validate_script_runs(require_validate_script, require_hipcc_and_gpu):
    """Smoke: validate.py example exits 0."""
    import sys as _sys
    r = subprocess.run(
        [_sys.executable, str(VALIDATE)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
