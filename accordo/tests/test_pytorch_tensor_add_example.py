# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Accordo: validate HIP tensor_add example (baseline vs optimized elemwise_add kernel)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from accordo import Accordo

VALIDATE = (
    Path(__file__).resolve().parents[1] / "examples" / "03_pytorch_tensor_add" / "validate.py"
)

BASELINE_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void elemwise_add(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

int main() {
    const int N = 4096;
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, N * sizeof(float));
    hipMalloc(&d_b, N * sizeof(float));
    hipMalloc(&d_c, N * sizeof(float));
    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) { h_a[i] = 0.001f * i; h_b[i] = 0.002f * (N - 1 - i); }
    hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, N * sizeof(float), hipMemcpyHostToDevice);
    dim3 block(256), grid((N + 255) / 256);
    hipLaunchKernelGGL(elemwise_add, grid, block, 0, 0, d_a, d_b, d_c, N);
    hipDeviceSynchronize();
    free(h_a); free(h_b); hipFree(d_a); hipFree(d_b); hipFree(d_c);
    return 0;
}
"""

OPTIMIZED_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void elemwise_add(const float* a, const float* b, float* c, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        c[i] = a[i] + b[i];
}

int main() {
    const int N = 4096;
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, N * sizeof(float));
    hipMalloc(&d_b, N * sizeof(float));
    hipMalloc(&d_c, N * sizeof(float));
    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) { h_a[i] = 0.001f * i; h_b[i] = 0.002f * (N - 1 - i); }
    hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, N * sizeof(float), hipMemcpyHostToDevice);
    dim3 block(256), grid((N + 511) / 512);
    if (grid.x < 1) grid.x = 1;
    hipLaunchKernelGGL(elemwise_add, grid, block, 0, 0, d_a, d_b, d_c, N);
    hipDeviceSynchronize();
    free(h_a); free(h_b); hipFree(d_a); hipFree(d_b); hipFree(d_c);
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


def test_accordo_elemwise_add_validation(require_hipcc_and_gpu):
    """Accordo validate: baseline vs grid-stride elemwise_add produce identical outputs."""
    with tempfile.TemporaryDirectory(prefix="accordo_test_add_") as tmp_dir:
        tmp = Path(tmp_dir)
        baseline = _compile(BASELINE_SRC, "baseline", tmp)
        optimized = _compile(OPTIMIZED_SRC, "optimized", tmp)

        validator = Accordo(
            binary=str(baseline),
            kernel_name="elemwise_add",
            working_directory=str(tmp),
        )
        baseline_snap = validator.capture_snapshot(binary=str(baseline), timeout_seconds=60)
        optimized_snap = validator.capture_snapshot(binary=str(optimized), timeout_seconds=60)
        result = validator.compare_snapshots(baseline_snap, optimized_snap, atol=1e-4, rtol=1e-5)

    assert result.is_valid, f"Validation failed:\n{result.summary()}"
    assert result.num_arrays_validated >= 1


def test_accordo_tensor_add_validate_script_runs(require_validate_script, require_hipcc_and_gpu):
    """Smoke: validate.py example exits 0."""
    import sys as _sys
    r = subprocess.run(
        [_sys.executable, str(VALIDATE)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
