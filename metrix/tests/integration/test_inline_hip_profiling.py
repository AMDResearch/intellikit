# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Metrix integration tests: inline HIP kernel, compile, profile via API.
- Parametrized over time_only, num_replays, single metrics.
- Profile presets (quick, memory), kernel_filter, results structure.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from metrix import Metrix

VECTOR_ADD_HIP = """
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vector_add(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }
    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(vector_add, dim3((N + 255) / 256), dim3(256), 0, 0, d_a, d_b, d_c, N);
    hipDeviceSynchronize();
    free(h_a);
    free(h_b);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    return 0;
}
"""


def _compile_hip(kernel_code: str, name: str, tmp_dir: Path) -> Path:
    """Write HIP source, compile with hipcc, return path to binary."""
    src = tmp_dir / f"{name}.hip"
    bin_path = tmp_dir / name
    src.write_text(kernel_code)
    r = subprocess.run(
        ["hipcc", str(src), "-o", str(bin_path), "-O2"],
        capture_output=True,
        text=True,
        cwd=tmp_dir,
    )
    if r.returncode != 0:
        raise RuntimeError(f"hipcc failed:\n{r.stderr}")
    return bin_path


@pytest.mark.parametrize("time_only", [True, False])
@pytest.mark.parametrize("num_replays", [1, 2])
def test_profile_vector_add_time_only(time_only, num_replays):
    """Profile inline-built vector_add: time_only and num_replays."""
    with tempfile.TemporaryDirectory(prefix="metrix_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path)
        profiler = Metrix()
        results = profiler.profile(
            command=str(bin_path),
            time_only=time_only,
            num_replays=num_replays,
            cwd=str(tmp_path),
            timeout_seconds=60,
        )
    assert results.total_kernels >= 1
    assert len(results.kernels) >= 1
    user_kernels = [k for k in results.kernels if "vector_add" in k.name]
    assert len(user_kernels) >= 1, f"Expected 'vector_add' in kernel names: {[k.name for k in results.kernels]}"
    kernel = user_kernels[0]
    assert kernel.duration_us.avg >= 0
    if num_replays > 1:
        assert hasattr(kernel.duration_us, "min") and hasattr(kernel.duration_us, "max")
    if not time_only:
        assert isinstance(kernel.metrics, dict)


@pytest.mark.parametrize("metric", ["memory.l2_hit_rate", "memory.coalescing_efficiency"])
def test_profile_vector_add_single_metric(metric):
    """Profile with a single metric."""
    with tempfile.TemporaryDirectory(prefix="metrix_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path)
        profiler = Metrix()
        results = profiler.profile(
            command=str(bin_path),
            metrics=[metric],
            num_replays=1,
            cwd=str(tmp_path),
            timeout_seconds=60,
        )
    assert results.total_kernels >= 1
    user_kernels = [k for k in results.kernels if "vector_add" in k.name]
    assert len(user_kernels) >= 1, f"Expected 'vector_add' in kernel names: {[k.name for k in results.kernels]}"
    kernel = user_kernels[0]
    if metric in kernel.metrics:
        assert kernel.metrics[metric].avg >= 0


def test_profile_results_structure():
    """ProfilingResults has command, kernels list, total_kernels."""
    with tempfile.TemporaryDirectory(prefix="metrix_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path)
        profiler = Metrix()
        results = profiler.profile(
            command=str(bin_path),
            time_only=True,
            num_replays=1,
            cwd=str(tmp_path),
            timeout_seconds=60,
        )
    assert hasattr(results, "command")
    assert results.command == str(bin_path)
    assert hasattr(results, "kernels")
    assert isinstance(results.kernels, list)
    assert hasattr(results, "total_kernels")
    assert results.total_kernels >= 1


@pytest.mark.parametrize("profile_name", ["quick", "memory"])
def test_profile_with_preset(profile_name):
    """Profile using preset (quick or memory) collects metrics."""
    with tempfile.TemporaryDirectory(prefix="metrix_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path)
        profiler = Metrix()
        results = profiler.profile(
            command=str(bin_path),
            profile=profile_name,
            num_replays=1,
            cwd=str(tmp_path),
            timeout_seconds=90,
        )
    assert results.total_kernels >= 1
    user_kernels = [k for k in results.kernels if "vector_add" in k.name]
    assert len(user_kernels) >= 1
    kernel = user_kernels[0]
    assert isinstance(kernel.metrics, dict)
    assert len(kernel.metrics) >= 1


def test_profile_kernel_filter_api():
    """profile() accepts kernel_filter; result may be filtered by backend if supported."""
    with tempfile.TemporaryDirectory(prefix="metrix_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path)
        profiler = Metrix()
        try:
            results = profiler.profile(
                command=str(bin_path),
                time_only=True,
                kernel_filter="vector_add",
                num_replays=1,
                cwd=str(tmp_path),
                timeout_seconds=60,
            )
            assert results.total_kernels >= 1
            if results.kernels:
                for k in results.kernels:
                    assert "vector_add" in k.name
        except RuntimeError as e:
            if "kernel" in str(e).lower() and "unrecognized" in str(e).lower():
                pytest.skip("Backend does not support kernel_filter")
            raise
