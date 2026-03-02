# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Nexus tests: inline HIP kernel, compile, run under Nexus, assert trace contents.
- Parametrized over log_level; save/load trace.
- Trace structure (iteration, kernel_names, dict-like access), Kernel attributes.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from nexus import Nexus

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
        ["hipcc", str(src), "-o", str(bin_path), "-O2", "-g"],
        capture_output=True,
        text=True,
        cwd=tmp_dir,
    )
    if r.returncode != 0:
        raise RuntimeError(f"hipcc failed:\n{r.stderr}")
    return bin_path


@pytest.mark.parametrize("log_level", [0, 1])
def test_trace_vector_add_kernel(log_level):
    """Run inline-built vector_add under Nexus; trace contains kernel with assembly and HIP."""
    with tempfile.TemporaryDirectory(prefix="nexus_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path)
        nexus = Nexus(log_level=log_level)
        trace = nexus.run([str(bin_path)], cwd=str(tmp_path))
    assert len(trace) >= 1
    # Find our kernel (trace may include runtime kernels)
    user_kernels = [k for k in trace if "vector_add" in k.name]
    assert len(user_kernels) >= 1, f"Expected kernel 'vector_add' in trace, got: {[k.name for k in trace]}"
    kernel = user_kernels[0]
    assert len(kernel.assembly) >= 1
    assert len(kernel.hip) >= 1
    assert kernel.signature != "" or "vector_add" in kernel.name


def test_trace_save_and_load():
    """Trace, save to JSON, load and assert same kernel count."""
    with tempfile.TemporaryDirectory(prefix="nexus_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path)
        nexus = Nexus(log_level=0)
        trace = nexus.run([str(bin_path)], cwd=str(tmp_path))
        out_json = tmp_path / "trace.json"
        trace.save(str(out_json))
        loaded = nexus.load(str(out_json))
    assert len(loaded) == len(trace)
    user_orig = [k for k in trace if "vector_add" in k.name]
    user_loaded = [k for k in loaded if "vector_add" in k.name]
    assert len(user_loaded) == len(user_orig)
    if user_orig and user_loaded:
        assert user_loaded[0].name == user_orig[0].name
        assert len(user_loaded[0].assembly) == len(user_orig[0].assembly)


def test_trace_run_with_output_path():
    """run(..., output=path) returns trace; when backend writes to path, load(path) matches."""
    with tempfile.TemporaryDirectory(prefix="nexus_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path)
        out_json = tmp_path / "out.json"
        nexus = Nexus(log_level=0)
        trace = nexus.run([str(bin_path)], cwd=str(tmp_path), output=str(out_json))
    assert len(trace) >= 1
    if out_json.exists():
        loaded = Nexus.load(str(out_json))
        assert len(loaded) == len(trace)
        assert loaded.kernel_names == trace.kernel_names


def test_trace_structure_and_dict_access():
    """Trace supports len(), iteration, kernel_names, and dict-like access by name."""
    with tempfile.TemporaryDirectory(prefix="nexus_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path)
        nexus = Nexus(log_level=0)
        trace = nexus.run([str(bin_path)], cwd=str(tmp_path))
    assert len(trace) >= 1
    assert trace.kernel_names
    names = list(trace.kernel_names)
    for k in trace:
        assert k.name in names
    # Dict-like access: use exact name from trace
    vector_add_name = next((n for n in trace.kernel_names if "vector_add" in n), None)
    if vector_add_name:
        kernel = trace[vector_add_name]
        assert kernel.name == vector_add_name
        assert len(kernel.assembly) >= 1
        assert "vector_add" in kernel.name


def test_trace_kernel_attributes():
    """Kernel has assembly, hip, files, lines, signature (may be empty)."""
    with tempfile.TemporaryDirectory(prefix="nexus_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path)
        nexus = Nexus(log_level=0)
        trace = nexus.run([str(bin_path)], cwd=str(tmp_path))
    user_kernels = [k for k in trace if "vector_add" in k.name]
    assert len(user_kernels) >= 1
    kernel = user_kernels[0]
    assert hasattr(kernel, "assembly")
    assert hasattr(kernel, "hip")
    assert hasattr(kernel, "files")
    assert hasattr(kernel, "lines")
    assert hasattr(kernel, "signature")
    assert isinstance(kernel.assembly, list)
    assert isinstance(kernel.hip, list)
    assert isinstance(kernel.files, list)
    assert isinstance(kernel.lines, list)
    assert len(kernel.assembly) >= 1
    assert len(kernel.hip) >= 1
