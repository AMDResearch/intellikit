# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Shared fixtures and skip guards for Omniprobe integration tests.

These tests exercise the real analysis path: a HIP kernel is compiled and then
analyzed by the installed Omniprobe tool. That is deliberately the one thing the
unit tests cannot do -- they mock the upstream subprocess, so they prove the
conversion layer and nothing about whether Omniprobe itself produces usable
output.

Two constraints shape how the guards below are written.

1. **Skips are declared per test, not per module.** A module-level
   ``pytest.importorskip`` or ``pytest.skip(allow_module_level=True)`` prevents
   the tests from being *collected* at all, which makes "the integration suite
   exists" and "the integration suite is empty" look identical to any tool that
   counts tests. Declaring the guards as ``pytest.mark.skipif`` constants keeps
   the tests collectable everywhere and skipped where they cannot run.

2. **Nothing upstream is imported at module scope.** ``omniprobe.api`` belongs to
   the separate Omniprobe installation, not to this package, and importing it at
   the top of a test module would turn a missing prerequisite into a collection
   error rather than a skip.
"""

import importlib.util
import os
import shutil
import subprocess

import pytest

# A HIP program with two kernels that do identical arithmetic and differ only in
# indexing, so a memory analysis has something unambiguous to find. Kept in sync
# with omniprobe/examples/01_memory_coalescing/analyze_memory.py.
STRIDED_KERNEL = """
#include <hip/hip_runtime.h>
#include <stdio.h>

#define STRIDE 16

// Adjacent threads touch adjacent elements: one cache line per wavefront.
__global__ void coalesced_copy(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

// Adjacent threads touch elements STRIDE apart: one cache line per thread.
__global__ void strided_copy(const float* in, float* out, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * STRIDE;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

int main() {
    const int N = 1024 * 1024;
    const size_t bytes = N * sizeof(float);

    float *d_in, *d_out;
    hipMalloc(&d_in, bytes);
    hipMalloc(&d_out, bytes);

    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;
    }
    hipMemcpy(d_in, h_in, bytes, hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(coalesced_copy, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_in, d_out, N);
    hipLaunchKernelGGL(strided_copy, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_in, d_out, N);
    hipDeviceSynchronize();

    printf("Both kernels completed\\n");

    free(h_in);
    hipFree(d_in);
    hipFree(d_out);

    return 0;
}
"""


def _has_omniprobe_binary() -> bool:
    """Check whether the omniprobe executable is on PATH."""
    return shutil.which("omniprobe") is not None


def _has_upstream_api() -> bool:
    """Check whether Omniprobe's own Python API is importable.

    ``find_spec`` is used rather than a real import so that merely asking the
    question cannot execute upstream module code during collection.
    """
    try:
        return importlib.util.find_spec("omniprobe.api") is not None
    except (ImportError, ValueError):
        return False


def _has_amd_gpu() -> bool:
    """Check whether an AMD GPU is accessible via /dev/kfd."""
    return os.path.exists("/dev/kfd")


def _has_hipcc() -> bool:
    """Check whether hipcc is available to build the test kernel."""
    return shutil.which("hipcc") is not None


skip_no_omniprobe = pytest.mark.skipif(
    not _has_omniprobe_binary(), reason="omniprobe executable not on PATH"
)
skip_no_upstream_api = pytest.mark.skipif(
    not _has_upstream_api(),
    reason="omniprobe.api not importable (Omniprobe repo not on PYTHONPATH)",
)
skip_no_gpu = pytest.mark.skipif(
    not _has_amd_gpu(), reason="No AMD GPU detected (/dev/kfd missing)"
)
skip_no_hipcc = pytest.mark.skipif(not _has_hipcc(), reason="hipcc not available on PATH")


@pytest.fixture(scope="session")
def strided_binary(tmp_path_factory):
    """Compile the two-kernel HIP program once per session and yield its path.

    Compiled with ``-g``: without debug info Omniprobe still reports correct
    counts, but ``source_file``/``line``/``column`` come back empty, and the
    location fields are most of what these tests are checking.
    """
    tmp_path = tmp_path_factory.mktemp("omniprobe_integration")
    source = tmp_path / "strided.hip"
    source.write_text(STRIDED_KERNEL)

    binary = tmp_path / "strided"
    result = subprocess.run(
        ["hipcc", str(source), "-o", str(binary), "-O2", "-g"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"hipcc failed to build the test kernel:\n{result.stderr}")

    return binary


@pytest.fixture
def probe():
    """Construct the IntelliKit Omniprobe wrapper.

    Imported inside the fixture rather than at module scope so that a missing
    upstream API surfaces as a skip on the tests that need it, not as an error
    while collecting the module.
    """
    from omniprobe_ik import Omniprobe

    return Omniprobe()
