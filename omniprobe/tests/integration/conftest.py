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

Every binary is compiled with the Omniprobe instrumentation plugin its analysis
needs (``omniprobe_ik.instrumentation_flags``). Omniprobe only analyzes kernels
that carry the plugin's instrumentation; a plain ``hipcc`` build is reported as
having no instrumented kernels, which ``test_uninstrumented_binary_raises``
checks deliberately.
"""

import importlib.util
import os
import shutil
import subprocess
import sys

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


# A single kernel with a data-dependent branch, for basic-block analysis. Kept in
# sync with omniprobe/examples/02_basic_block_timing/analyze_basic_blocks.py.
BRANCHING_KERNEL = """
#include <hip/hip_runtime.h>

__global__ void divergent_work(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    float acc = in[idx];

    if ((idx & 1) == 0) {
        acc = acc * 2.0f + 1.0f;
    } else {
        for (int i = 0; i < 32; i++) {
            acc = __sinf(acc) + __cosf(acc);
        }
    }

    out[idx] = acc;
}

int main() {
    const int N = 256 * 1024;
    const size_t bytes = N * sizeof(float);

    float *d_in, *d_out;
    hipMalloc(&d_in, bytes);
    hipMalloc(&d_out, bytes);
    hipMemset(d_in, 0, bytes);

    hipLaunchKernelGGL(divergent_work, dim3(N / 256), dim3(256), 0, 0, d_in, d_out, N);
    hipDeviceSynchronize();

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
    from omniprobe_ik.api import _upstream_source_root

    try:
        if importlib.util.find_spec("omniprobe.api") is not None:
            return True
    except (ImportError, ValueError):
        pass
    # The wrapper also finds the API next to the omniprobe executable on PATH.
    return _upstream_source_root() is not None


_HIP_COUNT_PROBE = """
import ctypes, os, sys
names = ["libamdhip64.so"]
for root in (os.environ.get("ROCM_PATH"), os.environ.get("HIP_PATH")):
    if root:
        names.append(os.path.join(root, "lib", "libamdhip64.so"))
hipcc = os.environ.get("_IK_HIPCC")
if hipcc:
    names.append(os.path.join(os.path.dirname(os.path.realpath(hipcc)), "..", "lib", "libamdhip64.so"))
for name in names:
    try:
        lib = ctypes.CDLL(name)
    except OSError:
        continue
    count = ctypes.c_int(0)
    lib.hipGetDeviceCount(ctypes.byref(count))
    print(count.value)
    sys.exit(0)
print(-1)
"""


def _hip_device_count() -> int:
    """Count the GPUs HIP can see, or -1 if the HIP runtime cannot be loaded.

    ``/dev/kfd`` existing is not enough: ``ROCR_VISIBLE_DEVICES`` and
    ``HIP_VISIBLE_DEVICES`` compose, so setting both to the same id selects zero
    devices on a machine that plainly has GPUs, and every test then fails on an
    empty result instead of skipping. Asked in a subprocess so the test process
    itself never initializes the HIP runtime.
    """
    if not os.path.exists("/dev/kfd"):
        return 0
    env = dict(os.environ, _IK_HIPCC=shutil.which("hipcc") or "")
    try:
        out = subprocess.run(
            [sys.executable, "-c", _HIP_COUNT_PROBE],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        ).stdout.strip()
        return int(out.splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return -1


_HIP_DEVICES = _hip_device_count()


def _has_hipcc() -> bool:
    """Check whether hipcc is available to build the test kernel."""
    return shutil.which("hipcc") is not None


skip_no_omniprobe = pytest.mark.skipif(
    not _has_omniprobe_binary(), reason="omniprobe executable not on PATH"
)
skip_no_upstream_api = pytest.mark.skipif(
    not _has_upstream_api(),
    reason="Omniprobe's Python API not found (no omniprobe/api next to the omniprobe on PATH)",
)
skip_no_gpu = pytest.mark.skipif(
    _HIP_DEVICES <= 0,
    reason=(
        f"No HIP-visible GPU (HIP device count {_HIP_DEVICES}; /dev/kfd missing, or "
        "ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES select nothing -- HIP indexes "
        "within the ROCR set, so do not set both to the same id)"
    ),
)
skip_no_hipcc = pytest.mark.skipif(not _has_hipcc(), reason="hipcc not available on PATH")


def _compile(tmp_path_factory, source_text, name, analysis):
    """Compile *source_text* and return the binary's path.

    *analysis* selects the instrumentation plugin (``"memory"`` or
    ``"basic_blocks"``); ``None`` compiles without one. Always ``-g``: without
    debug info memory analysis cannot match accesses to source lines and drops
    them (``unattributed_accesses``), and the location fields are most of what
    these tests check. A failed build is a test failure, not a skip -- the
    prerequisites are already guarded, so a failure here is a real problem.
    """
    tmp_path = tmp_path_factory.mktemp(f"omniprobe_{name}")
    source = tmp_path / f"{name}.hip"
    source.write_text(source_text)
    binary = tmp_path / name

    flags = []
    if analysis is not None:
        from omniprobe_ik import instrumentation_flags

        flags = instrumentation_flags(analysis)

    cmd = ["hipcc", str(source), "-o", str(binary), "-O2", "-g", *flags]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"failed to build {name}: {' '.join(cmd)}\n{result.stderr}")
    return binary


@pytest.fixture(scope="session")
def strided_binary(tmp_path_factory):
    """The two-kernel program, instrumented for memory analysis."""
    return _compile(tmp_path_factory, STRIDED_KERNEL, "strided_mem", "memory")


@pytest.fixture(scope="session")
def strided_bb_binary(tmp_path_factory):
    """The two-kernel program, instrumented for basic-block analysis."""
    return _compile(tmp_path_factory, STRIDED_KERNEL, "strided_bb", "basic_blocks")


@pytest.fixture(scope="session")
def branching_bb_binary(tmp_path_factory):
    """The single-kernel branching program, instrumented for basic-block analysis."""
    return _compile(tmp_path_factory, BRANCHING_KERNEL, "branching_bb", "basic_blocks")


@pytest.fixture(scope="session")
def uninstrumented_binary(tmp_path_factory):
    """The two-kernel program built with plain hipcc -- invisible to Omniprobe."""
    return _compile(tmp_path_factory, STRIDED_KERNEL, "strided_plain", None)


@pytest.fixture
def probe():
    """Construct the IntelliKit Omniprobe wrapper.

    Imported inside the fixture rather than at module scope so that a missing
    upstream API surfaces as a skip on the tests that need it, not as an error
    while collecting the module.
    """
    from omniprobe_ik import Omniprobe

    return Omniprobe()
