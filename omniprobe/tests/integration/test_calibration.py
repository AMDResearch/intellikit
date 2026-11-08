# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Calibrate what Omniprobe's memory-analysis numbers mean.

Each kernel here has a closed-form expected result, derived from Omniprobe's
model (``src/memory_analysis_handler.cc``) and confirmed on hardware. The
formulas are what ``docs/tools/omniprobe.md`` tells agents to rely on, so if
Omniprobe changes its model, these tests are the ones that notice.

Model, per wave execution of one instrumented instruction (gfx90a/gfx94x,
128-byte cache lines, 4-byte accesses, 64 active lanes):

* global: ``cache_lines_needed = 2``; ``used = min(64, 2 * S)`` for lane
  stride ``S`` floats; ``excess_cache_lines = used - needed``.
* LDS: 32 banks of 4 bytes; the wave is checked as two halves of 32 lanes;
  each half costs ``ways - 1`` where ``ways`` is the largest number of distinct
  dwords mapped to one bank (same-dword broadcast is free).
* ``execution_count`` counts wave executions of the instruction.

All values are totals over the dispatch, so the expectations scale by the
number of waves.
"""

import pytest

from .conftest import (
    skip_no_gpu,
    skip_no_hipcc,
    skip_no_omniprobe,
    skip_no_upstream_api,
)

pytestmark = [skip_no_omniprobe, skip_no_upstream_api, skip_no_gpu, skip_no_hipcc]

WAVES = 64  # 64 blocks of one 64-lane wave each
STRIDES = (1, 2, 4, 8, 16, 32, 64)
LDS_STRIDES = (1, 2, 4, 8, 16, 32, 33, 64)

CALIBRATION_SOURCE = """
#include <hip/hip_runtime.h>

struct Arrays { const float* in; float* out; };

template <int S>
__global__ void gstride(const float* in, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = in[(size_t)i * S];  // CAL_GLOBAL_LOAD
    out[i] = v;
}

template <int S>
__global__ void lstride(float* out) {
    __shared__ float tile[64 * 65];
    int l = threadIdx.x;
    for (int k = l; k < 64 * 65; k += 64) tile[k] = (float)k;
    __syncthreads();
    float v = tile[l * S];  // CAL_LDS_LOAD
    out[blockIdx.x * 64 + l] = v;
}

__global__ void via_struct(Arrays a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a.out[i] = a.in[i * 16];  // CAL_STRUCT
}

__global__ void gather(const float* in, const int* map, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = in[map[i]];  // CAL_GATHER
}

__global__ void via_table(float* const* table) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float* in = table[0];
    float* out = table[1];
    out[i] = in[i * 16];  // CAL_TABLE
}

__constant__ const float* __restrict__ cin;

__global__ void no_args() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = cin[i];  // CAL_NOARGS
    if (v == -1.0f) __builtin_trap();
}

int main() {
    const int blocks = 64, threads = 64, n = blocks * threads;
    float *in, *out; int* map; float** table;
    hipMalloc(&in, sizeof(float) * n * 64);
    hipMalloc(&out, sizeof(float) * n);
    hipMalloc(&map, sizeof(int) * n);
    hipMalloc(&table, sizeof(float*) * 2);
    hipMemset(in, 0, sizeof(float) * n * 64);
    int* h = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) h[i] = (i * 7919) % (n * 16);
    hipMemcpy(map, h, sizeof(int) * n, hipMemcpyHostToDevice);
    float* ht[2] = {in, out};
    hipMemcpy(table, ht, sizeof(ht), hipMemcpyHostToDevice);
    hipMemcpyToSymbol(HIP_SYMBOL(cin), &in, sizeof(in));
    Arrays a{in, out};
#define G(S) hipLaunchKernelGGL(gstride<S>, dim3(blocks), dim3(threads), 0, 0, in, out);
#define L(S) hipLaunchKernelGGL(lstride<S>, dim3(blocks), dim3(threads), 0, 0, out);
    G(1) G(2) G(4) G(8) G(16) G(32) G(64)
    L(1) L(2) L(4) L(8) L(16) L(32) L(33) L(64)
    hipLaunchKernelGGL(via_struct, dim3(blocks), dim3(threads), 0, 0, a);
    hipLaunchKernelGGL(gather, dim3(blocks), dim3(threads), 0, 0, in, map, out);
    hipLaunchKernelGGL(via_table, dim3(blocks), dim3(threads), 0, 0, table);
    hipLaunchKernelGGL(no_args, dim3(blocks), dim3(threads), 0, 0);
    hipDeviceSynchronize();
    return 0;
}
"""


def _line_of(marker):
    for number, text in enumerate(CALIBRATION_SOURCE.splitlines(), start=1):
        if marker in text:
            return number
    raise AssertionError(marker)


@pytest.fixture(scope="module")
def calibration(tmp_path_factory):
    """Analyze the calibration program once; index results by kernel name."""
    from omniprobe_ik import Omniprobe

    from .conftest import _compile

    probe = Omniprobe()

    binary = _compile(tmp_path_factory, CALIBRATION_SOURCE, "calibration", "memory")
    result = probe.analyze_memory(command=str(binary))
    kernels = {kernel.kernel_name: kernel for kernel in result.kernels}
    kernels["__uninstrumented__"] = result.uninstrumented_kernels
    return kernels


def _kernel(calibration, prefix):
    matches = [
        k
        for name, k in calibration.items()
        if name.startswith(prefix) and name != "__uninstrumented__"
    ]
    assert len(matches) == 1, f"{prefix!r}: {sorted(calibration)}"
    return matches[0]


def _totals(kernel, line, space, access_type):
    """Sum entries for one source line/space/type.

    Omniprobe may split one location into several entries (it keys on access
    sizes too), so totals, not single entries, are the stable quantity.
    """
    entries = [
        a
        for a in kernel.accesses
        if a.line == line and a.memory_space == space and a.access_type == access_type
    ]
    return (
        sum(a.excess_cache_lines for a in entries),
        sum(a.bank_conflicts for a in entries),
        sum(a.execution_count for a in entries),
    )


@pytest.mark.parametrize("stride", STRIDES)
def test_global_excess_cache_lines(calibration, stride):
    kernel = _kernel(calibration, f"void gstride<{stride}>")
    excess, _, executions = _totals(kernel, _line_of("CAL_GLOBAL_LOAD"), "global", "read")
    assert executions == WAVES
    assert excess == WAVES * (min(64, 2 * stride) - 2)


@pytest.mark.parametrize("stride", LDS_STRIDES)
def test_lds_bank_conflicts(calibration, stride):
    kernel = _kernel(calibration, f"void lstride<{stride}>")
    _, conflicts, executions = _totals(kernel, _line_of("CAL_LDS_LOAD"), "lds", "read")
    ways = 1 if stride % 2 else min(stride, 32)
    assert executions == WAVES
    assert conflicts == WAVES * 2 * (ways - 1)


@pytest.mark.parametrize("stride", LDS_STRIDES)
def test_lds_location_is_one_entry(calibration, stride):
    """Omniprobe 3c6182d reports one entry per LDS source location (earlier ones split it)."""
    kernel = _kernel(calibration, f"void lstride<{stride}>")
    line = _line_of("CAL_LDS_LOAD")
    entries = [a for a in kernel.accesses if a.line == line and a.memory_space == "lds"]
    assert len([a for a in entries if a.access_type == "read"]) == 1, entries


def test_pointers_in_a_by_value_struct_are_seen(calibration):
    kernel = _kernel(calibration, "via_struct")
    excess, _, executions = _totals(kernel, _line_of("CAL_STRUCT"), "global", "read")
    assert executions == WAVES
    assert excess == WAVES * (32 - 2)


def test_indirect_gather_is_seen(calibration):
    """An index array, then a data-dependent load."""
    kernel = _kernel(calibration, "gather")
    reads = [
        a for a in kernel.accesses if a.line == _line_of("CAL_GATHER") and a.access_type == "read"
    ]
    assert len(reads) >= 2, "expected both the index load and the gathered load"
    assert max(a.excess_cache_lines for a in reads) > 0


def test_pointer_loaded_from_memory_is_seen(calibration):
    """Omniprobe e807927+ classifies flat accesses per wave at run time: a pointer loaded
    from device memory is analyzed as global (in[i * 16]: 30 excess lines per execution)."""
    kernel = _kernel(calibration, "via_table")
    line = _line_of("CAL_TABLE")
    excess, _, executions = _totals(kernel, line, "global", "read")
    assert (executions, excess) == (WAVES, 30 * WAVES)
    excess, _, executions = _totals(kernel, line, "global", "write")
    assert (executions, excess) == (WAVES, 0)


def test_kernel_without_arguments_is_seen(calibration):
    names = [n for n in calibration if n.startswith("no_args")]
    if not names:
        raise KeyError("no_args")
    kernel = calibration[names[0]]
    _, _, executions = _totals(kernel, _line_of("CAL_NOARGS"), "global", "read")
    assert executions == WAVES


def test_kernel_without_arguments_is_not_reported_as_uninstrumented(calibration):
    """Omniprobe ac6ce8e instruments kernels without parameters; they must not be listed as missing."""
    missing = calibration["__uninstrumented__"] or []
    assert not [k for k in missing if k.startswith("no_args")], missing


def test_pointer_loaded_from_memory_is_not_left_unanalyzed(calibration):
    """All lanes of each wave point into global memory, so nothing is left as flat."""
    assert _kernel(calibration, "via_table").unanalyzed_flat_accesses == 0


def test_unattributed_access_in_a_kernel_with_debug_info_is_not_blamed_on_g(calibration):
    """via_table is built with -g and has attributed accesses; Omniprobe 22c4d95 still leaves
    one wave-level access per wave without a line. The warning must not ask for -g."""
    from omniprobe_ik import MemoryAnalysisResult, summarize_memory

    kernel = _kernel(calibration, "via_table")
    assert kernel.accesses and kernel.unattributed_accesses == WAVES
    warnings = " ".join(summarize_memory(MemoryAnalysisResult(kernels=[kernel]))["warnings"])
    assert "not missing" in warnings and "rebuild with -g" not in warnings.lower()
