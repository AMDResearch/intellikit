# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Failures in the instrumented run must surface as errors, never as results.

Upstream's Python API raises ``RuntimeError`` when ``omniprobe`` exits non-zero,
and that check is the only one between a crashed run and a truncated result that
looks valid. These tests pin both halves: that the wrapper propagates the error,
and that a real crash under a real Omniprobe produces one.
"""

import pytest

from .conftest import (
    skip_no_gpu,
    skip_no_hipcc,
    skip_no_omniprobe,
    skip_no_upstream_api,
)

# Launches the instrumented kernel, then dies. A run that ignores the exit
# status would report coalesced_copy's accesses as if nothing went wrong.
CRASHING_KERNEL = """
#include <hip/hip_runtime.h>
#include <stdlib.h>

__global__ void coalesced_copy(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

int main() {
    const int N = 1024;
    float *d_in, *d_out;
    hipMalloc(&d_in, N * sizeof(float));
    hipMalloc(&d_out, N * sizeof(float));
    hipLaunchKernelGGL(coalesced_copy, dim3(N / 256), dim3(256), 0, 0, d_in, d_out, N);
    hipDeviceSynchronize();
    exit(3);
}
"""


@skip_no_upstream_api
def test_nonzero_exit_raises(tmp_path):
    """A non-zero omniprobe exit status reaches the caller as RuntimeError.

    Needs no GPU: a stand-in ``omniprobe`` that only exits 3 exercises the same
    code path a crashed run takes.
    """
    from omniprobe_ik import Omniprobe

    fake = tmp_path / "omniprobe"
    fake.write_text("#!/bin/sh\necho 'simulated failure' >&2\nexit 3\n")
    fake.chmod(0o755)

    probe = Omniprobe(omniprobe_path=str(fake))
    with pytest.raises(RuntimeError, match="exited with code 3"):
        probe.analyze_memory(command="./whatever")


@skip_no_omniprobe
@skip_no_upstream_api
@skip_no_gpu
@skip_no_hipcc
def test_crashing_target_raises(probe, tmp_path_factory):
    """An instrumented target that exits non-zero is an error, not a result."""
    from .conftest import _compile

    binary = _compile(tmp_path_factory, CRASHING_KERNEL, "crashing_mem", "memory")
    with pytest.raises(RuntimeError, match="exited with code"):
        probe.analyze_memory(command=str(binary))
