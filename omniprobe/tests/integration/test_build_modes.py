# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""How the application's build mode affects instrumentation.

``instrumentation_flags`` leaves out ``-fgpu-rdc`` by default. These tests pin
why: a multi-file program built without RDC has every kernel matched, including
``static`` and anonymous-namespace ones, while the same program built with RDC
and the plugin fails to link.
"""

import subprocess

import pytest

from .conftest import (
    skip_no_gpu,
    skip_no_hipcc,
    skip_no_omniprobe,
    skip_no_upstream_api,
)

pytestmark = [skip_no_omniprobe, skip_no_upstream_api, skip_no_gpu, skip_no_hipcc]

MAIN_TU = """
#include <hip/hip_runtime.h>
namespace {
__global__ void anon_kernel(const float* in, float* out) {
    int i = blockIdx.x * 64 + threadIdx.x;
    out[i] = in[i * 4];
}
}
static __global__ void static_kernel(const float* in, float* out) {
    int i = blockIdx.x * 64 + threadIdx.x;
    out[i] = in[i * 8];
}
void launch_other(const float*, float*);
int main() {
    float *a, *b;
    hipMalloc(&a, 1 << 22);
    hipMalloc(&b, 1 << 20);
    hipLaunchKernelGGL(anon_kernel, dim3(16), dim3(64), 0, 0, a, b);
    hipLaunchKernelGGL(static_kernel, dim3(16), dim3(64), 0, 0, a, b);
    launch_other(a, b);
    hipDeviceSynchronize();
    return 0;
}
"""

OTHER_TU = """
#include <hip/hip_runtime.h>
__global__ void other_tu_kernel(const float* in, float* out) {
    int i = blockIdx.x * 64 + threadIdx.x;
    out[i] = in[i * 16];
}
void launch_other(const float* a, float* b) {
    hipLaunchKernelGGL(other_tu_kernel, dim3(16), dim3(64), 0, 0, a, b);
}
"""


def _build_two_tus(tmp_path_factory, name, rdc):
    from omniprobe_ik import instrumentation_flags

    tmp = tmp_path_factory.mktemp(name)
    main, other = tmp / "main.hip", tmp / "other.hip"
    main.write_text(MAIN_TU)
    other.write_text(OTHER_TU)
    binary = tmp / name
    flags = instrumentation_flags("memory", rdc=rdc)
    result = subprocess.run(
        ["hipcc", "-O2", "-g", *flags, str(main), str(other), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    return binary, result


def test_non_rdc_multi_file_build_matches_every_kernel(probe, tmp_path_factory):
    binary, build = _build_two_tus(tmp_path_factory, "two_tu", rdc=False)
    assert build.returncode == 0, build.stderr[-2000:]

    result = probe.analyze_memory(command=str(binary))

    def excess(fragment):
        matches = [k for k in result.kernels if fragment in k.kernel_name]
        assert len(matches) == 1, [k.kernel_name for k in result.kernels]
        return sum(a.excess_cache_lines for a in matches[0].accesses)

    # 16 waves; per wave min(64, 2S) - 2 excess lines for lane stride S floats
    assert len(result.kernels) == 3
    assert excess("anon_kernel") == 16 * 6
    assert excess("static_kernel") == 16 * 14
    assert excess("other_tu_kernel") == 16 * 30


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Omniprobe defect: with -fgpu-rdc and the pass plugin, linking two or more "
        "translation units fails with 'duplicate symbol: __hip_cuid_...'. Reported 2026-11-08."
    ),
)
def test_rdc_multi_file_build_links(tmp_path_factory):
    _, build = _build_two_tus(tmp_path_factory, "two_tu_rdc", rdc=True)
    assert build.returncode == 0, build.stderr[-500:]
