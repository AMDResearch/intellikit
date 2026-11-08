# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""How the application's build mode affects instrumentation.

``instrumentation_flags`` leaves out ``-fgpu-rdc`` by default; instrumentation
does not need it. A multi-file program has every kernel matched, including
``static`` and anonymous-namespace ones, both without RDC and (Omniprobe 3003058
and later) with it.
"""

import subprocess
from pathlib import Path

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


def test_rdc_multi_file_build_matches_every_kernel(probe, tmp_path_factory):
    """With -fgpu-rdc, two translation units link and every kernel is measured exactly."""
    binary, build = _build_two_tus(tmp_path_factory, "two_tu_rdc", rdc=True)
    assert build.returncode == 0, build.stderr[-2000:]

    result = probe.analyze_memory(command=str(binary))

    def excess(fragment):
        matches = [k for k in result.kernels if fragment in k.kernel_name]
        assert len(matches) == 1, [k.kernel_name for k in result.kernels]
        return sum(a.excess_cache_lines for a in matches[0].accesses)

    assert not result.uninstrumented_kernels, result.uninstrumented_kernels
    assert len(result.kernels) == 3
    assert excess("anon_kernel") == 16 * 6
    assert excess("static_kernel") == 16 * 14
    assert excess("other_tu_kernel") == 16 * 30


def test_partially_instrumented_build_names_the_missing_kernel(probe, tmp_path_factory):
    """Only main.hip gets the plugin: other_tu_kernel runs uninstrumented and must be named."""
    from omniprobe_ik import instrumentation_flags, summarize_memory

    tmp = tmp_path_factory.mktemp("partial")
    main, other = tmp / "main.hip", tmp / "other.hip"
    main.write_text(MAIN_TU)
    other.write_text(OTHER_TU)
    flags = instrumentation_flags("memory")
    arch = [f for f in flags if f.startswith("--offload-arch=")]
    steps = [
        ["hipcc", "-O2", "-g", *flags, "-c", str(main), "-o", str(tmp / "main.o")],
        ["hipcc", "-O2", "-g", *arch, "-c", str(other), "-o", str(tmp / "other.o")],
        ["hipcc", *arch, str(tmp / "main.o"), str(tmp / "other.o"), "-o", str(tmp / "app")],
    ]
    for step in steps:
        build = subprocess.run(step, capture_output=True, text=True)
        assert build.returncode == 0, build.stderr[-2000:]

    result = probe.analyze_memory(command=str(tmp / "app"))

    assert result.uninstrumented_kernels is not None, "omniprobe's log was not captured"
    assert [k for k in result.uninstrumented_kernels if "other_tu_kernel" in k]
    assert not [k for k in result.uninstrumented_kernels if "static_kernel" in k]
    warnings = summarize_memory(result)["warnings"]
    assert warnings and "other_tu_kernel" in warnings[0]


def test_O0_build_is_analyzed(probe, tmp_path_factory):
    """Omniprobe e807927+ classifies -O0's flat accesses at run time: same finding as -O2."""
    from omniprobe_ik import instrumentation_flags, summarize_memory

    from .conftest import STRIDED_KERNEL

    tmp = tmp_path_factory.mktemp("o0")
    src = tmp / "strided.hip"
    src.write_text(STRIDED_KERNEL)
    build = subprocess.run(
        ["hipcc", "-O0", "-g", *instrumentation_flags("memory"), str(src), "-o", str(tmp / "app")],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr[-2000:]

    summary = summarize_memory(probe.analyze_memory(command=str(tmp / "app")))
    line = STRIDED_KERNEL.splitlines().index("        out[idx] = in[idx] * 2.0f;", 15) + 1
    sites = {
        (s["kernel"].split("(")[0], s["line"], s["access_type"]): s["excess_per_execution"]
        for s in summary["global_sites"]
    }
    # Lanes 16 floats apart: min(64, 2 * 16) - 2 = 30 excess lines per wave execution.
    assert sites == {("strided_copy", line, "read"): 30.0, ("strided_copy", line, "write"): 30.0}
    assert summary["warnings"] == []


def test_build_without_debug_info_is_flagged(probe, tmp_path_factory):
    """Without -g every global access is dropped; the summary must say so and name -g."""
    from omniprobe_ik import instrumentation_flags, summarize_memory

    from .conftest import STRIDED_KERNEL

    tmp = tmp_path_factory.mktemp("nog")
    src = tmp / "strided.hip"
    src.write_text(STRIDED_KERNEL)
    build = subprocess.run(
        ["hipcc", "-O2", *instrumentation_flags("memory"), str(src), "-o", str(tmp / "app")],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr[-2000:]

    result = probe.analyze_memory(command=str(tmp / "app"))
    assert sum(k.unattributed_accesses for k in result.kernels) > 0
    warnings = summarize_memory(result)["warnings"]
    assert any("-g" in w and "strided_copy" in w for w in warnings), warnings
    assert not any("-O0" in w for w in warnings), warnings


@pytest.mark.skipif(
    Path("/opt/rocm/llvm/bin/llvm-objdump").exists(),
    reason="needs a node where /opt/rocm is not a full ROCm install",
)
def test_unset_rocm_path_still_reports(probe, tmp_path_factory, monkeypatch):
    """Omniprobe 149cf2c+ finds a ROCm with llvm-objdump itself; without ROCM_PATH (and no
    /opt/rocm) it still attributes accesses. Before, it reported nothing."""
    from omniprobe_ik import instrumentation_flags

    from .conftest import STRIDED_KERNEL

    tmp = tmp_path_factory.mktemp("norocm")
    src = tmp / "strided.hip"
    src.write_text(STRIDED_KERNEL)
    build = subprocess.run(
        ["hipcc", "-O2", "-g", *instrumentation_flags("memory"), str(src), "-o", str(tmp / "app")],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr[-2000:]

    monkeypatch.delenv("ROCM_PATH", raising=False)
    result = probe.analyze_memory(command=str(tmp / "app"))
    names = {k.kernel_name.split("(")[0] for k in result.kernels if k.accesses}
    assert names == {"coalesced_copy", "strided_copy"}
