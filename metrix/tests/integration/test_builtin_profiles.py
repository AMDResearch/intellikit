# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Every built-in profile, on whatever GPU is present.

`METRIC_PROFILES` is declared once for all architectures, but metric coverage
varies: the CDNA-only entries in `memory` have no counterpart on RDNA. A
profile must therefore either collect the metrics this architecture supports,
or fail with a message naming the architecture — never crash on a metric the
backend has never heard of.

Deliberately not gated on a single architecture. The absence of an
architecture-generic profile test is why `memory`, `memory_cache`, and
`compute` shipped broken on every RDNA part.
"""

import tempfile
from pathlib import Path

import pytest

from metrix import Metrix
from metrix.metrics import METRIC_PROFILES

from ..unit.conftest import ARCHS_WITH_COUNTER_DEFS, HW_ARCH
from .test_inline_hip_profiling import _compile_hip

# Every profile except `compute` (CDNA-only) has at least one metric on every
# architecture that exposes hardware counters, so taking the "nothing available
# here" branch for any of them is a regression rather than an expected outcome.
# On a GPU with no counters at all -- gfx1103, and any other arch missing from
# counter_defs.yaml -- no profile can resolve, so there is nothing to guarantee.
#
# The exemption is keyed on the architecture being absent from counter_defs.yaml,
# never on the observed metric set being empty. Keying it on emptiness would be
# self-exempting: if metrics disappeared on a supported arch such as gfx942, the
# observed set would go empty, UNIVERSAL_PROFILES would collapse to nothing, and
# every assertion below would pass vacuously -- total metric loss would read as
# green. An arch that counter_defs.yaml does define must produce metrics, and
# fails loudly here when it does not.
_GPU_HAS_NO_COUNTERS = HW_ARCH is not None and HW_ARCH not in ARCHS_WITH_COUNTER_DEFS
UNIVERSAL_PROFILES = (
    frozenset() if _GPU_HAS_NO_COUNTERS else frozenset(METRIC_PROFILES) - {"compute"}
)

# Larger than the shared vector_add fixture on purpose: the memory profiles
# need enough traffic for the L2 and VRAM counters to register.
PROFILE_KERNEL_HIP = """
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void saxpy(const float* __restrict__ a, float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] * 2.0f + 1.0f;
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    float *d_a, *d_b;
    if (hipMalloc(&d_a, bytes) != hipSuccess) return 1;
    if (hipMalloc(&d_b, bytes) != hipSuccess) return 1;
    if (hipMemset(d_a, 0, bytes) != hipSuccess) return 1;
    hipLaunchKernelGGL(saxpy, dim3((N + 255) / 256), dim3(256), 0, 0, d_a, d_b, N);
    if (hipDeviceSynchronize() != hipSuccess) return 1;
    hipFree(d_a);
    hipFree(d_b);
    return 0;
}
"""


@pytest.fixture(scope="module")
def saxpy_binary():
    """Compile the kernel once for the whole module, not once per profile."""
    with tempfile.TemporaryDirectory(prefix="metrix_profiles_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        yield _compile_hip(PROFILE_KERNEL_HIP, "saxpy", tmp_path), tmp_path


@pytest.mark.slow
@pytest.mark.parametrize("profile_name", sorted(METRIC_PROFILES))
def test_builtin_profile_runs_or_reports_unavailable(profile_name, saxpy_binary):
    """Each built-in profile collects what this GPU supports, or says why it can't."""
    bin_path, tmp_path = saxpy_binary
    profiler = Metrix()
    arch = profiler.backend.device_specs.arch

    try:
        results = profiler.profile(
            command=str(bin_path),
            profile=profile_name,
            num_replays=1,
            cwd=str(tmp_path),
            timeout_seconds=120,
        )
    except ValueError as exc:
        # Acceptable only for a profile with no metric on this architecture,
        # and only with a message the user can act on.
        assert profile_name not in UNIVERSAL_PROFILES, (
            f"profile '{profile_name}' should be supported on every architecture "
            f"but reported nothing available on {arch}: {exc}"
        )
        message = str(exc)
        assert arch in message, (
            f"profile '{profile_name}' failed without naming the architecture: {message}"
        )
        assert profile_name in message
        return

    user_kernels = [k for k in results.kernels if "saxpy" in k.name]
    assert user_kernels, f"Expected 'saxpy' among {[k.name for k in results.kernels]}"

    # Whatever survived the architecture filter must actually produce values.
    available = set(profiler.backend.get_available_metrics())
    expected = [m for m in METRIC_PROFILES[profile_name]["metrics"] if m in available]
    assert expected, f"profile '{profile_name}' returned results but selected no metrics"
    missing = set(expected) - set(user_kernels[0].metrics)
    unexpected_missing = set()
    for metric_name in missing:
        metric = profiler.backend._metrics[metric_name]
        weight_counter = metric.get("weight_counter")
        matching_stats = [
            counters for key, counters in profiler.backend._aggregated.items() if "saxpy" in key
        ]
        # A collected, nonpositive weight means the workload had no valid
        # sample for this metric. A missing weight counter is instead a
        # collection regression and must remain in unexpected_missing.
        unavailable_sample = (
            metric.get("aggregation") == "samples"
            and weight_counter
            and matching_stats
            and all(
                weight_counter in counters and counters[weight_counter].max <= 0
                for counters in matching_stats
            )
        )
        if not unavailable_sample:
            unexpected_missing.add(metric_name)

    assert not unexpected_missing, (
        f"profile '{profile_name}' on {arch} is missing {sorted(unexpected_missing)}"
    )
