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

import subprocess
import tempfile
from pathlib import Path

import pytest

from metrix import Metrix
from metrix.metrics import METRIC_PROFILES

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
        timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"hipcc failed:\n{r.stderr}")
    return bin_path


@pytest.mark.slow
@pytest.mark.parametrize("profile_name", sorted(METRIC_PROFILES))
def test_builtin_profile_runs_or_reports_unavailable(profile_name):
    """Each built-in profile collects what this GPU supports, or says why it can't."""
    with tempfile.TemporaryDirectory(prefix="metrix_profiles_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(PROFILE_KERNEL_HIP, "saxpy", tmp_path)
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
            # Acceptable only when the profile is wholly unsupported here, and
            # only with a message the user can act on.
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
    assert set(expected) <= set(user_kernels[0].metrics), (
        f"profile '{profile_name}' on {arch} is missing "
        f"{sorted(set(expected) - set(user_kernels[0].metrics))}"
    )
