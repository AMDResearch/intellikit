# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Shared fixtures and helpers for per-metric microbenchmarks.

Provides HIP compilation, GPU architecture detection, and a profiling
helper that returns metric values for the longest-running kernel.
"""

import subprocess
from pathlib import Path

import pytest

from metrix import Metrix

# ---------------------------------------------------------------------------
# HIP compilation
# ---------------------------------------------------------------------------

HIP_KERNELS_DIR = Path(__file__).parent / "hip_kernels"


def compile_hip(kernel_name: str, tmp_path: Path, extra_flags: list | None = None) -> Path:
    """Compile a .hip file from hip_kernels/ and return the binary path.

    Parameters
    ----------
    kernel_name : str
        Filename (without directory) inside hip_kernels/, e.g. "hbm_bandwidth.hip".
    tmp_path : Path
        Temporary directory for the compiled binary.
    extra_flags : list, optional
        Additional hipcc flags (e.g. ["-DUSE_FLOAT4"]).
    """
    src = HIP_KERNELS_DIR / kernel_name
    if not src.exists():
        raise FileNotFoundError(f"HIP kernel not found: {src}")

    binary = tmp_path / src.stem
    cmd = ["hipcc", str(src), "-o", str(binary), "-O2"]
    if extra_flags:
        cmd.extend(extra_flags)

    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"hipcc failed for {kernel_name}:\n{r.stderr}")
    return binary


# ---------------------------------------------------------------------------
# GPU architecture detection
# ---------------------------------------------------------------------------

_CACHED_ARCH: str | None = None


def get_arch() -> str:
    """Return the gfx arch string (e.g. 'gfx942') for device 0.

    Uses metrix's own detection first, falls back to rocminfo parsing.
    """
    global _CACHED_ARCH
    if _CACHED_ARCH is not None:
        return _CACHED_ARCH

    # Try metrix's built-in detection
    try:
        from metrix.backends.detect import detect_gpu_arch

        _CACHED_ARCH = detect_gpu_arch()
        return _CACHED_ARCH
    except Exception:
        pass

    # Fallback: parse rocminfo
    try:
        r = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10
        )
        for line in r.stdout.splitlines():
            if "gfx" in line and "Name:" in line:
                _CACHED_ARCH = line.split()[-1].strip()
                return _CACHED_ARCH
    except Exception:
        pass

    return "unknown"


# ---------------------------------------------------------------------------
# Profiling helper
# ---------------------------------------------------------------------------


def profile_longest_kernel(
    command: str,
    metrics: list[str],
    cwd: str | None = None,
    num_replays: int = 2,
    timeout: int = 120,
) -> dict[str, float]:
    """Profile a command and return metric values for the longest kernel.

    Parameters
    ----------
    command : str
        Shell command to profile (binary path + args).
    metrics : list[str]
        Metric names to collect (e.g. ["memory.hbm_read_bandwidth"]).
    cwd : str, optional
        Working directory for the profiled command.
    num_replays : int
        Number of profiling replays for statistical stability.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    dict[str, float]
        Mapping of metric name to average value for the longest kernel.
    """
    profiler = Metrix()
    results = profiler.profile(
        command=command,
        metrics=metrics,
        num_replays=num_replays,
        aggregate_by_kernel=True,
        cwd=cwd,
        timeout_seconds=timeout,
    )
    if not results.kernels:
        pytest.fail("No kernels were profiled — binary may not launch any GPU kernels")

    # Pick kernel with longest average duration
    kernel = max(results.kernels, key=lambda k: k.duration_us.avg)
    return {m: kernel.metrics[m].avg for m in metrics if m in kernel.metrics}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def arch():
    """Current GPU architecture string."""
    a = get_arch()
    if a == "unknown":
        pytest.skip("Could not detect GPU architecture")
    return a
