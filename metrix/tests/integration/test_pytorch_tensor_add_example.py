# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Integration tests: PyTorch tensor_add example with Metrix CLI (GPU + torch + rocprof)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples" / "02_pytorch_tensor_add" / "tensor_add.py"
)


def _torch_cuda_ok() -> bool:
    torch = pytest.importorskip("torch")
    return bool(torch.cuda.is_available())


@pytest.fixture(scope="module")
def require_example_and_torch_cuda():
    if not SCRIPT.is_file():
        pytest.skip(f"Example script missing: {SCRIPT}")
    if not _torch_cuda_ok():
        pytest.skip("PyTorch CUDA/ROCm device not available")


@pytest.fixture(scope="module")
def require_metrix_cli(require_example_and_torch_cuda):
    if shutil.which("metrix") is None:
        pytest.skip("metrix CLI not on PATH")


@pytest.mark.integration
def test_tensor_add_script_runs(require_example_and_torch_cuda):
    """Sanity: example runs without Metrix."""
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--size", "256"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "sum(a+b)" in r.stdout


@pytest.mark.integration
def test_metrix_time_only_on_pytorch_example(require_metrix_cli):
    """README-style: metrix --time-only -n 1 on the PyTorch workload."""
    cmd = f"{sys.executable} {SCRIPT} --size 256"
    r = subprocess.run(
        ["metrix", "--time-only", "-n", "1", "--timeout", "300", cmd],
        capture_output=True,
        text=True,
        timeout=360,
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "Duration:" in out and "μs" in out, out
