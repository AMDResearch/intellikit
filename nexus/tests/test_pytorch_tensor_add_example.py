# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Nexus: trace the PyTorch tensor_add example (GPU + torch + libnexus)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nexus import Nexus

SCRIPT = (
    Path(__file__).resolve().parents[1] / "examples" / "06_pytorch_tensor_add" / "tensor_add.py"
)


def _torch_cuda_ok() -> bool:
    torch = pytest.importorskip("torch")
    return bool(torch.cuda.is_available())


@pytest.fixture(scope="module")
def require_example_torch_cuda_and_nexus():
    if not SCRIPT.is_file():
        pytest.skip(f"Example script missing: {SCRIPT}")
    if not _torch_cuda_ok():
        pytest.skip("PyTorch CUDA/ROCm device not available")
    try:
        Nexus(log_level=0)
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("log_level", [0, 1])
def test_nexus_trace_pytorch_tensor_add(require_example_torch_cuda_and_nexus, log_level):
    """trace_pytorch-style: Nexus().run on tensor_add.py; trace has assembly."""
    nexus = Nexus(log_level=log_level)
    cmd = [sys.executable, str(SCRIPT), "--size", "256"]
    trace = nexus.run(cmd, cwd=str(SCRIPT.parent))
    assert len(trace) >= 1
    assert any(len(k.assembly) >= 1 for k in trace), (
        f"Expected at least one kernel with assembly, got: {[(k.name, len(k.assembly)) for k in trace]}"
    )
