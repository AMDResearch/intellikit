# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Linex: profile the PyTorch tensor_add example (GPU + torch + rocprofv3)."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from linex import Linex

SCRIPT = (
    Path(__file__).resolve().parents[1] / "examples" / "02_pytorch_tensor_add" / "tensor_add.py"
)


def _torch_cuda_ok() -> bool:
    torch = pytest.importorskip("torch")
    return bool(torch.cuda.is_available())


@pytest.fixture(scope="module")
def require_example_torch_cuda_rocprof():
    if not SCRIPT.is_file():
        pytest.skip(f"Example script missing: {SCRIPT}")
    if not _torch_cuda_ok():
        pytest.skip("PyTorch CUDA/ROCm device not available")
    if shutil.which("rocprofv3") is None:
        pytest.skip("rocprofv3 not on PATH")


def test_linex_profile_pytorch_tensor_add(require_example_torch_cuda_rocprof):
    """README-style: Linex().profile on tensor_add.py with a temp SQTT output dir."""
    with tempfile.TemporaryDirectory(prefix="linex_pytorch_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Larger than 256 helps rocprof/SQTT see waves on the forced CU (Linex sets HSA_CU_MASK).
        cmd = f"{sys.executable} {SCRIPT} --size 2048"
        profiler = Linex()
        profiler.profile(command=cmd, output_dir=str(tmp_path / "sqtt_out"))
    assert len(profiler.instructions) >= 1
