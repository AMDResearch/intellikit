# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Kerncap: profile the PyTorch tensor_add example (GPU + torch + rocprofv3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kerncap import Kerncap

from tests.integration.conftest import skip_no_gpu, skip_no_rocprof

SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples" / "02_pytorch_tensor_add" / "tensor_add.py"
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
    pytest.importorskip("torch")


@skip_no_gpu
@skip_no_rocprof
def test_kerncap_profile_pytorch_tensor_add(require_example_torch_cuda_rocprof):
    """README-style: Kerncap().profile on tensor_add.py (profile only, no extract)."""
    kc = Kerncap()
    profile = kc.profile([sys.executable, str(SCRIPT), "--size", "2048"])
    assert len(profile) >= 1
