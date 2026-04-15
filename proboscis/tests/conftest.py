# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import os
import pytest


def has_rocm():
    """Check if ROCm is available."""
    return os.path.exists("/opt/rocm") or os.environ.get("ROCM_PATH")


requires_rocm = pytest.mark.skipif(
    not has_rocm(),
    reason="ROCm not available"
)
