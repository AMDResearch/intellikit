# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
omniprobe_ik: GPU Memory Coalescing and Basic-Block Timing Analysis

Wraps Omniprobe's Python API to provide memory coalescing analysis
(uncoalesced global accesses, LDS bank conflicts) and basic-block
ISA timing through IntelliKit's standard interfaces.
"""

from importlib.metadata import PackageNotFoundError, version as _get_version

from .api import (
    BasicBlock,
    BasicBlockResult,
    KernelBBResult,
    KernelMemoryResult,
    MemoryAccess,
    MemoryAnalysisResult,
    Omniprobe,
)

try:
    __version__ = _get_version("omniprobe-ik")
except PackageNotFoundError:
    __version__ = "0.1.1"

__all__ = [
    "Omniprobe",
    "MemoryAnalysisResult",
    "KernelMemoryResult",
    "MemoryAccess",
    "BasicBlockResult",
    "KernelBBResult",
    "BasicBlock",
]
