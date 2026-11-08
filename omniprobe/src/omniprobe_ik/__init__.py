# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
omniprobe_ik: source-level GPU memory coalescing and LDS bank-conflict analysis

Wraps Omniprobe's Python API to provide memory coalescing analysis
(uncoalesced global accesses, LDS bank conflicts) and basic-block
ISA timing through IntelliKit's standard interfaces.
"""

from importlib.metadata import PackageNotFoundError, version as _get_version

from .api import (
    ANALYSIS_PLUGINS,
    BasicBlock,
    BasicBlockResult,
    KernelBBResult,
    KernelMemoryResult,
    MemoryAccess,
    MemoryAnalysisResult,
    NoInstrumentedKernelsError,
    Omniprobe,
    detect_offload_arch,
    instrumentation_flags,
    parse_uninstrumented_kernels,
    summarize_memory,
)

try:
    __version__ = _get_version("omniprobe-ik")
except PackageNotFoundError:
    __version__ = "0.1.1"

__all__ = [
    "ANALYSIS_PLUGINS",
    "Omniprobe",
    "NoInstrumentedKernelsError",
    "instrumentation_flags",
    "summarize_memory",
    "parse_uninstrumented_kernels",
    "detect_offload_arch",
    "MemoryAnalysisResult",
    "KernelMemoryResult",
    "MemoryAccess",
    "BasicBlockResult",
    "KernelBBResult",
    "BasicBlock",
]
