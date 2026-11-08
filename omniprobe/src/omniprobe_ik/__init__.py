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
    INSTALL_DOC,
    BasicBlock,
    BasicBlockResult,
    KernelBBResult,
    KernelMemoryResult,
    MemoryAccess,
    MemoryAnalysisResult,
    KernelFilterMatchedNothingError,
    NoInstrumentedKernelsError,
    Omniprobe,
    check_installation,
    detect_offload_arch,
    instrumentation_flags,
    parse_uninstrumented_kernels,
    summarize_memory,
    uninstrumented_warning,
)

from .raw import (
    ANALYZER_PLUGINS,
    AddressRecord,
    AddressTrace,
    AnalyzerRun,
    BasicBlockTrace,
    BBRecord,
    HeatmapPage,
    HeatmapResult,
    KernelHeatmap,
    analyze_heatmap,
    handler_build_flags,
    log_addresses,
    log_basic_blocks,
    run_analyzer,
)

try:
    __version__ = _get_version("omniprobe-ik")
except PackageNotFoundError:
    __version__ = "0.1.1"

__all__ = [
    "run_analyzer",
    "AnalyzerRun",
    "ANALYZER_PLUGINS",
    "handler_build_flags",
    "analyze_heatmap",
    "HeatmapResult",
    "KernelHeatmap",
    "HeatmapPage",
    "log_addresses",
    "AddressTrace",
    "AddressRecord",
    "log_basic_blocks",
    "BasicBlockTrace",
    "BBRecord",
    "ANALYSIS_PLUGINS",
    "Omniprobe",
    "NoInstrumentedKernelsError",
    "KernelFilterMatchedNothingError",
    "instrumentation_flags",
    "summarize_memory",
    "parse_uninstrumented_kernels",
    "uninstrumented_warning",
    "detect_offload_arch",
    "check_installation",
    "INSTALL_DOC",
    "MemoryAnalysisResult",
    "KernelMemoryResult",
    "MemoryAccess",
    "BasicBlockResult",
    "KernelBBResult",
    "BasicBlock",
]
