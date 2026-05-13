# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""rocm-mcp - ROCm-based tools for use by LLM agents."""

from ._version import __version__
from .compile import HipCompiler, HipCompilerResult
from .doc import HipDocs
from .sysinfo import (
    AgentInfo,
    AmdSmi,
    DeviceType,
    DriverInformationResult,
    FirmwareEntry,
    GpuBadPageInfo,
    GpuFirmwareInfo,
    GpuInfo,
    GpuMetrics,
    GpuProcessInfo,
    GpuStaticInfo,
    ProcessEntry,
    Rocminfo,
    RocminfoResult,
)

__all__ = [
    "AgentInfo",
    "AmdSmi",
    "DeviceType",
    "DriverInformationResult",
    "FirmwareEntry",
    "GpuBadPageInfo",
    "GpuFirmwareInfo",
    "GpuInfo",
    "GpuMetrics",
    "GpuProcessInfo",
    "GpuStaticInfo",
    "HipCompiler",
    "HipCompilerResult",
    "HipDocs",
    "ProcessEntry",
    "Rocminfo",
    "RocminfoResult",
    "__version__",
]
