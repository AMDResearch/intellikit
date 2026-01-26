# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""rocm-mcp - Tools for use by LLM agents."""

from importlib.metadata import version

__version__ = version("rocm-mcp")

from .compile import HipCompiler, HipCompilerResult
from .doc import HipDocs
from .sysinfo import AgentInfo, DeviceType, Rocminfo, RocminfoResult

__all__ = [
    "AgentInfo",
    "DeviceType",
    "HipCompiler",
    "HipCompilerResult",
    "HipDocs",
    "Rocminfo",
    "RocminfoResult",
    "__version__",
]
