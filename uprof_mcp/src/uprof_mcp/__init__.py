# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""uprof_mcp - AMD UProf MCP tool."""

from ._version import __version__
from .uprof_profiler import UProfProfiler, UProfProfilerResult

__all__ = ["UProfProfiler", "UProfProfilerResult", "__version__"]
