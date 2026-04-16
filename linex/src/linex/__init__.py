# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
linex: Source-Level SQTT Profiling

Maps rocprofv3 SQTT traces to source code granularity,
providing cycle counts and performance metrics per source line.
"""

from .api import Linex, SourceLine, InstructionData

from importlib.metadata import PackageNotFoundError, version as _get_version

try:
    __version__ = _get_version("linex")
except PackageNotFoundError:
    __version__ = "0.1.0"
__all__ = ["Linex", "SourceLine", "InstructionData"]
