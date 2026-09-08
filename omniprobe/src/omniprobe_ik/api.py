#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Python API for Omniprobe GPU memory and basic-block analysis.

Wraps Omniprobe's Python API (``omniprobe.api``) to provide memory coalescing
analysis and basic-block ISA timing through IntelliKit's standard interface.

Omniprobe must be pre-installed on the system and its Python API importable
(i.e. the Omniprobe repo must be on ``PYTHONPATH`` or installed).

Example::

    from omniprobe_ik import Omniprobe

    probe = Omniprobe()
    result = probe.analyze_memory("./my_gpu_app")
    for kernel in result.kernels:
        for access in kernel.accesses:
            print(f"{access.source_file}:{access.line} bank_conflicts={access.bank_conflicts}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses (IntelliKit-style, mirroring upstream with clean names)
# ---------------------------------------------------------------------------


@dataclass
class MemoryAccess:
    """A single observed memory access from Omniprobe's memory analysis."""

    source_file: str
    line: int
    column: int
    excess_cache_lines: int
    bank_conflicts: int
    memory_space: str = ""
    access_type: str = ""
    execution_count: int = 0


@dataclass
class KernelMemoryResult:
    """Memory analysis results for a single kernel dispatch."""

    kernel_name: str
    dispatch_id: int
    accesses: list[MemoryAccess] = field(default_factory=list)


@dataclass
class MemoryAnalysisResult:
    """Top-level result from ``analyze_memory``."""

    kernels: list[KernelMemoryResult] = field(default_factory=list)


@dataclass
class BasicBlock:
    """Timing data for a single basic block."""

    bb_id: int
    source_file: str
    line: int
    min_cycles: float
    max_cycles: float
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    wave_count: int = 0


@dataclass
class KernelBBResult:
    """Basic-block analysis results for a single kernel dispatch."""

    kernel_name: str
    dispatch_id: int
    basic_blocks: list[BasicBlock] = field(default_factory=list)


@dataclass
class BasicBlockResult:
    """Top-level result from ``analyze_basic_blocks``."""

    kernels: list[KernelBBResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main API class
# ---------------------------------------------------------------------------


class Omniprobe:
    """High-level API for Omniprobe GPU memory and basic-block analysis.

    This class wraps Omniprobe's upstream Python API (``omniprobe.api.Omniprobe``)
    and converts results into IntelliKit-style dataclasses.

    Args:
        omniprobe_path: Explicit path to the ``omniprobe`` executable.
            If *None*, Omniprobe will locate it via ``PATH``.

    Raises:
        ImportError: If the ``omniprobe.api`` module is not importable.
        FileNotFoundError: If the ``omniprobe`` executable cannot be found.
    """

    def __init__(self, omniprobe_path: Optional[str] = None) -> None:
        try:
            from omniprobe.api import Omniprobe as _UpstreamOmniprobe
        except ImportError as exc:
            msg = (
                "Could not import omniprobe.api. "
                "Ensure the Omniprobe repository is on PYTHONPATH or installed."
            )
            raise ImportError(msg) from exc

        self._upstream = _UpstreamOmniprobe(omniprobe_path=omniprobe_path)
        logger.info("Omniprobe initialized (exe: %s)", self._upstream._exe)

    # -- public methods -----------------------------------------------------

    def analyze_memory(
        self,
        command: str,
        kernel_filter: Optional[str] = None,
        dispatches: str = "all",
    ) -> MemoryAnalysisResult:
        """Run memory analysis on *command* and return structured results.

        Args:
            command: Application command line to instrument.
            kernel_filter: Optional ECMAScript regex to limit which kernels
                are analysed.
            dispatches: Which dispatches to capture — ``"all"``, ``"random"``,
                or ``"1"``.

        Returns:
            A :class:`MemoryAnalysisResult` containing per-kernel memory
            access observations.
        """
        upstream_result = self._upstream.analyze_memory(
            command=command,
            kernel_filter=kernel_filter,
            dispatches=dispatches,
        )
        return self._convert_memory_result(upstream_result)

    def analyze_basic_blocks(
        self,
        command: str,
        kernel_filter: Optional[str] = None,
        dispatches: str = "all",
    ) -> BasicBlockResult:
        """Run basic-block analysis on *command* and return structured results.

        Args:
            command: Application command line to instrument.
            kernel_filter: Optional ECMAScript regex to limit which kernels
                are analysed.
            dispatches: Which dispatches to capture — ``"all"``, ``"random"``,
                or ``"1"``.

        Returns:
            A :class:`BasicBlockResult` containing per-kernel basic-block
            timing data.
        """
        upstream_result = self._upstream.analyze_basic_blocks(
            command=command,
            kernel_filter=kernel_filter,
            dispatches=dispatches,
        )
        return self._convert_bb_result(upstream_result)

    # -- conversion helpers -------------------------------------------------

    @staticmethod
    def _convert_memory_result(upstream) -> MemoryAnalysisResult:
        kernels = []
        for k in upstream.kernels:
            accesses = [
                MemoryAccess(
                    source_file=a.source_file,
                    line=a.line,
                    column=a.column,
                    excess_cache_lines=a.excess_cache_lines,
                    bank_conflicts=a.bank_conflicts,
                    memory_space=a.memory_space,
                    access_type=a.access_type,
                    execution_count=a.execution_count,
                )
                for a in k.accesses
            ]
            kernels.append(
                KernelMemoryResult(
                    kernel_name=k.kernel_name,
                    dispatch_id=k.dispatch_id,
                    accesses=accesses,
                )
            )
        return MemoryAnalysisResult(kernels=kernels)

    @staticmethod
    def _convert_bb_result(upstream) -> BasicBlockResult:
        kernels = []
        for k in upstream.kernels:
            blocks = [
                BasicBlock(
                    bb_id=b.bb_id,
                    source_file=b.source_file,
                    line=b.line,
                    min_cycles=b.min_cycles,
                    max_cycles=b.max_cycles,
                    p25=b.p25,
                    p50=b.p50,
                    p75=b.p75,
                    p95=b.p95,
                    p99=b.p99,
                    wave_count=b.wave_count,
                )
                for b in k.basic_blocks
            ]
            kernels.append(
                KernelBBResult(
                    kernel_name=k.kernel_name,
                    dispatch_id=k.dispatch_id,
                    basic_blocks=blocks,
                )
            )
        return BasicBlockResult(kernels=kernels)
