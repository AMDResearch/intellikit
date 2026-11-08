#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Python API for Omniprobe GPU memory and basic-block analysis.

Wraps Omniprobe's Python API (``omniprobe.api``) to provide memory coalescing
analysis and basic-block ISA timing through IntelliKit's standard interface.

Omniprobe must be pre-installed on the system and its Python API importable
(i.e. the Omniprobe repo must be on ``PYTHONPATH`` or installed).

Omniprobe only analyzes kernels that were compiled with one of its LLVM
instrumentation plugins. Build the application with the flags returned by
:func:`instrumentation_flags` first; a binary built without them is reported
as :class:`NoInstrumentedKernelsError`, not as an empty result.

Example::

    from omniprobe_ik import Omniprobe, instrumentation_flags

    # hipcc my_gpu_app.hip -o my_gpu_app -g <flags>
    flags = instrumentation_flags("memory")

    probe = Omniprobe()
    result = probe.analyze_memory("./my_gpu_app")
    for kernel in result.kernels:
        for access in kernel.accesses:
            print(f"{access.source_file}:{access.line} bank_conflicts={access.bank_conflicts}")
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compile-time instrumentation
# ---------------------------------------------------------------------------

# Which Omniprobe LLVM pass plugin each analysis needs. The analyzers consume
# the messages a specific plugin injects, so a kernel compiled with the wrong
# plugin is as invisible to an analysis as one compiled with none.
ANALYSIS_PLUGINS = {
    "memory": "libAMDGCNSubmitAddressMessages-rocm.so",
    "basic_blocks": "libAMDGCNSubmitBBStart-rocm.so",
}


class NoInstrumentedKernelsError(RuntimeError):
    """Omniprobe ran but reported no instrumented kernel.

    Almost always means the application was compiled without the Omniprobe
    instrumentation plugin; see :func:`instrumentation_flags`.
    """


def _normalize_analysis(analysis: str) -> str:
    key = analysis.replace("-", "_")
    if key not in ANALYSIS_PLUGINS:
        choices = ", ".join(sorted(ANALYSIS_PLUGINS))
        raise ValueError(f"unknown analysis {analysis!r}; expected one of: {choices}")
    return key


def find_plugin_dir(omniprobe_path: Optional[str] = None) -> Path:
    """Locate the directory holding Omniprobe's instrumentation plugins.

    Honors ``OMNIPROBE_PLUGIN_DIR`` if set. Otherwise looks in ``lib/plugins``
    next to the ``omniprobe`` executable, which is where both Omniprobe's build
    tree and its install tree put them. The executable's path is tried as found
    first and then with symlinks resolved: in a build tree ``bin/omniprobe`` is
    a symlink into the source tree, so resolving it first would look in the
    wrong place.

    Raises:
        FileNotFoundError: If no plugin directory can be found.
    """
    override = os.environ.get("OMNIPROBE_PLUGIN_DIR")
    if override:
        return Path(override)

    exe = omniprobe_path or shutil.which("omniprobe")
    if exe is None:
        raise FileNotFoundError(
            "omniprobe executable not found on PATH, so its instrumentation plugins "
            "cannot be located; put omniprobe on PATH or set OMNIPROBE_PLUGIN_DIR"
        )

    candidates = [Path(exe).parent.parent / "lib" / "plugins"]
    candidates.append(Path(exe).resolve().parent.parent / "lib" / "plugins")
    for candidate in candidates:
        if any((candidate / name).is_file() for name in ANALYSIS_PLUGINS.values()):
            return candidate

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"no Omniprobe instrumentation plugins found (searched {searched}); "
        "set OMNIPROBE_PLUGIN_DIR to the directory holding libAMDGCNSubmit*-rocm.so"
    )


def detect_offload_arch() -> Optional[str]:
    """Return the gfx architecture of the first GPU ``rocminfo`` reports, or *None*."""
    rocminfo = shutil.which("rocminfo")
    if rocminfo is None:
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        candidate = Path(rocm_path) / "bin" / "rocminfo"
        rocminfo = str(candidate) if candidate.is_file() else None
    if rocminfo is None:
        return None

    try:
        output = subprocess.run([rocminfo], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    match = re.search(r"^\s*Name:\s+(gfx[0-9a-z]+)\s*$", output, re.MULTILINE)
    return match.group(1) if match else None


def instrumentation_flags(
    analysis: str = "memory",
    offload_arch: Optional[str] = None,
    omniprobe_path: Optional[str] = None,
) -> list[str]:
    """Return the ``hipcc`` flags that make kernels visible to an Omniprobe analysis.

    Omniprobe works by cloning each kernel at compile time and injecting
    instrumentation into the clone; at run time it swaps the clone in. Kernels
    compiled without these flags have no clone, and Omniprobe reports nothing
    for them. Add ``-g`` as well to get source-line attribution.

    Args:
        analysis: ``"memory"`` or ``"basic_blocks"`` (``"basic-blocks"`` is
            accepted too). Each needs a different plugin.
        offload_arch: GPU architecture, e.g. ``"gfx90a"``. Detected with
            ``rocminfo`` if omitted.
        omniprobe_path: Path to the ``omniprobe`` executable, used to find the
            plugins. Defaults to the one on ``PATH``.

    Returns:
        A list such as ``["-fpass-plugin=/.../libAMDGCNSubmitAddressMessages-rocm.so",
        "-fgpu-rdc", "--offload-arch=gfx90a"]``. Pass ``-fgpu-rdc`` to the link
        step too when compiling and linking separately.

    Raises:
        ValueError: Unknown analysis.
        FileNotFoundError: The plugins cannot be located.
        RuntimeError: No GPU architecture was given and none could be detected.
    """
    plugin = find_plugin_dir(omniprobe_path) / ANALYSIS_PLUGINS[_normalize_analysis(analysis)]
    if not plugin.is_file():
        raise FileNotFoundError(f"Omniprobe plugin not found: {plugin}")

    arch = offload_arch or detect_offload_arch()
    if arch is None:
        raise RuntimeError(
            "could not detect the GPU architecture with rocminfo; pass offload_arch, e.g. 'gfx90a'"
        )

    return [f"-fpass-plugin={plugin}", "-fgpu-rdc", f"--offload-arch={arch}"]


def _no_data_message(analysis: str, command: str, kernel_filter: Optional[str]) -> str:
    plugin = ANALYSIS_PLUGINS[analysis]
    try:
        flags = " ".join(instrumentation_flags(analysis))
    except (FileNotFoundError, RuntimeError):
        flags = (
            f"-fpass-plugin=<omniprobe>/lib/plugins/{plugin} -fgpu-rdc --offload-arch=<gfx arch>"
        )

    lines = [
        f"Omniprobe reported no instrumented kernels for {command!r}.",
        f"Omniprobe only sees kernels compiled with its {plugin} instrumentation plugin.",
        f"Rebuild the application with: hipcc -g {flags} ...",
        "(instrumentation_flags() / 'omniprobe-ik flags' print these flags.)",
    ]
    if kernel_filter is not None:
        lines.append(f"Also check that kernel_filter {kernel_filter!r} matches a launched kernel.")
    lines.append("Otherwise the application launched no GPU kernel.")
    return "\n".join(lines)


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

        Raises:
            NoInstrumentedKernelsError: No kernel was reported, typically
                because the application was not compiled with
                ``instrumentation_flags("memory")``.
        """
        upstream_result = self._upstream.analyze_memory(
            command=command,
            kernel_filter=kernel_filter,
            dispatches=dispatches,
        )
        result = self._convert_memory_result(upstream_result)
        if not result.kernels:
            raise NoInstrumentedKernelsError(_no_data_message("memory", command, kernel_filter))
        return result

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

        Raises:
            NoInstrumentedKernelsError: No kernel was reported, typically
                because the application was not compiled with
                ``instrumentation_flags("basic_blocks")``.
        """
        upstream_result = self._upstream.analyze_basic_blocks(
            command=command,
            kernel_filter=kernel_filter,
            dispatches=dispatches,
        )
        result = self._convert_bb_result(upstream_result)
        if not result.kernels:
            raise NoInstrumentedKernelsError(
                _no_data_message("basic_blocks", command, kernel_filter)
            )
        return result

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
