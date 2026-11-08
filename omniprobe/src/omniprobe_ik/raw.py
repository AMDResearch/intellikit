# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Run any Omniprobe analyzer or bespoke handler, and the typed traces built on it.

Upstream's Python API wraps two analyzers. ``run_analyzer`` drives the
``omniprobe`` CLI directly so that every other analyzer -- and a message-handler
plugin written for the occasion -- is reachable today. ``analyze_heatmap``,
``log_addresses`` and ``log_basic_blocks`` return typed results: from Omniprobe's own Python API when
it provides them (Omniprobe 75ef62b and later), otherwise parsed from
``run_analyzer`` output. The field names are the same either way.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from .api import find_plugin_dir, parse_uninstrumented_kernels

# Analyzer names Omniprobe accepts with -a, and the plugin each needs at compile time.
ANALYZER_PLUGINS = {
    "MemoryAnalysis": "memory",
    "Heatmap": "memory",
    "AddressLogger": "memory",
    "BasicBlockAnalysis": "basic_blocks",
    "BasicBlockLogger": "basic_blocks",
    "PythonHandler": "memory",
}


@dataclass
class AnalyzerRun:
    """Everything one ``omniprobe`` run produced."""

    analyzer: str
    command: str
    returncode: int
    stdout: str
    stderr: str
    output: str = ""  # contents of the -o file (JSON formats); "" if none was written
    uninstrumented_kernels: list[str] = field(default_factory=list)

    @property
    def log(self) -> str:
        """stdout and stderr together. Since Omniprobe 821ba07 what the application
        and message handlers (C++ ``report()``, Python ``handleReport``) print is on
        ``stdout``, and Omniprobe's own messages and warnings are on ``stderr``."""
        return f"{self.stdout}\n{self.stderr}"

    def json_records(self) -> Iterator[dict]:
        """Records of a one-object-per-line (or JSON array) output, in order."""
        text = self.output.strip()
        if not text:
            return
        if text.startswith("["):
            yield from json.loads(text)
            return
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if line:
                yield json.loads(line)


def _omniprobe_exe(omniprobe_path: Optional[str]) -> str:
    exe = omniprobe_path or shutil.which("omniprobe")
    if exe is None:
        raise FileNotFoundError("omniprobe executable not found on PATH")
    return exe


def run_analyzer(
    analyzer: str,
    command: str,
    kernel_filter: Optional[str] = None,
    dispatches: str = "all",
    log_format: str = "json",
    extra_args: Optional[list[str]] = None,
    omniprobe_path: Optional[str] = None,
    check: bool = True,
    python_handler: Optional[str] = None,
) -> AnalyzerRun:
    """Run ``omniprobe -i -a <analyzer> ... -- <command>`` and return what it produced.

    Args:
        analyzer: An Omniprobe analyzer name (``MemoryAnalysis``, ``Heatmap``,
            ``AddressLogger``, ``BasicBlockAnalysis``, ``BasicBlockLogger``), the
            path to a message-handler plugin (``.so``) you built, or the path to a
            Python handler module (``.py``; same as ``python_handler``).
        command: Application command line; split with ``shlex``, so quoting works.
        kernel_filter: ECMAScript regex searched in the demangled kernel name.
        dispatches: ``"all"``, ``"random"`` or ``"1"``.
        log_format: ``"json"`` (written to a file, returned as ``output``) or
            ``"csv"`` (printed). What the application and message handlers print
            is in ``AnalyzerRun.stdout``; Omniprobe's own messages in ``stderr``.
        extra_args: More ``omniprobe`` options, e.g. ``["--library-filter", "f.json"]``.
        check: Raise ``RuntimeError`` if omniprobe exits non-zero.
        python_handler: Path of a Python module defining ``handleAddressMessage(msg)``,
            ``handleTimeMessage(msg)`` and optionally ``handleReport(msg)``; runs
            it through Omniprobe's ``PythonHandler`` (``analyzer`` is then ignored;
            build with the memory plugin). Each address message carries only
            ``address``, ``kernel`` and ``dispatch_id``. Omniprobe 13ca707 or later.

    The application must be built with the plugin the analyzer needs (see
    ``ANALYZER_PLUGINS`` and ``instrumentation_flags``).
    """
    exe = _omniprobe_exe(omniprobe_path)
    env = None
    if python_handler is None and analyzer.endswith(".py"):
        python_handler = analyzer
    if python_handler is not None:
        handler = Path(python_handler).resolve()
        if not handler.is_file():
            raise FileNotFoundError(f"Python handler not found: {handler}")
        analyzer = "PythonHandler"
        env = {**os.environ, "OMNIPROBE_PYTHON_HANDLER": str(handler)}
    if analyzer.endswith(".so"):
        analyzer = str(Path(analyzer).resolve())
    with tempfile.TemporaryDirectory(prefix="omniprobe_ik_") as tmp:
        out = Path(tmp) / "output.json"
        args = [exe, "-i", "-a", analyzer, "-t", log_format, "-d", dispatches]
        if log_format == "json":
            args += ["-o", str(out)]
        if kernel_filter is not None:
            args += ["-k", kernel_filter]
        args += list(extra_args or [])
        args += ["--", *shlex.split(command)]
        completed = subprocess.run(args, capture_output=True, text=True, env=env)
        output = out.read_text() if out.exists() else ""
    run = AnalyzerRun(
        analyzer=analyzer,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        output=output,
        uninstrumented_kernels=parse_uninstrumented_kernels(
            f"{completed.stdout}\n{completed.stderr}"
        ),
    )
    if python_handler is not None:
        # Omniprobe falls back to its default handler (exit 0) when the module is unusable.
        failed = [
            line for line in completed.stderr.splitlines() if "no Python handler installed" in line
        ]
        if failed and check:
            raise RuntimeError(f"Python handler not used: {failed[0]}")
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"omniprobe exited with code {completed.returncode}: "
            f"{(completed.stderr or completed.stdout)[-2000:]}"
        )
    return run


def handler_build_flags(omniprobe_path: Optional[str] = None) -> dict:
    """Compiler and linker flags for building a bespoke message-handler plugin.

    Uses the headers Omniprobe installs (``<prefix>/omniprobe/include/dh_comms``,
    Omniprobe 5cbb8e2 and later) or, failing that, its source tree, and links
    against ``libdh_comms``. Compile with ``hipcc`` (the headers need HIP's
    ``__device__``) as a shared library::

        hipcc -std=c++20 -fPIC -shared -O2 <cflags> handler.cc -o libhandler.so <ldflags>

    A handler includes ``dh_comms.h`` and ``message_handlers.h``.
    """
    lib = find_plugin_dir(omniprobe_path).parent
    if not (lib / "libdh_comms.so").is_file():
        raise FileNotFoundError(f"libdh_comms.so not found in {lib}")
    installed = lib.parent / "include" / "dh_comms"
    if (installed / "dh_comms.h").is_file():
        include = installed
    else:
        import omniprobe.api

        source = Path(omniprobe.api.__file__).resolve().parent.parent.parent
        include = source / "external" / "dh_comms" / "include"
        if not (include / "dh_comms.h").is_file():
            raise FileNotFoundError(
                "Omniprobe's dh_comms headers not found (install or source tree)"
            )
    cflags = [f"-I{include}"]
    ldflags = [f"-L{lib}", "-ldh_comms", f"-Wl,-rpath,{lib}"]
    return {
        "cflags": cflags,
        "ldflags": ldflags,
        "compile": "hipcc -std=c++20 -fPIC -shared -O2 "
        + " ".join(cflags)
        + " handler.cc -o libhandler.so "
        + " ".join(ldflags),
    }


# ---------------------------------------------------------------------------
# Typed results (field names agreed with the Omniprobe API request, 2026-11-08)
# ---------------------------------------------------------------------------


@dataclass
class HeatmapPage:
    start_address: int
    end_address: int
    accesses: int


@dataclass
class KernelHeatmap:
    kernel_name: str
    dispatch_id: int
    page_size: int
    pages: list[HeatmapPage] = field(default_factory=list)


@dataclass
class HeatmapResult:
    kernels: list[KernelHeatmap] = field(default_factory=list)
    uninstrumented_kernels: list[str] = field(default_factory=list)


@dataclass
class AddressRecord:
    """One wave execution of an instrumented memory instruction.

    ``memory_space``, ``access_size`` and ``source_file`` are empty/0 with
    Omniprobe older than 40a46b3, whose AddressLogger did not emit them.
    """

    kernel_name: str
    dispatch_id: int
    timestamp: int
    exec_mask: int
    active_lane_count: int
    block_idx: tuple[int, int, int]
    wave_num: int
    xcc_id: int
    se_id: int
    cu_id: int
    op_type: str
    memory_space: str
    access_size: int
    source_file: str
    line: int
    column: int
    addresses: list[int] = field(default_factory=list)


@dataclass
class AddressTrace:
    records: list[AddressRecord] = field(default_factory=list)
    truncated: bool = False
    uninstrumented_kernels: list[str] = field(default_factory=list)


@dataclass
class BBRecord:
    """One wave entering a basic block. ``basic_block_id`` is the id basic-block
    analysis reports (the message's ``user_data``)."""

    kernel_name: str
    dispatch_id: int
    timestamp: int
    exec_mask: int
    active_lane_count: int
    block_idx: tuple[int, int, int]
    wave_num: int
    xcc_id: int
    se_id: int
    cu_id: int
    basic_block_id: int
    source_file: str
    line: int
    column: int


@dataclass
class BasicBlockTrace:
    records: list[BBRecord] = field(default_factory=list)
    truncated: bool = False
    uninstrumented_kernels: list[str] = field(default_factory=list)


def _common(r: dict) -> dict:
    return {
        "kernel_name": r.get("kernel_name", r.get("kernel", "")),
        "dispatch_id": r.get("dispatch_id", 0),
        "timestamp": r.get("timestamp", 0),
        "exec_mask": r.get("exec", r.get("exec_mask", 0)),
        "active_lane_count": r.get("active_lane_count", 0),
        "block_idx": (r.get("block_idx_x", 0), r.get("block_idx_y", 0), r.get("block_idx_z", 0)),
        "wave_num": r.get("wave_num", 0),
        "xcc_id": r.get("xcc_id", 0),
        "se_id": r.get("se_id", 0),
        "cu_id": r.get("cu_id", 0),
        "source_file": r.get("source_file", ""),
        "line": r.get("dwarf_line", r.get("line", 0)),
        "column": r.get("dwarf_column", r.get("column", 0)),
    }


def _upstream_call(method: str, **kwargs):
    """Call upstream Omniprobe's typed method if its API has it; None otherwise."""
    from .api import Omniprobe

    probe = Omniprobe()
    func = getattr(probe._upstream, method, None)
    return None if func is None else func(**kwargs)


def _names(uninstrumented) -> list[str]:
    if isinstance(uninstrumented, dict):
        return [k for k in uninstrumented if not k.startswith("__amd_rocclr_")]
    return list(uninstrumented or [])


def _copy(cls, obj, **override):
    """Build an IntelliKit dataclass from an upstream object with the same field names."""
    import dataclasses

    values = {f.name: getattr(obj, f.name) for f in dataclasses.fields(cls) if hasattr(obj, f.name)}
    values.update(override)
    return cls(**values)


def analyze_heatmap(
    command: str, kernel_filter: Optional[str] = None, dispatches: str = "all"
) -> HeatmapResult:
    """Accesses per memory page (1 MiB pages) per kernel. Build with the memory plugin."""
    up = _upstream_call(
        "analyze_heatmap", command=command, kernel_filter=kernel_filter, dispatches=dispatches
    )
    if up is not None:
        return HeatmapResult(
            kernels=[
                _copy(KernelHeatmap, k, pages=[_copy(HeatmapPage, p) for p in k.pages])
                for k in up.kernels
            ],
            uninstrumented_kernels=_names(up.uninstrumented_kernels),
        )
    run = run_analyzer("Heatmap", command, kernel_filter, dispatches)
    result = HeatmapResult(uninstrumented_kernels=run.uninstrumented_kernels)
    for r in run.json_records():
        pages = [
            HeatmapPage(p["start_address"], p["end_address"], p["accesses"])
            for p in r.get("pages", [])
        ]
        size = pages[0].end_address - pages[0].start_address + 1 if pages else 0
        result.kernels.append(
            KernelHeatmap(r.get("kernel", ""), r.get("dispatch_id", 0), size, pages)
        )
    return result


def log_addresses(
    command: str,
    kernel_filter: Optional[str] = None,
    dispatches: str = "all",
    max_records: Optional[int] = None,
) -> AddressTrace:
    """Raw per-wave address trace (AddressLogger). Build with the memory plugin."""
    up = _upstream_call(
        "log_addresses",
        command=command,
        kernel_filter=kernel_filter,
        dispatches=dispatches,
        max_records=max_records,
    )
    if up is not None:
        return AddressTrace(
            records=[_copy(AddressRecord, r) for r in up.records],
            truncated=up.truncated,
            uninstrumented_kernels=_names(up.uninstrumented_kernels),
        )
    run = run_analyzer("AddressLogger", command, kernel_filter, dispatches)
    trace = AddressTrace(uninstrumented_kernels=run.uninstrumented_kernels)
    for r in run.json_records():
        if max_records is not None and len(trace.records) >= max_records:
            trace.truncated = True
            break
        trace.records.append(
            AddressRecord(
                **_common(r),
                op_type=r.get("op_type", ""),
                memory_space=r.get("memory_space", ""),
                access_size=r.get("access_size", 0),
                addresses=list(r.get("addresses", [])),
            )
        )
    return trace


def log_basic_blocks(
    command: str,
    kernel_filter: Optional[str] = None,
    dispatches: str = "all",
    max_records: Optional[int] = None,
) -> BasicBlockTrace:
    """Raw per-wave basic-block entry trace (BasicBlockLogger). Build with the BBStart plugin."""
    up = _upstream_call(
        "log_basic_blocks",
        command=command,
        kernel_filter=kernel_filter,
        dispatches=dispatches,
        max_records=max_records,
    )
    if up is not None:
        return BasicBlockTrace(
            records=[_copy(BBRecord, r) for r in up.records],
            truncated=up.truncated,
            uninstrumented_kernels=_names(up.uninstrumented_kernels),
        )
    run = run_analyzer("BasicBlockLogger", command, kernel_filter, dispatches)
    trace = BasicBlockTrace(uninstrumented_kernels=run.uninstrumented_kernels)
    for r in run.json_records():
        if max_records is not None and len(trace.records) >= max_records:
            trace.truncated = True
            break
        trace.records.append(
            BBRecord(**_common(r), basic_block_id=r.get("basic_block_id", r.get("user_data", 0)))
        )
    return trace
