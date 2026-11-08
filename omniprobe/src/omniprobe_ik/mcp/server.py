#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MCP Server for Omniprobe - GPU Memory and Basic-Block Analysis."""

import argparse

from fastmcp import FastMCP

from omniprobe_ik import ANALYSIS_PLUGINS, Omniprobe, instrumentation_flags, summarize_memory
from omniprobe_ik.api import KNOWN_BLIND_SPOTS, uninstrumented_warning

mcp = FastMCP(
    "IntelliKit Omniprobe",
    instructions=(
        "Omniprobe names the exact source line (file:line:column) of each memory access that "
        "wastes bandwidth: uncoalesced or scattered global loads/stores (e.g. gathers through an "
        "index array) and LDS bank conflicts, with a measured cost per wave execution. Kernel-level "
        "profilers cannot do this: metrix gives per-kernel counters and linex gives stall cycles "
        "per line, but neither says which access is scattered. Use Omniprobe when a kernel shows "
        "low memory.global_load_efficiency (or store efficiency) or LDS bank conflicts in metrix, "
        "or when a kernel indexes memory indirectly, before concluding where the memory "
        "inefficiency is or rewriting code. Workflow: get_instrumentation_flags, rebuild with "
        "those flags plus -g, then analyze_memory on the rebuilt binary."
    ),
)


@mcp.tool()
def analyze_memory(
    command: str,
    kernel_filter: str = None,
    dispatches: str = "all",
) -> dict:
    """Name the source line of each memory access that wastes bandwidth (uncoalesced/scattered global access, LDS bank conflict), measured, worst first; build with get_instrumentation_flags first.

    Use this when metrix shows low memory.global_load_efficiency or
    global_store_efficiency (scattered accesses), LDS bank conflicts, or when a
    kernel reads through an index array (gather) -- to find WHICH access is
    responsible and how bad it is, instead of guessing from the source. A high
    achieved bandwidth does not rule this out: wasted bytes count as bandwidth.

    First build the application with the flags from
    get_instrumentation_flags('memory') plus -g; a normal build raises "no
    instrumented kernels". -O0 and pointers loaded from memory are analyzed
    (Omniprobe e807927 and later).

    Args:
        command: Shell command to analyze (e.g., './my_gpu_app' or 'python train.py').
        kernel_filter: Optional ECMAScript regex, searched anywhere in the demangled
            kernel name ('void name<4>(...)' for a template kernel, so '^name' misses
            it). If it matches no dispatched kernel, the error says so and lists them.
        dispatches: Which dispatches to capture — 'all', 'random', or '1' (default: 'all').

    Returns:
        Dictionary with:
        - summary: read this first. 'warnings' lists problems with coverage,
          e.g. kernels that ran without instrumentation and are missing. 'global_sites' ranked by wasted cache lines
          and 'lds_sites' ranked by bank conflicts (worst first, zero sites
          omitted), each with file/line/column, totals and a per-wave-execution
          value; per-kernel totals; and 'how_to_read' explaining the units.
        - kernels: the raw per-kernel access list (source location,
          excess_cache_lines, bank_conflicts, memory_space, access_type,
          execution_count).
    """
    probe = Omniprobe()
    result = probe.analyze_memory(
        command=command,
        kernel_filter=kernel_filter,
        dispatches=dispatches,
    )

    return {
        "summary": summarize_memory(result),
        "uninstrumented_kernels": result.uninstrumented_kernels,
        "kernels": [
            {
                "kernel_name": k.kernel_name,
                "dispatch_id": k.dispatch_id,
                "unanalyzed_flat_accesses": k.unanalyzed_flat_accesses,
                "unattributed_accesses": k.unattributed_accesses,
                "accesses": [
                    {
                        "source_file": a.source_file,
                        "line": a.line,
                        "column": a.column,
                        "excess_cache_lines": a.excess_cache_lines,
                        "cache_lines_needed": a.cache_lines_needed,
                        "cache_lines_used": a.cache_lines_used,
                        "bank_conflicts": a.bank_conflicts,
                        "memory_space": a.memory_space,
                        "access_type": a.access_type,
                        "execution_count": a.execution_count,
                    }
                    for a in k.accesses
                ],
            }
            for k in result.kernels
        ],
    }


@mcp.tool()
def analyze_basic_blocks(
    command: str,
    kernel_filter: str = None,
    dispatches: str = "all",
) -> dict:
    """Count exactly how often each basic block of a GPU kernel executes (wave entries, active lanes), with source lines. Not timing.

    First build the application with the flags from
    get_instrumentation_flags('basic_blocks') (plus -g) -- a different plugin
    from the memory analysis, so a separate build; a binary built for memory
    analysis must not be used here.

    Args:
        command: Shell command to analyze (e.g., './my_gpu_app').
        kernel_filter: Optional ECMAScript regex, searched anywhere in the demangled
            kernel name ('^name' misses a template kernel 'void name<4>(...)').
        dispatches: Which dispatches to capture — 'all', 'random', or '1' (default: 'all').

    Returns:
        Dictionary with 'kernels' list. Each kernel contains:
        - kernel_name: GPU kernel function name
        - dispatch_id: Dispatch identifier
        - basic_blocks: List of basic blocks with bb_id, source location,
          wave_visits (exact number of wave entries) and active_lanes (lanes
          summed over those entries; active_lanes / wave_visits = mean lane
          occupancy). No timing: use metrix/rocprof for where time goes.
    """
    probe = Omniprobe()
    result = probe.analyze_basic_blocks(
        command=command,
        kernel_filter=kernel_filter,
        dispatches=dispatches,
    )

    warning = uninstrumented_warning(result.uninstrumented_kernels)
    return {
        "warnings": [warning] if warning else [],
        "uninstrumented_kernels": result.uninstrumented_kernels,
        "kernels": [
            {
                "kernel_name": k.kernel_name,
                "dispatch_id": k.dispatch_id,
                "basic_blocks": [
                    {
                        "bb_id": b.bb_id,
                        "source_file": b.source_file,
                        "line": b.line,
                        "column": b.column,
                        "wave_visits": b.wave_visits,
                        "active_lanes": b.active_lanes,
                    }
                    for b in k.basic_blocks
                ],
            }
            for k in result.kernels
        ],
    }


@mcp.tool()
def get_instrumentation_flags(
    analysis: str = "memory", offload_arch: str = None, rdc: bool = False
) -> dict:
    """Get the hipcc flags that make kernels visible to an Omniprobe analysis.

    Omniprobe only analyzes kernels compiled with its LLVM instrumentation
    plugin. Call this before analyze_memory or analyze_basic_blocks and rebuild
    the application with the returned flags (plus -g for source lines).

    Args:
        analysis: 'memory' or 'basic_blocks'. Each needs a different plugin.
        offload_arch: GPU architecture such as 'gfx90a'; detected if omitted.
        rdc: Add -fgpu-rdc. Leave False unless the application already builds
            with relocatable device code: instrumentation does not need it, and
            with the plugin, RDC builds of two or more source files fail to link.

    Returns:
        Dictionary with 'flags' (list of hipcc arguments), 'plugin' (path of
        the pass plugin) and 'compile_example'.
    """
    flags = instrumentation_flags(analysis, offload_arch=offload_arch, rdc=rdc)
    return {
        "analysis": analysis,
        "flags": flags,
        "plugin": flags[0].split("=", 1)[1],
        "compile_example": "hipcc -g -O2 " + " ".join(flags) + " app.hip -o app",
        "note": (
            "Add these to the application's existing device-compile flags and keep its "
            "build mode (RDC or not). With rdc=True, also pass -fgpu-rdc to the link step."
        ),
    }


@mcp.tool()
def run_analyzer(
    analyzer: str,
    command: str,
    kernel_filter: str = None,
    dispatches: str = "all",
    max_output_chars: int = 20000,
) -> dict:
    """Run any Omniprobe analyzer, or a message handler you wrote (.so plugin or .py module), and return its raw output.

    Use for analyses analyze_memory/analyze_basic_blocks do not cover: 'Heatmap'
    (accesses per memory page), 'AddressLogger' (per-wave address trace),
    'BasicBlockLogger' (per-wave block entries), or the absolute path of a
    bespoke handler that counts something no stock analyzer does: a C++ plugin
    (.so) or a Python module (.py). A Python module defines handleAddressWave(msg)
    -- once per wave-level memory instruction, with addresses (one per active
    lane), lanes, exec, access_type, memory_space, access_size, source_file/line/
    column, block_idx, wave_num, cu_id, se_id, xcc_id, timestamp, kernel,
    dispatch_id -- and/or handleAddressMessage(msg) (once per active lane: address,
    kernel, dispatch_id), and optionally handleReport(msg) once per dispatch;
    kernel-argument loads arrive as memory_space 'constant'. Build the application
    with the plugin the analyzer needs first (memory plugin for Heatmap/
    AddressLogger/Python handlers, basic-block plugin for BasicBlockLogger).

    Returns:
        returncode, output (the analyzer's JSON, truncated to max_output_chars),
        stdout (what the application and handlers printed -- a handler's report
        is here), stderr (Omniprobe's own messages and warnings), both truncated
        from the front, truncated flag, and uninstrumented_kernels.
    """
    from omniprobe_ik import run_analyzer as _run

    run = _run(analyzer, command, kernel_filter=kernel_filter, dispatches=dispatches)
    return {
        "returncode": run.returncode,
        "output": run.output[:max_output_chars],
        "stdout": run.stdout[-max_output_chars:],
        "stderr": run.stderr[-max_output_chars:],
        "truncated": any(len(s) > max_output_chars for s in (run.output, run.stdout, run.stderr)),
        "uninstrumented_kernels": run.uninstrumented_kernels,
    }


@mcp.tool()
def list_analyzers() -> dict:
    """List available Omniprobe analysis types.

    Returns the analysis types that Omniprobe supports, along with
    descriptions and the MCP tool name to use for each.

    Returns:
        Dictionary with 'analyzers' list containing name, description,
        and tool name for each analysis type.
    """
    return {
        "analyzers": [
            {
                "name": "memory",
                "description": (
                    "Memory coalescing analysis — detects uncoalesced global "
                    "memory accesses, LDS bank conflicts, and excess cache lines."
                ),
                "tool": "analyze_memory",
                "plugin": ANALYSIS_PLUGINS["memory"],
            },
            {
                "name": "basic_blocks",
                "description": (
                    "Basic-block execution counts — exact wave entries and active "
                    "lanes per block, with source lines. Not timing: use "
                    "metrix/rocprof kernel timing for time hotspots."
                ),
                "tool": "analyze_basic_blocks",
                "plugin": ANALYSIS_PLUGINS["basic_blocks"],
            },
        ],
        "note": (
            "Use the 'tool' field value to call the corresponding analysis. The "
            "application must first be compiled with the analysis's plugin; "
            "get_instrumentation_flags returns the exact hipcc flags."
        ),
        "raw_analyzers": [
            {"name": "Heatmap", "plugin": "memory", "tool": "run_analyzer"},
            {"name": "AddressLogger", "plugin": "memory", "tool": "run_analyzer"},
            {"name": "BasicBlockLogger", "plugin": "basic_blocks", "tool": "run_analyzer"},
            {
                "name": "<path to your handler .so>",
                "plugin": "the one your handler expects",
                "tool": "run_analyzer",
            },
            {
                "name": "<path to your Python handler .py>",
                "plugin": "memory",
                "tool": "run_analyzer",
            },
        ],
        "known_blind_spots": KNOWN_BLIND_SPOTS,
    }


def main() -> None:
    """Run the MCP server."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport to use",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the HTTP server to (only used if transport is http)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server to (only used if transport is http)",
    )
    parser.add_argument(
        "--path",
        default="/omniprobe",
        help="Path to serve the HTTP server on (only used if transport is http)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
    main()
