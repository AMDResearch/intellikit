#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MCP Server for Omniprobe - GPU Memory and Basic-Block Analysis."""

import argparse

from fastmcp import FastMCP

from omniprobe_ik import ANALYSIS_PLUGINS, Omniprobe, instrumentation_flags, summarize_memory
from omniprobe_ik.api import KNOWN_BLIND_SPOTS, uninstrumented_warning

mcp = FastMCP("IntelliKit Omniprobe")


@mcp.tool()
def analyze_memory(
    command: str,
    kernel_filter: str = None,
    dispatches: str = "all",
) -> dict:
    """Find the source lines whose memory accesses waste bandwidth: uncoalesced global accesses and LDS bank conflicts.

    First build the application with the flags from
    get_instrumentation_flags('memory') (plus -g); a normal build raises "no
    instrumented kernels". Use it after a kernel-level profiler (e.g. metrix
    coalescing efficiency or LDS bank conflicts) shows a problem, to find which
    access causes it.

    Args:
        command: Shell command to analyze (e.g., './my_gpu_app' or 'python train.py').
        kernel_filter: Optional ECMAScript regex to limit which kernels are analysed.
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
                "accesses": [
                    {
                        "source_file": a.source_file,
                        "line": a.line,
                        "column": a.column,
                        "excess_cache_lines": a.excess_cache_lines,
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
    """Report which basic blocks of a GPU kernel execute and how many waves reach each.

    First build the application with the flags from
    get_instrumentation_flags('basic_blocks') (plus -g) -- a different plugin
    from the memory analysis, so a separate build; a binary built for memory
    analysis must not be used here.

    Args:
        command: Shell command to analyze (e.g., './my_gpu_app').
        kernel_filter: Optional ECMAScript regex to limit which kernels are analysed.
        dispatches: Which dispatches to capture — 'all', 'random', or '1' (default: 'all').

    Returns:
        Dictionary with 'kernels' list. Each kernel contains:
        - kernel_name: GPU kernel function name
        - dispatch_id: Dispatch identifier
        - basic_blocks: List of basic blocks with bb_id, source location and
          wave_count (waves that reached the block). The cycle fields
          (min/max, p25-p99) are intervals dominated by instrumentation
          overhead, NOT per-block cost; use metrix/rocprof for time.
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
                        "min_cycles": b.min_cycles,
                        "max_cycles": b.max_cycles,
                        "p25": b.p25,
                        "p50": b.p50,
                        "p75": b.p75,
                        "p95": b.p95,
                        "p99": b.p99,
                        "wave_count": b.wave_count,
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
                    "Basic-block execution — which blocks run and how many waves "
                    "reach each (wave_count). Its cycle fields are not per-block "
                    "cost; use metrix/rocprof kernel timing for time hotspots."
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
