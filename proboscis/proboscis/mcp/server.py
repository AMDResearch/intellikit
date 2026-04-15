#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MCP Server for Proboscis — Agent-Driven GPU Kernel Instrumentation."""

from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from proboscis import Proboscis
from proboscis.planner import list_probe_types as _list_probe_types

mcp = FastMCP("IntelliKit Proboscis")


@mcp.tool()
def instrument_kernel(
    command: List[str],
    probe: str,
    target_kernel: Optional[str] = None,
    sample_rate: int = 1,
    max_records: int = 10000,
    log_level: int = 1,
) -> dict:
    """
    Instrument GPU kernels in a running application.

    Intercepts kernel launches, injects probe instrumentation using the
    hidden-argument ABI trick, runs the program, and returns structured
    results. No source code modification needed — works on compiled binaries.

    Args:
        command: Command to run as list (e.g., ['./vec_add'])
        probe: What to instrument — natural language description.
               Examples: "find memory accesses", "count basic block executions",
               "check register pressure"
        target_kernel: Optional kernel name pattern to filter (None = all kernels)
        sample_rate: Sample 1-in-N kernel dispatches (1 = every dispatch)
        max_records: Maximum number of probe records to collect
        log_level: Verbosity (0=quiet, 1=normal, 2=verbose)

    Returns:
        Dictionary with per-kernel instrumentation results:
        {
            "probe_type": "memory_trace",
            "command": ["./vec_add"],
            "kernels": {
                "vec_add(float*, float*, float*, int)": {
                    "records": [...],
                    "summary": {...}
                }
            }
        }
    """
    p = Proboscis(log_level=log_level)
    result = p.instrument(
        command=command,
        probe=probe,
        target_kernel=target_kernel,
        sample_rate=sample_rate,
        max_records=max_records,
    )

    output = {
        "probe_type": result.probe_type,
        "command": result.command,
        "kernels": {},
    }
    for kernel_result in result:
        output["kernels"][kernel_result.kernel_name] = {
            "records": kernel_result.records,
            "summary": kernel_result.summary,
            "record_count": kernel_result.record_count,
        }

    return output


@mcp.tool()
def list_probes() -> dict:
    """
    List available instrumentation probe types.

    Returns descriptions of each probe type including what it measures
    and the schema of the result records.
    """
    return _list_probe_types()


@mcp.tool()
def analyze_results(results_path: str) -> dict:
    """
    Load and summarize previously collected probe results.

    Args:
        results_path: Path to a saved proboscis results JSON file

    Returns:
        Summary of the instrumentation results
    """
    result = Proboscis.load(results_path)
    return {
        "probe_type": result.probe_type,
        "kernel_count": len(result),
        "kernels": {
            kr.kernel_name: {
                "record_count": kr.record_count,
                "summary": kr.summary,
            }
            for kr in result
        },
    }


def main():
    mcp.run()


if __name__ == "__main__":
    main()
