#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MCP Server for Metrix - Human-Readable GPU Metrics."""

import argparse

from fastmcp import FastMCP

from metrix import Metrix

mcp = FastMCP("IntelliKit Metrix")

# Below this, a load (or store) instruction generates more L2 requests than even a
# fully coalesced 8-byte-per-lane access would (100 / 12.5 = 8 requests).
_SCATTERED_EFFICIENCY = 12.5

COALESCING_CAVEAT = (
    "memory.coalescing_efficiency assumes 4-byte accesses: a perfectly coalesced kernel on "
    "8-byte (double) data reads 25%, and a random gather can read higher than a strided "
    "access. Do not use it to rule coalescing in or out; use memory.global_load_efficiency "
    "and memory.global_store_efficiency, and Omniprobe to find the source line."
)


def _memory_notes(metrics: dict) -> list[str]:
    """Plain-language notes on a kernel's memory-pattern metrics, pointing to the next tool."""
    notes = []
    for metric, kind in (
        ("memory.global_load_efficiency", "load"),
        ("memory.global_store_efficiency", "store"),
    ):
        value = metrics.get(metric, {}).get("avg")
        if value is not None and 0 < value < _SCATTERED_EFFICIENCY:
            notes.append(
                f"Low {metric} ({value:.1f}%): about {100 / value:.0f} L2 requests per {kind} "
                "instruction, where a fully coalesced wave needs about 4 (4-byte elements), "
                "8 (8-byte) or 16 (16-byte). Far more than the element size explains means "
                f"scattered or uncoalesced {kind}s, e.g. through an index array. Achieved "
                "bandwidth includes the wasted bytes, so a high GB/s does not rule this out. "
                "To find which source line is responsible, use Omniprobe (omniprobe-mcp "
                "analyze_memory, after get_instrumentation_flags)."
            )
    conflicts = metrics.get("memory.lds_bank_conflicts", {}).get("avg")
    if conflicts is not None and conflicts > 0:
        notes.append(
            f"LDS bank conflicts ({conflicts:.2f} per LDS access). To find which LDS access "
            "causes them, use Omniprobe (omniprobe-mcp analyze_memory)."
        )
    return notes


@mcp.tool()
def profile_metrics(command: str, metrics: list[str] = None) -> dict:
    """
    Profile a GPU application and collect hardware performance metrics.

    Runs the given command under rocprofv3 and returns per-kernel metrics
    such as memory bandwidth utilization, cache hit rates, arithmetic
    intensity, and FLOP counts.

    Call list_available_metrics first to discover valid metric names.
    A kernel may carry 'notes' interpreting its memory-pattern metrics (e.g.
    scattered loads) and naming the tool that localises the problem; read them.

    Args:
        command: Shell command to profile (e.g., 'python train.py' or './my_app --size 1024').
                 The command is parsed with shell quoting rules, so quoted arguments are preserved.
        metrics: List of metric names to collect. Use names returned by list_available_metrics.
                 If omitted, collects all available metrics for the detected GPU architecture.

    Returns:
        Dictionary with a 'kernels' list. Each kernel entry contains:
        - name: GPU kernel function name
        - duration_us_avg: Average execution time of a single dispatch, in microseconds
        - dispatch_count: Number of times the kernel was launched per run
        - metrics: Dictionary mapping metric name to {avg, unit}
        - notes (when there is something to say): findings with a next step
        Top-level 'notes', when present, carries caveats about the metrics requested.
    """
    profiler = Metrix()

    # Collect all available metrics if none specified
    if metrics is None:
        metrics = profiler.list_metrics()

    results_obj = profiler.profile(command, metrics=metrics)

    results = {"kernels": []}

    for kernel in results_obj.kernels:
        kernel_data = {
            "name": kernel.name,
            "duration_us_avg": float(kernel.avg_time_us),
            "dispatch_count": int(kernel.dispatch_count),
            "metrics": {},
        }

        # Add metrics
        for metric_name in metrics:
            if hasattr(kernel, "metrics") and metric_name in kernel.metrics:
                metric_obj = kernel.metrics[metric_name]
                kernel_data["metrics"][metric_name] = {
                    "avg": float(metric_obj.avg) if hasattr(metric_obj, "avg") else 0.0,
                    "unit": getattr(metric_obj, "unit", ""),
                }

        notes = _memory_notes(kernel_data["metrics"])
        if notes:
            kernel_data["notes"] = notes
        results["kernels"].append(kernel_data)

    if "memory.coalescing_efficiency" in metrics and results["kernels"]:
        results["notes"] = [COALESCING_CAVEAT]
    return results


@mcp.tool()
def list_available_metrics() -> dict:
    """
    List all available GPU performance metrics that can be collected.

    Returns metric names organized by category. Use these names with the
    'metrics' parameter of profile_metrics to collect specific metrics,
    or omit 'metrics' to collect all of them.

    Categories include:
    - compute: FLOP counts, GFLOP/s throughput, arithmetic intensity
    - memory_bandwidth: HBM bandwidth utilization, read/write bandwidth, bytes transferred
    - memory_cache: L1/L2 hit rates, L2 bandwidth
    - memory_pattern: coalescing efficiency, load/store efficiency
    - memory_lds: LDS bank conflicts

    Returns:
        Dictionary with:
        - metrics: Flat list of all metric names
        - by_category: Metrics grouped by category
        - note: Usage hint
    """
    from metrix.metrics import METRIC_CATALOG

    # Query the backend for actually-supported metrics (includes YAML-defined
    # metrics that may not be in the Python METRIC_CATALOG).
    # Fall back to METRIC_CATALOG if the backend can't be initialized (no GPU).
    try:
        profiler = Metrix()
        metrics = sorted(profiler.list_metrics())
    except (RuntimeError, Exception):
        metrics = sorted(METRIC_CATALOG.keys())

    # Group by category for better discoverability
    by_category = {}
    for name in metrics:
        meta = METRIC_CATALOG.get(name)
        if meta is not None:
            cat = meta["category"].value
        else:
            # YAML-only metric not in Python catalog — use prefix as category
            cat = name.split(".", 1)[0] if "." in name else "other"
        by_category.setdefault(cat, []).append(name)

    return {
        "metrics": metrics,
        "by_category": by_category,
        "note": "Use profile_metrics with these metric names",
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
        default="/metrix",
        help="Path to serve the HTTP server on (only used if transport is http)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
    main()
