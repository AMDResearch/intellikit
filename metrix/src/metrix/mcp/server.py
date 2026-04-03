#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MCP Server for Metrix - Human-Readable GPU Metrics."""

import argparse

from fastmcp import FastMCP

from metrix import Metrix

mcp = FastMCP("IntelliKit Metrix")


@mcp.tool()
def profile_metrics(command: str, metrics: list[str] = None) -> dict:
    """
    Profile GPU application and collect hardware performance metrics.

    Returns human-readable metrics like memory bandwidth utilization,
    compute utilization, cache hit rates, etc. Use this to understand
    kernel performance characteristics and identify bottlenecks.

    Args:
        command: Command to profile (e.g., './app')
        metrics: List of metrics to collect (default: common metrics)

    Returns:
        Dictionary with kernels list containing metrics and durations
    """
    profiler = Metrix()

    # Use default common metrics if none specified
    if metrics is None:
        metrics = ["memory.hbm_bandwidth_utilization"]

    results_obj = profiler.profile(command, metrics=metrics)

    results = {"kernels": []}

    for kernel in results_obj.kernels:
        kernel_data = {
            "name": kernel.name,
            "duration_us_avg": float(kernel.duration_us.avg)
            if hasattr(kernel.duration_us, "avg")
            else 0.0,
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

        results["kernels"].append(kernel_data)

    return results


@mcp.tool()
def list_available_metrics() -> dict:
    """
    List all available GPU performance metrics.

    Returns a list of metric names that can be collected, organized by category.

    Returns:
        Dictionary with metrics list and category groupings
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
