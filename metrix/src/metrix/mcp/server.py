#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MCP Server for Metrix - Human-Readable GPU Metrics."""

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
            "duration_us_avg": float(kernel.duration_us.avg) if hasattr(kernel.duration_us, 'avg') else 0.0,
            "metrics": {}
        }

        # Add metrics
        for metric_name in metrics:
            if hasattr(kernel, 'metrics') and metric_name in kernel.metrics:
                metric_obj = kernel.metrics[metric_name]
                kernel_data["metrics"][metric_name] = {
                    "avg": float(metric_obj.avg) if hasattr(metric_obj, 'avg') else 0.0,
                    "unit": getattr(metric_obj, 'unit', ''),
                }

        results["kernels"].append(kernel_data)

    return results


@mcp.tool()
def list_available_metrics() -> dict:
    """
    List all available GPU performance metrics.

    Returns a list of metric names that can be collected.

    Returns:
        Dictionary with metrics list
    """
    # Common ROCm metrics
    common_metrics = [
        "memory.hbm_bandwidth_utilization",
        "memory.l2_cache_hit_rate",
        "compute.cu_utilization",
        "compute.wave_occupancy",
    ]

    return {
        "metrics": common_metrics,
        "note": "Use profile_metrics with these metric names"
    }


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
