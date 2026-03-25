#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MCP Server for Samplex - GPU PC Sampling."""

import argparse

from fastmcp import FastMCP

from samplex import Samplex

mcp = FastMCP("IntelliKit Samplex")


@mcp.tool()
def pc_sample(
    command: str,
    method: str = "stochastic",
    interval: int = 65536,
    kernel_filter: str = None,
    top_n: int = 10,
) -> dict:
    """
    Run PC sampling on a GPU application to find instruction-level hotspots.

    Two methods available:
      - stochastic (default): Hardware-based, cycle-accurate, zero skid. Shows
        stall reasons, instruction types, wave counts. Requires MI300+.
      - host_trap: Software-based, time-based. Broader support (MI200+). No
        stall reasons or instruction types.

    Args:
        command: Command to profile (e.g., './app' or 'python3 bench.py')
        method: "stochastic" (MI300+, richer data) or "host_trap" (MI200+)
        interval: Sampling interval (default 65536). Stochastic: cycles. Host_trap: nanoseconds.
        kernel_filter: Regex to filter kernels by name
        top_n: Number of top instructions to report per kernel

    Returns:
        Dictionary with per-kernel instruction hotspots and analysis
    """
    sampler = Samplex()
    results = sampler.sample(
        command=command,
        interval=interval,
        method=method,
        kernel_filter=kernel_filter,
        top_n=top_n,
    )

    output = {
        "method": results.method,
        "total_samples": results.total_samples,
        "total_dispatches": results.total_dispatches,
        "interval": results.interval,
        "kernels": [],
    }

    for kernel in results.kernels:
        kernel_data = {
            "name": kernel.name,
            "total_samples": kernel.total_samples,
            "duration_us": round(kernel.duration_us, 2),
            "full_mask_pct": round(kernel.full_mask_pct, 2),
            "issued_pct": round(kernel.issued_pct, 2),
            "top_instructions": [
                {
                    "opcode": h.opcode,
                    "percentage": round(h.percentage, 2),
                    "sample_count": h.sample_count,
                    "issued_count": h.issued_count,
                    "stalled_count": h.stalled_count,
                }
                for h in kernel.top_instructions
            ],
        }
        output["kernels"].append(kernel_data)

    return output


@mcp.tool()
def list_pc_sampling_configs() -> str:
    """
    List available PC sampling configurations for all GPUs on the system.

    Shows supported methods, units, and interval ranges for each GPU.
    Use this to check hardware support before running pc_sample.

    Returns:
        Text listing of available configurations per GPU
    """
    sampler = Samplex()
    return sampler.list_configs()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--path", default="/samplex")
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
    main()
