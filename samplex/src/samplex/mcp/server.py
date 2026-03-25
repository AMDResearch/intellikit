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
    method: str = "host_trap",
    interval: int = 1,
    kernel_filter: str = None,
    top_n: int = 10,
) -> dict:
    """
    Run PC sampling on a GPU application to find instruction-level hotspots.

    Captures statistical samples of the program counter while GPU kernels execute.
    Shows which instructions the GPU spends the most time on, revealing bottlenecks
    like memory waits, ALU dependencies, and barrier stalls.

    Two sampling methods:
    - host_trap: Reliable, time-based (microsecond intervals). Works on MI200+.
    - stochastic: Precise, cycle-based. Provides stall reasons. Requires MI300+.

    Args:
        command: Command to profile (e.g., './app' or 'python3 bench.py')
        method: Sampling method - "host_trap" or "stochastic"
        interval: Sampling interval (1 us for host_trap, 256+ cycles for stochastic)
        kernel_filter: Regex to filter kernels by name
        top_n: Number of top instructions to report per kernel

    Returns:
        Dictionary with per-kernel instruction hotspots and stall analysis
    """
    sampler = Samplex()
    results = sampler.sample(
        command=command,
        method=method,
        interval=interval,
        kernel_filter=kernel_filter,
        top_n=top_n,
    )

    output = {
        "total_samples": results.total_samples,
        "total_dispatches": results.total_dispatches,
        "method": results.method,
        "kernels": [],
    }

    for kernel in results.kernels:
        kernel_data = {
            "name": kernel.name,
            "total_samples": kernel.total_samples,
            "duration_us": round(kernel.duration_us, 2),
            "full_mask_pct": round(kernel.full_mask_pct, 2),
            "top_instructions": [
                {
                    "opcode": h.opcode,
                    "percentage": round(h.percentage, 2),
                    "sample_count": h.sample_count,
                }
                for h in kernel.top_instructions
            ],
        }

        if kernel.issued_pct is not None:
            kernel_data["issued_pct"] = round(kernel.issued_pct, 2)
            kernel_data["top_stall_reasons"] = kernel.top_stall_reasons

        output["kernels"].append(kernel_data)

    return output


@mcp.tool()
def list_pc_sampling_configs() -> str:
    """
    List available PC sampling configurations for all GPUs on the system.

    Shows supported methods (host_trap, stochastic), units, and interval ranges
    for each GPU. Use this to check hardware support before running pc_sample.

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
