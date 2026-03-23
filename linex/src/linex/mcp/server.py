#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MCP Server for Linex - Source-Level GPU Performance Profiling."""

from mcp.server.fastmcp import FastMCP

from linex import Linex

mcp = FastMCP("IntelliKit Linex")


@mcp.tool()
def profile_application(command: str, kernel_filter: str = None, top_n: int = 10, launcher: str = None) -> dict:
    """
    Profile a GPU application and get source-level performance metrics.

    Returns cycle counts, stalls, and execution counts mapped to source lines.
    Use this to find performance hotspots and understand where GPU cycles are spent.

    Args:
        command: Command to profile (e.g., './my_app' or 'python script.py')
        kernel_filter: Optional regex filter for kernel names
        top_n: Number of top hotspots to return (default: 10)

    Returns:
        Dictionary with total_source_lines, total_instructions, and hotspots list
    """
    profiler = Linex()
    profiler.profile(command, kernel_filter=kernel_filter, launcher=launcher)

    results = {
        "distributed_context": {
            "global_rank": profiler.distributed_context.global_rank,
            "local_rank": profiler.distributed_context.local_rank,
            "world_size": profiler.distributed_context.world_size,
            "hostname": profiler.distributed_context.hostname,
            "launcher": profiler.distributed_context.launcher,
        },
        "total_source_lines": len(profiler.source_lines),
        "total_instructions": len(profiler.instructions),
        "hotspots": [],
        "per_rank_hotspots": [],
    }

    for i, line in enumerate(profiler.source_lines[:top_n], 1):
        results["hotspots"].append(
            {
                "rank": i,
                "file": line.file,
                "line_number": line.line_number,
                "source_location": line.source_location,
                "total_cycles": line.total_cycles,
                "stall_cycles": line.stall_cycles,
                "stall_percent": round(line.stall_percent, 2),
                "idle_cycles": line.idle_cycles,
                "execution_count": line.execution_count,
                "num_instructions": len(line.instructions),
            }
        )

    for rank_key, rank_profile in profiler.rank_profiles.items():
        rank_entry = {
            "rank_key": rank_key,
            "global_rank": rank_profile.global_rank,
            "local_rank": rank_profile.local_rank,
            "world_size": rank_profile.world_size,
            "hostname": rank_profile.hostname,
            "launcher": rank_profile.launcher,
            "ui_output_dir": rank_profile.ui_output_dir,
            "total_source_lines": len(rank_profile.source_lines),
            "total_instructions": len(rank_profile.instructions),
            "hotspots": [],
        }
        for i, line in enumerate(rank_profile.source_lines[:top_n], 1):
            rank_entry["hotspots"].append(
                {
                    "rank": i,
                    "file": line.file,
                    "line_number": line.line_number,
                    "source_location": line.source_location,
                    "total_cycles": line.total_cycles,
                    "stall_cycles": line.stall_cycles,
                    "stall_percent": round(line.stall_percent, 2),
                    "idle_cycles": line.idle_cycles,
                    "execution_count": line.execution_count,
                    "num_instructions": len(line.instructions),
                }
            )
        results["per_rank_hotspots"].append(rank_entry)

    return results


@mcp.tool()
def analyze_instruction_hotspots(
    command: str, kernel_filter: str = None, top_lines: int = 5, top_instructions_per_line: int = 10
) -> dict:
    """
    Get detailed instruction-level analysis for the hottest source lines.

    Shows ISA instructions with their cycle counts, stalls, and execution frequency.
    Use this to drill down into why specific lines are taking time.

    Args:
        command: Command to profile
        kernel_filter: Optional regex filter for kernel names
        top_lines: Number of hottest source lines to analyze (default: 5)
        top_instructions_per_line: Max instructions to show per line (default: 10)

    Returns:
        Dictionary with hotspot_analysis list containing ISA-level details
    """
    profiler = Linex()
    profiler.profile(command, kernel_filter=kernel_filter, launcher=launcher)

    results = {
        "distributed_context": {
            "global_rank": profiler.distributed_context.global_rank,
            "local_rank": profiler.distributed_context.local_rank,
            "world_size": profiler.distributed_context.world_size,
            "hostname": profiler.distributed_context.hostname,
            "launcher": profiler.distributed_context.launcher,
        },
        "hotspot_analysis": [],
        "per_rank_hotspot_analysis": [],
    }

    for line in profiler.source_lines[:top_lines]:
        # Sort instructions by latency
        sorted_insts = sorted(line.instructions, key=lambda x: x.latency_cycles, reverse=True)

        line_data = {
            "source_location": line.source_location,
            "total_cycles": line.total_cycles,
            "stall_percent": round(line.stall_percent, 2),
            "instructions": [],
        }

        for inst in sorted_insts[:top_instructions_per_line]:
            line_data["instructions"].append(
                {
                    "isa": inst.isa,
                    "latency_cycles": inst.latency_cycles,
                    "stall_cycles": inst.stall_cycles,
                    "stall_percent": round(inst.stall_percent, 2),
                    "idle_cycles": inst.idle_cycles,
                    "execution_count": inst.execution_count,
                    "instruction_address": f"0x{inst.instruction_address:08x}",
                }
            )

        results["hotspot_analysis"].append(line_data)

    for rank_key, rank_profile in profiler.rank_profiles.items():
        rank_entry = {
            "rank_key": rank_key,
            "global_rank": rank_profile.global_rank,
            "local_rank": rank_profile.local_rank,
            "world_size": rank_profile.world_size,
            "hostname": rank_profile.hostname,
            "launcher": rank_profile.launcher,
            "ui_output_dir": rank_profile.ui_output_dir,
            "hotspot_analysis": [],
        }
        for line in rank_profile.source_lines[:top_lines]:
            sorted_insts = sorted(line.instructions, key=lambda x: x.latency_cycles, reverse=True)
            line_data = {
                "source_location": line.source_location,
                "total_cycles": line.total_cycles,
                "stall_percent": round(line.stall_percent, 2),
                "instructions": [],
            }
            for inst in sorted_insts[:top_instructions_per_line]:
                line_data["instructions"].append(
                    {
                        "isa": inst.isa,
                        "latency_cycles": inst.latency_cycles,
                        "stall_cycles": inst.stall_cycles,
                        "stall_percent": round(inst.stall_percent, 2),
                        "idle_cycles": inst.idle_cycles,
                        "execution_count": inst.execution_count,
                        "instruction_address": f"0x{inst.instruction_address:08x}",
                    }
                )
            rank_entry["hotspot_analysis"].append(line_data)
        results["per_rank_hotspot_analysis"].append(rank_entry)

    return results


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
