#!/usr/bin/env python3.12
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MCP Server for Linex - Source-Level GPU Performance Profiling."""

import sys
from pathlib import Path

from fastmcp import FastMCP

# Add linex to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "linex" / "src"))

from linex import Linex

mcp = FastMCP("IntelliKit Linex")


@mcp.tool()
def profile_application(command: str, kernel_filter: str = None, top_n: int = 10) -> dict:
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
    profiler.profile(command, kernel_filter=kernel_filter)
    
    results = {
        "total_source_lines": len(profiler.source_lines),
        "total_instructions": len(profiler.instructions),
        "hotspots": [],
    }
    
    for i, line in enumerate(profiler.source_lines[:top_n], 1):
        results["hotspots"].append({
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
        })
    
    return results


@mcp.tool()
def analyze_instruction_hotspots(
    command: str, 
    kernel_filter: str = None, 
    top_lines: int = 5, 
    top_instructions_per_line: int = 10
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
    profiler.profile(command, kernel_filter=kernel_filter)
    
    results = {"hotspot_analysis": []}
    
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
            line_data["instructions"].append({
                "isa": inst.isa,
                "latency_cycles": inst.latency_cycles,
                "stall_cycles": inst.stall_cycles,
                "stall_percent": round(inst.stall_percent, 2),
                "idle_cycles": inst.idle_cycles,
                "execution_count": inst.execution_count,
                "instruction_address": f"0x{inst.instruction_address:08x}",
            })
        
        results["hotspot_analysis"].append(line_data)
    
    return results


if __name__ == "__main__":
    mcp.run()
