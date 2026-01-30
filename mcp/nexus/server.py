#!/usr/bin/env python3.12
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MCP Server for Nexus - HSA Packet Source Code Extractor."""

import sys
from pathlib import Path

from fastmcp import FastMCP

# Add nexus to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nexus"))

from nexus import Nexus

mcp = FastMCP("IntelliKit Nexus")


@mcp.tool()
def extract_kernel_code(command: list[str], log_level: int = 1, include_assembly: bool = False) -> dict:
    """
    Extract GPU kernel source code and assembly from an application.
    
    Intercepts kernel launches and captures the actual code running on the GPU.
    Use this to see what kernels are launched, view their source code,
    examine generated assembly, or understand kernel compilation.
    
    Args:
        command: Command to run as list (e.g., ['python', 'app.py'])
        log_level: Logging level (0=quiet, 1=normal, 2=verbose)
        include_assembly: Include assembly code in results
    
    Returns:
        Dictionary with kernels list containing source and metadata
    """
    nexus = Nexus(log_level=log_level)
    trace = nexus.run(command)
    
    results = {"kernels": []}
    
    for kernel in trace:
        kernel_data = {
            "name": kernel.name,
            "language": getattr(kernel, "language", "unknown"),
            "num_instructions": len(kernel.assembly) if kernel.assembly else 0,
        }
        
        # Add source code if available
        if hasattr(kernel, "hip") and kernel.hip:
            kernel_data["hip_source"] = kernel.hip
        if hasattr(kernel, "triton") and kernel.triton:
            kernel_data["triton_source"] = kernel.triton
        
        # Add assembly if requested
        if include_assembly and kernel.assembly:
            kernel_data["assembly"] = kernel.assembly[:100]  # First 100 lines
            if len(kernel.assembly) > 100:
                kernel_data["assembly_truncated"] = True
        
        results["kernels"].append(kernel_data)
    
    return results


@mcp.tool()
def list_kernels(command: list[str]) -> dict:
    """
    List all GPU kernels launched by an application.
    
    Returns kernel names, source languages, and basic metadata.
    Use this to understand what kernels run in your application.
    
    Args:
        command: Command to run as list
    
    Returns:
        Dictionary with total_kernels and kernels list
    """
    nexus = Nexus(log_level=0)
    trace = nexus.run(command)
    
    results = {
        "total_kernels": len(trace.kernels),
        "kernels": [
            {
                "name": k.name,
                "language": getattr(k, "language", "unknown"),
                "has_source": bool(getattr(k, "hip", None) or getattr(k, "triton", None)),
                "has_assembly": bool(k.assembly),
            }
            for k in trace
        ],
    }
    
    return results


if __name__ == "__main__":
    mcp.run()
