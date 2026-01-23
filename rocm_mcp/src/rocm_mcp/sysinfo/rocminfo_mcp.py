# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import logging

from mcp.server.fastmcp import FastMCP

from rocm_mcp.sysinfo import DeviceType, Rocminfo

# initialize server
mcp = FastMCP(
    name="rocminfo",
    instructions=("MCP server for querying ROCm GPU and system information."),
)
logger = logging.getLogger(mcp.name)
rocminfo = Rocminfo(logger=logger)


@mcp.tool()
def get_gpu_architecture() -> str:
    """Get the architecture of all GPUs in the system.

    Returns:
        str: Information about GPU architectures including name and marketing name.
    """
    try:
        result = rocminfo.get_agents()
        gpus = [agent for agent in result.agents if agent.device_type == DeviceType.GPU]

        if not gpus:
            return "No GPUs found in the system."

        output = []
        for i, gpu in enumerate(gpus):
            output.append(f"GPU {i}:")
            output.append(f"  Architecture: {gpu.name}")
            output.append(f"  Marketing Name: {gpu.marketing_name}")
            output.append(f"  Vendor: {gpu.vendor_name}")
            if gpu.compute_units:
                output.append(f"  Compute Units: {gpu.compute_units}")
            if gpu.max_clock_freq:
                output.append(f"  Max Clock Frequency: {gpu.max_clock_freq} MHz")
            output.append("")

        return "\n".join(output)
    except Exception as e:
        logger.exception("Failed to get GPU architecture: %s", str(e))
        return f"Failed to get GPU architecture: {e!s}"


@mcp.tool()
def get_all_agents() -> str:
    """Get information about all HSA agents (CPUs, GPUs, etc.) in the system.

    Returns:
        str: Detailed information about all HSA agents.
    """
    try:
        result = rocminfo.get_agents()

        if not result.agents:
            return "No HSA agents found in the system."

        output = []
        for agent in result.agents:
            output.append(f"Agent {agent.agent_number}:")
            output.append(f"  Name: {agent.name}")
            output.append(f"  Type: {agent.device_type.value}")
            output.append(f"  Marketing Name: {agent.marketing_name}")
            output.append(f"  Vendor: {agent.vendor_name}")
            output.append(f"  UUID: {agent.uuid}")
            if agent.profile:
                output.append(f"  Profile: {agent.profile}")
            if agent.compute_units:
                output.append(f"  Compute Units: {agent.compute_units}")
            if agent.max_clock_freq:
                output.append(f"  Max Clock Frequency: {agent.max_clock_freq} MHz")
            output.append("")

        return "\n".join(output)
    except Exception as e:
        logger.exception("Failed to get agent information: %s", str(e))
        return f"Failed to get agent information: {e!s}"


def main() -> None:
    """Main function to run the rocminfo MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
