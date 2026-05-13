# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import argparse
from typing import Annotated

from fastmcp import Context, FastMCP
from fastmcp.utilities.logging import get_logger
from pydantic import Field

from rocm_mcp.sysinfo import AmdSmi

# initialize server
mcp = FastMCP(
    name="amd-smi",
    instructions=("MCP server for querying AMD GPU system management information."),
)
logger = get_logger(mcp.name)
amd_smi = AmdSmi(logger=logger)


@mcp.tool()
async def get_driver_information(ctx: Annotated[Context, Field(description="MCP context.")]) -> str:
    """Get the AMD GPU driver version via the amdsmi Python API.

    Returns:
        str: The driver version, name, and date.
    """
    try:
        result = amd_smi.get_driver_information()
    except Exception as e:
        msg = f"Failed to get driver version: {e!s}"
        await ctx.error(msg)
        return msg
    else:
        return (
            f"Driver Name: {result.name}\n"
            f"Driver Version: {result.version}\n"
            f"Driver Date: {result.date}"
        )


def main() -> None:
    """Main function to run the amd-smi MCP server."""
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
        default=8001,
        help="Port to bind the HTTP server to (only used if transport is http)",
    )
    parser.add_argument(
        "--path",
        default="/rocm_mcp/amd_smi",
        help="Path to serve the HTTP server on (only used if transport is http)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
    main()
