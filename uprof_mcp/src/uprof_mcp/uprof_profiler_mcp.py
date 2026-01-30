# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import tempfile
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger
from pydantic import Field

from uprof_mcp.uprof_profiler import UProfProfiler

# initialize server
mcp = FastMCP(
    name="uprof_profiler",
    instructions=("MCP server for profiling x86 CPU code for hotspots using AMD uProf."),
)
logger = get_logger(mcp.name)
profiler = UProfProfiler(logger=logger)


@mcp.tool()
async def profile_for_hotspots(
    ctx: Context,
    executable: Annotated[str | Path, Field(description="Path to the executable to be profiled.")],
    executable_arguments: Annotated[
        list[str], Field(description="Arguments to be passed to the executable.")
    ],
    output_dir: Annotated[
        str | Path | None, Field(description="Optional directory to store profiling results.")
    ] = None,
) -> str:
    """Profiles the x86 executable with the arguments to identify performance hotspots.

    Returns:
        str: A summary of the profiling results.
    """
    try:
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            result = profiler.find_hotspots(
                output_dir=output_dir,
                executable=executable,
                executable_args=executable_arguments,
            )
        else:
            with tempfile.TemporaryDirectory(delete=False) as tmpdir:
                result = profiler.find_hotspots(
                    output_dir=tmpdir,
                    executable=executable,
                    executable_args=executable_arguments,
                )

        await ctx.info(f"Profiling of {executable} completed with results in {result.report_path}.")
        with result.report_path.open() as file:
            return file.read()
    except Exception as e:
        msg = f"Profiling of {executable} failed: {e!s}"
        await ctx.error(msg)
        return msg


def main() -> None:
    """Main function to run the uProf profiler MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
