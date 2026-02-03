# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import tempfile
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger
from pydantic import Field

from rocm_mcp.compile.hip_compiler import HipCompiler

# initialize server
mcp = FastMCP(
    name="hip_compiler",
    instructions=(
        "MCP server for compiling HIP C/C++ code to a binary executable via the ROCm HIP compiler."
    ),
)
logger = get_logger(mcp.name)
compiler = HipCompiler(logger=logger)


@mcp.tool()
async def compile_hip_source_file(
    ctx: Annotated[Context, Field(description="MCP context.")],
    source_file: Annotated[
        str | Path, Field(description="Path to the HIP C/C++ source file to compile.")
    ],
    output_file: Annotated[
        str | Path | None,
        Field(
            description=(
                "Path to save the compiled binary executable. If None, a temporary file is created."
            )
        ),
    ] = None,
    extra_flags: Annotated[
        list[str] | None, Field(description="Additional flags to pass to the HIP compiler.")
    ] = None,
) -> str:
    """Compile a source file written in HIP language to a binary executable.

    This function uses the ROCm HIP compiler to compile a source file written in HIP C/C++ into a
    binary executable.

    Returns:
        str: Message indicating success or failure of the compilation and the path to the binary
        executable.
    """
    source_file = Path(source_file)

    try:
        if output_file is None:
            with tempfile.TemporaryDirectory(delete=False) as tmpdir:
                output_file = Path(tmpdir) / source_file.stem
    except Exception as e:
        msg = f"Failed to create output file: {e!s}"
        await ctx.error(msg)
        return msg

    try:
        result = compiler.compile(
            source_file=source_file,
            output_file=output_file,
            extra_flags=extra_flags,
        )
    except Exception as e:
        msg = f"Compilation of {source_file} failed: {e!s}"
        await ctx.error(msg)
        return msg

    if not result.success:
        msg = f"Compilation of HIP code in {source_file} failed: {result.errors}"
        await ctx.error(msg)
        return msg

    msg = f"Compilation of HIP code in {source_file} succeeded. Executable at {output_file}"
    await ctx.info(msg)
    return msg


@mcp.tool()
async def compile_hip_source_string(
    ctx: Annotated[Context, Field(description="MCP context.")],
    source: Annotated[str, Field(description="HIP C/C++ source code as a string.")],
    output_file: Annotated[
        str | Path | None,
        Field(
            description=(
                "Path to save the compiled binary executable. If None, a temporary file is created."
            )
        ),
    ] = None,
    extra_flags: Annotated[
        list[str] | None, Field(description="Additional flags to pass to the HIP compiler.")
    ] = None,
) -> str:
    """Compile source provided as a string written in HIP language to a binary executable.

    This function uses the ROCm HIP compiler to compile a string with HIP C/C++ code into a
    binary executable.

    Returns:
        str: Message indicating success or failure of the compilation and the path to the binary
        executable.
    """
    try:
        with tempfile.TemporaryDirectory(delete=False) as tmpdir:
            source_file = Path(tmpdir) / "hip_source.cpp"
            if output_file is None:
                output_file = Path(tmpdir) / "hip_exe"
            with source_file.open("w") as f:
                f.write(source)
    except Exception as e:
        msg = f"Failed to create temporary source file: {e!s}"
        await ctx.error(msg)
        return msg

    try:
        result = compiler.compile(
            source_file=source_file,
            output_file=output_file,
            extra_flags=extra_flags,
        )
    except Exception as e:
        msg = f"Compilation of {source_file} failed: {e!s}"
        await ctx.error(msg)
        return msg

    if not result.success:
        msg = f"Compilation of HIP code in {source_file} failed: {result.errors}"
        await ctx.error(msg)
        return msg

    msg = f"Compilation of HIP code in {source_file} succeeded. Executable at {output_file}"
    await ctx.info(msg)
    return msg


def main() -> None:
    """Main function to run the HIP compiler MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
