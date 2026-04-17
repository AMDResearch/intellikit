#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""MCP Server for Kerncap - GPU Kernel Extraction and Isolation."""

import argparse

from fastmcp import FastMCP

mcp = FastMCP("IntelliKit Kerncap")


@mcp.tool()
def profile_kernels(
    cmd: list[str],
    output_path: str | None = None,
) -> dict:
    """Profile an application and rank GPU kernels by execution time.

    Runs rocprofv3 --kernel-trace --stats on the given command and returns
    kernel statistics sorted by total duration.

    Args:
        cmd: Application command as list (e.g., ['./my_app', '--flag'])
        output_path: Optional path to write profile results as JSON

    Returns:
        Dictionary with ranked kernel statistics
    """
    from kerncap import Kerncap

    kc = Kerncap()
    kernels = kc.profile(cmd, output_path=output_path)
    return {
        "kernels": [
            {
                "name": k.name,
                "calls": k.calls,
                "total_duration_ns": k.total_duration_ns,
                "avg_duration_ns": k.avg_duration_ns,
                "percentage": k.percentage,
            }
            for k in kernels
        ]
    }


@mcp.tool()
def extract_kernel(
    kernel_name: str,
    cmd: str,
    source_dir: str | None = None,
    output: str | None = None,
    language: str | None = None,
    dispatch: int = -1,
) -> dict:
    """Extract a GPU kernel into a standalone reproducer.

    Captures a kernel dispatch at runtime and generates a self-contained
    project that can replay the kernel in isolation.

    Args:
        kernel_name: Kernel name (or substring) to capture
        cmd: Application command string (e.g., './my_app --flag')
        source_dir: Source directory to search for kernel source
        output: Output directory for reproducer
        language: Kernel language ('hip' or 'triton'), auto-detected if omitted
        dispatch: Dispatch index to capture (-1 = first match)

    Returns:
        Dictionary with extraction results
    """
    from kerncap import Kerncap

    kc = Kerncap()
    result = kc.extract(
        kernel_name=kernel_name,
        cmd=cmd,
        source_dir=source_dir,
        output=output,
        language=language,
        dispatch=dispatch,
    )
    return {
        "output_dir": result.output_dir,
        "language": result.language,
        "has_source": result.has_source,
        "generated_files": result.generated_files,
    }


@mcp.tool()
def validate_reproducer(
    reproducer_dir: str,
    tolerance: float = 1e-6,
    rtol: float = 1e-5,
    hsaco: str | None = None,
) -> dict:
    """Validate a reproducer by comparing outputs to captured reference.

    For HIP kernels (VA-faithful captures), baseline validation is a smoke
    test. With --hsaco, compares captured vs variant replay byte-for-byte.
    For Triton reproducers, compares outputs using tolerance.

    Args:
        reproducer_dir: Path to the reproducer project directory
        tolerance: Absolute tolerance for comparisons (default: 1e-6)
        rtol: Relative tolerance for comparisons (default: 1e-5)
        hsaco: Optional path to an alternative HSACO for variant comparison

    Returns:
        Dictionary with validation results
    """
    from kerncap import Kerncap

    kc = Kerncap()
    result = kc.validate(
        reproducer_dir=reproducer_dir,
        tolerance=tolerance,
        rtol=rtol,
        hsaco=hsaco,
    )
    return {
        "passed": result.passed,
        "max_error": result.max_error,
        "details": result.details,
    }


def main() -> None:
    """Run the MCP server."""
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
        default=8000,
        help="Port to bind the HTTP server to (only used if transport is http)",
    )
    parser.add_argument(
        "--path",
        default="/kerncap",
        help="Path to serve the HTTP server on (only used if transport is http)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
    main()
