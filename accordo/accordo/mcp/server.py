#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MCP Server for Accordo - Automated Kernel Validation."""

import argparse
from typing import Optional

from fastmcp import FastMCP

from accordo import Accordo

mcp = FastMCP("IntelliKit Accordo")


def run_validate_kernel_correctness(
    kernel_name: str,
    reference_command: list[str],
    optimized_command: list[str],
    tolerance: Optional[float] = None,
    atol: float = 1e-08,
    rtol: float = 1e-05,
    equal_nan: bool = False,
    working_directory: str = ".",
) -> dict:
    """Run kernel correctness validation. Call this from Python; MCP tool wraps it."""
    validator = Accordo(
        binary=reference_command,
        kernel_name=kernel_name,
        kernel_args=None,
        working_directory=working_directory,
    )

    ref_snapshot = validator.capture_snapshot(binary=reference_command)
    opt_snapshot = validator.capture_snapshot(binary=optimized_command)
    result = validator.compare_snapshots(
        ref_snapshot,
        opt_snapshot,
        tolerance=tolerance,
        atol=atol,
        rtol=rtol,
        equal_nan=equal_nan,
    )

    return {
        "is_valid": result.is_valid,
        "num_arrays_validated": result.num_arrays_validated,
        "summary": result.summary(),
    }


@mcp.tool()
def validate_kernel_correctness(
    kernel_name: str,
    reference_command: list[str],
    optimized_command: list[str],
    tolerance: Optional[float] = None,
    atol: float = 1e-08,
    rtol: float = 1e-05,
    equal_nan: bool = False,
    working_directory: str = ".",
) -> dict:
    """
    Validate that an optimized kernel produces the same results as a reference.

    Captures outputs from both versions and compares them for correctness.
    Use this to verify kernel optimizations don't break functionality.

    Matching semantics: |a - b| <= atol + rtol * |b| (same as torch.allclose).

    Args:
        kernel_name: Name of the kernel to validate
        reference_command: Command for reference version as list (e.g., ['./ref'])
        optimized_command: Command for optimized version as list (e.g., ['./opt'])
        tolerance: Legacy alias for atol (overrides atol when set)
        atol: Absolute tolerance (default: 1e-08)
        rtol: Relative tolerance (default: 1e-05)
        equal_nan: Whether NaN values should compare equal (default: False)
        working_directory: Working directory for commands (default: '.')

    Returns:
        Dictionary with is_valid, num_arrays_validated, and summary
    """
    return run_validate_kernel_correctness(
        kernel_name=kernel_name,
        reference_command=reference_command,
        optimized_command=optimized_command,
        tolerance=tolerance,
        atol=atol,
        rtol=rtol,
        equal_nan=equal_nan,
        working_directory=working_directory,
    )


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
        default="/accordo",
        help="Path to serve the HTTP server on (only used if transport is http)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
    main()
