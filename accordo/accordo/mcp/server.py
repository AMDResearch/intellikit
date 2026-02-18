#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MCP Server for Accordo - Automated Kernel Validation."""

from fastmcp import FastMCP

from accordo import Accordo

mcp = FastMCP("IntelliKit Accordo")


@mcp.tool()
def validate_kernel_correctness(
    kernel_name: str,
    reference_command: list[str],
    optimized_command: list[str],
    tolerance: float = 1e-6,
    working_directory: str = ".",
) -> dict:
    """
    Validate that an optimized kernel produces the same results as a reference.

    Captures outputs from both versions and compares them for correctness.
    Use this to verify kernel optimizations don't break functionality.

    Args:
        kernel_name: Name of the kernel to validate
        reference_command: Command for reference version as list (e.g., ['./ref'])
        optimized_command: Command for optimized version as list (e.g., ['./opt'])
        tolerance: Numerical tolerance for comparisons (default: 1e-6)
        working_directory: Working directory for commands (default: '.')

    Returns:
        Dictionary with is_valid, num_arrays_validated, and summary
    """
    validator = Accordo(
        binary=reference_command,
        kernel_name=kernel_name,
        kernel_args=None,
        working_directory=working_directory,
    )

    # Capture snapshots
    ref_snapshot = validator.capture_snapshot(binary=reference_command)
    opt_snapshot = validator.capture_snapshot(binary=optimized_command)

    # Compare
    result = validator.compare_snapshots(ref_snapshot, opt_snapshot, tolerance=tolerance)

    return {
        "is_valid": result.is_valid,
        "num_arrays_validated": result.num_arrays_validated,
        "summary": result.summary(),
    }


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
