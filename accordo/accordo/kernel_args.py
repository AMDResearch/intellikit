# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Automatic kernel argument extraction using kernelDB."""

import logging
from pathlib import Path
from typing import List, Tuple

from .exceptions import AccordoError

try:
    from kerneldb import KernelDB

    KERNELDB_AVAILABLE = True
except ImportError:
    KERNELDB_AVAILABLE = False


def extract_kernel_arguments(
    binary_path: str, kernel_name: str, working_directory: str = "."
) -> List[Tuple[str, str]]:
    """Extract kernel arguments from a compiled binary using kernelDB.

    Args:
        binary_path: Path to the compiled binary (executable or .hsaco)
        kernel_name: Name of the kernel to extract arguments for
        working_directory: Working directory to resolve relative paths

    Returns:
        List of (name, type) tuples

    Raises:
        AccordoError: If kernelDB is not available or extraction fails

    Example:
        >>> args = extract_kernel_arguments("./my_app", "reduce_sum")
        >>> # Returns: [("input", "const float*"), ("output", "float*"), ("N", "int")]
    """
    if not KERNELDB_AVAILABLE:
        raise AccordoError(
            "kernelDB not available. Install it with: pip install git+https://github.com/AMDResearch/kerneldb.git"
        )

    # Resolve binary path
    if not Path(binary_path).is_absolute():
        binary_path = str(Path(working_directory) / binary_path)

    if not Path(binary_path).exists():
        raise AccordoError(f"Binary not found: {binary_path}")

    try:
        logging.debug(f"Loading binary with kernelDB: {binary_path}")
        kdb = KernelDB(binary_path)

        # Get all available kernels
        available_kernels = kdb.get_kernels()
        logging.debug(f"Found {len(available_kernels)} kernel(s) in binary")

        # Find matching kernel (exact match or pattern match)
        matching_kernels = [k for k in available_kernels if kernel_name in k]

        if not matching_kernels:
            raise AccordoError(
                f"Kernel '{kernel_name}' not found in binary. Available kernels: {', '.join(available_kernels)}"
            )

        if len(matching_kernels) > 1:
            logging.warning(
                f"Multiple kernels match '{kernel_name}': {matching_kernels}. Using first match: {matching_kernels[0]}"
            )

        matched_kernel = matching_kernels[0]
        logging.info(f"Extracting arguments from kernel: {matched_kernel}")

        # Extract arguments using kernelDB
        kdb_args = kdb.get_kernel_arguments(matched_kernel, resolve_typedefs=True)

        if not kdb_args:
            raise AccordoError(
                f"No argument information available for kernel '{matched_kernel}'. "
                "Make sure the binary was compiled with debug symbols (-g)."
            )

        # Convert kernelDB arguments to simple (name, type) tuples
        kernel_args = [(arg.name, arg.type_name) for arg in kdb_args]

        logging.info(
            f"Extracted {len(kernel_args)} argument(s): {[name for name, _ in kernel_args]}"
        )
        return kernel_args

    except Exception as e:
        if isinstance(e, AccordoError):
            raise
        raise AccordoError(f"Failed to extract kernel arguments: {str(e)}")


def list_available_kernels(binary_path: str, working_directory: str = ".") -> List[str]:
    """List all kernels available in a binary.

    Args:
        binary_path: Path to the compiled binary
        working_directory: Working directory to resolve relative paths

    Returns:
        List of kernel names found in the binary

    Raises:
        AccordoError: If kernelDB is not available or listing fails
    """
    if not KERNELDB_AVAILABLE:
        raise AccordoError(
            "kernelDB not available. Install it with: pip install git+https://github.com/AMDResearch/kerneldb.git"
        )

    # Resolve binary path
    if not Path(binary_path).is_absolute():
        binary_path = str(Path(working_directory) / binary_path)

    if not Path(binary_path).exists():
        raise AccordoError(f"Binary not found: {binary_path}")

    try:
        kdb = KernelDB(binary_path)
        return kdb.get_kernels()
    except Exception as e:
        raise AccordoError(f"Failed to list kernels: {str(e)}")
