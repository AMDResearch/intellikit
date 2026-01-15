# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""Accordo: Automated side-by-side correctness validation for GPU kernels.

Accordo uses kernelDB to automatically extract kernel arguments from compiled binaries,
eliminating the need for manual specification. Each validator instance is tied to a
specific kernel signature, with the library built once and reused for all captures.

Quick Example:
    >>> from accordo import Accordo
    >>>
    >>> # Create validator for a specific kernel (builds library once)
    >>> validator = Accordo(
    ...     binary="./app_ref",
    ...     kernel_name="reduce_sum"
    ... )
    >>>
    >>> # Capture snapshots from different binaries
    >>> ref = validator.capture_snapshot(binary="./app_ref")
    >>> opt = validator.capture_snapshot(binary="./app_opt")
    >>>
    >>> # Compare with specified tolerance
    >>> result = validator.compare_snapshots(ref, opt, tolerance=1e-6)
    >>> print(f"Valid: {result.is_valid}")

Efficient Example (multiple comparisons):
    >>> # Build once for this kernel signature
    >>> validator = Accordo(binary="./ref", kernel_name="my_kernel")
    >>>
    >>> # Capture reference once
    >>> ref = validator.capture_snapshot(binary="./ref")
    >>>
    >>> # Compare against multiple optimizations
    >>> for opt_bin in ["./opt1", "./opt2", "./opt3"]:
    ...     opt = validator.capture_snapshot(binary=opt_bin)
    ...     result = validator.compare_snapshots(ref, opt, tolerance=1e-6)
    ...     print(f"{opt_bin}: {'PASS' if result.is_valid else 'FAIL'}")

Multiple Kernels:
    >>> # Each kernel gets its own validator instance
    >>> reduce_val = Accordo(binary="./app", kernel_name="reduce_sum")
    >>> matmul_val = Accordo(binary="./app", kernel_name="matmul")
"""

# Public API exports
from .exceptions import (
	AccordoBuildError,
	AccordoError,
	AccordoProcessError,
	AccordoTimeoutError,
	AccordoValidationError,
)
from .kernel_args import extract_kernel_arguments, list_available_kernels
from .result import ArrayMismatch, ValidationResult
from .snapshot import Snapshot
from .validator import Accordo

# Version
__version__ = "0.4.0"

# Public API
__all__ = [
	"Accordo",
	"Snapshot",
	"ValidationResult",
	"ArrayMismatch",
	"extract_kernel_arguments",
	"list_available_kernels",
	"AccordoError",
	"AccordoBuildError",
	"AccordoTimeoutError",
	"AccordoProcessError",
	"AccordoValidationError",
]
