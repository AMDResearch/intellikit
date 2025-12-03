# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""Accordo: Automated side-by-side correctness validation for GPU kernels.

Public API:
    - ValidationConfig: Configuration for kernel validation
    - KernelArg: Structured kernel argument representation
    - Snapshot: Captured kernel argument data from binary execution
    - ValidationResult: Result of validation with detailed metrics
    - ArrayMismatch: Information about array validation failures
    - Accordo: Main validator class for kernel validation
    - Exceptions: AccordoError, AccordoBuildError, AccordoTimeoutError, etc.

Quick Example (one-off validation):
    >>> from accordo import Accordo
    >>> config = Accordo.Config(
    ...     kernel_name="my_kernel",
    ...     kernel_args=[
    ...         Accordo.KernelArg(name="result", type="double*"),
    ...         Accordo.KernelArg(name="input", type="const double*"),
    ...     ],
    ...     tolerance=1e-6
    ... )
    >>> validator = Accordo(config)
    >>> result = validator.validate(
    ...     reference_binary=["./app_ref"],
    ...     optimized_binary=["./app_opt"],
    ...     working_directory=".",
    ...     baseline_time_ms=10.0
    ... )

Efficient Example (multiple optimizations vs same reference):
    >>> # Capture reference once (returns Snapshot object)
    >>> ref_snapshot = validator.capture_snapshot(
    ...     binary=["./app_ref"],
    ...     working_directory=".",
    ...     timeout_seconds=30
    ... )
    >>> print(ref_snapshot)  # Snapshot(binary='./app_ref', arrays=3, execution_time_ms=12.50)
    >>>
    >>> # Compare multiple optimizations
    >>> for opt_binary in optimized_binaries:
    ...     opt_snapshot = validator.capture_snapshot(
    ...         binary=opt_binary,
    ...         working_directory=".",
    ...         timeout_seconds=60
    ...     )
    ...     result = validator.compare_snapshots(ref_snapshot, opt_snapshot)
"""

# Public API exports
from .config import KernelArg, ValidationConfig
from .exceptions import (
	AccordoBuildError,
	AccordoError,
	AccordoProcessError,
	AccordoTimeoutError,
	AccordoValidationError,
)
from .result import ArrayMismatch, ValidationResult
from .snapshot import Snapshot
from .validator import Accordo as _Accordo

# Version
__version__ = "0.2.0"


# Nest all classes under Accordo namespace
class Accordo(_Accordo):
	"""Main Accordo validator with nested classes for clean API.

	All Accordo components are accessible as Accordo.ClassName:
	- Accordo.Config (ValidationConfig)
	- Accordo.KernelArg
	- Accordo.Snapshot
	- Accordo.Result (ValidationResult)
	- Accordo.ArrayMismatch
	- Accordo.Error (AccordoError)
	- Accordo.BuildError (AccordoBuildError)
	- Accordo.TimeoutError (AccordoTimeoutError)
	- Accordo.ProcessError (AccordoProcessError)
	- Accordo.ValidationError (AccordoValidationError)
	"""

	# Configuration
	Config = ValidationConfig
	KernelArg = KernelArg

	# Data structures
	Snapshot = Snapshot
	Result = ValidationResult
	ArrayMismatch = ArrayMismatch

	# Exceptions
	Error = AccordoError
	BuildError = AccordoBuildError
	TimeoutError = AccordoTimeoutError
	ProcessError = AccordoProcessError
	ValidationError = AccordoValidationError


# Public API
__all__ = [
	"Accordo",
]
