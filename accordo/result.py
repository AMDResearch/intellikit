# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Result classes for Accordo validation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ArrayMismatch:
    """Represents a mismatch between reference and optimized arrays.

    Args:
            arg_index: Index of the argument that failed validation
            arg_name: Name of the argument
            arg_type: Type string of the argument
            max_difference: Maximum absolute difference between arrays
            mean_difference: Mean absolute difference between arrays
            reference_sample: Sample values from reference array
            optimized_sample: Sample values from optimized array
    """

    arg_index: int
    arg_name: str
    arg_type: str
    max_difference: float
    mean_difference: float
    reference_sample: np.ndarray
    optimized_sample: np.ndarray

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Mismatch in arg '{self.arg_name}' ({self.arg_type}): "
            f"max_diff={self.max_difference:.2e}, mean_diff={self.mean_difference:.2e}"
        )


@dataclass
class ValidationResult:
    """Result of Accordo validation.

    Args:
            is_valid: True if all arrays matched within tolerance
            error_message: Error message if validation failed
            mismatches: List of array mismatches
            matched_arrays: Dictionary of successfully matched arrays
            execution_time_ms: Execution times for reference and optimized kernels
            timeout_used: Timeout value used (if applicable)
    """

    is_valid: bool
    error_message: Optional[str] = None
    mismatches: list[ArrayMismatch] = None
    matched_arrays: dict[str, dict] = None
    execution_time_ms: dict[str, float] = None
    timeout_used: Optional[float] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.mismatches is None:
            self.mismatches = []
        if self.matched_arrays is None:
            self.matched_arrays = {}
        if self.execution_time_ms is None:
            self.execution_time_ms = {}

    @property
    def num_arrays_validated(self) -> int:
        """Total number of arrays validated (matched + mismatched)."""
        return len(self.matched_arrays) + len(self.mismatches)

    @property
    def num_mismatches(self) -> int:
        """Number of array mismatches."""
        return len(self.mismatches)

    @property
    def success_rate(self) -> float:
        """Percentage of arrays that matched."""
        total = self.num_arrays_validated
        if total == 0:
            return 0.0
        return (len(self.matched_arrays) / total) * 100.0

    def summary(self) -> str:
        """Get a human-readable summary of validation results."""
        if self.is_valid:
            return (
                f"✓ Validation passed! {self.num_arrays_validated} arrays matched within tolerance."
            )
        else:
            lines = [f"✗ Validation failed: {self.error_message}"]
            if self.mismatches:
                lines.append(f"\nMismatched arrays ({len(self.mismatches)}):")
                for mismatch in self.mismatches:
                    lines.append(f"  - {mismatch}")
            return "\n".join(lines)

    def __str__(self) -> str:
        """String representation."""
        return self.summary()
