# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Snapshot: Represents captured kernel argument data from a binary execution."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Snapshot:
	"""Represents a captured snapshot of kernel argument data.

	Attributes:
		arrays: List of numpy arrays containing kernel argument data
		execution_time_ms: Time taken to execute and capture the snapshot (milliseconds)
		binary: The binary command that was executed
		working_directory: The directory where the binary was executed
		grid_size: Optional grid dimensions dict with x,y,z (if available)
		block_size: Optional workgroup dimensions dict with x,y,z (if available)


	Example:
		>>> snapshot = Snapshot(
		...     arrays=[np.array([1, 2, 3]), np.array([4, 5, 6])],
		...     execution_time_ms=12.5,
		...     binary=["./my_app"],
		...     working_directory="/path/to/project",
		...     grid_size={"x": 1, "y": 1, "z": 1},
		...     block_size={"x": 256, "y": 1, "z": 1},
		... )
		>>> print(f"Captured {len(snapshot.arrays)} arrays in {snapshot.execution_time_ms}ms")
	"""

	arrays: List[np.ndarray]
	execution_time_ms: float
	binary: List[str]
	working_directory: str
	grid_size: Optional[dict] = None
	block_size: Optional[dict] = None

	def __repr__(self) -> str:
		"""Pretty representation of snapshot."""
		binary_str = " ".join(self.binary)
		return (
			f"Snapshot(binary='{binary_str}', "
			f"arrays={len(self.arrays)}, "
			f"execution_time_ms={self.execution_time_ms:.2f})"
		)

	def summary(self) -> str:
		"""Get a detailed summary of the snapshot."""
		binary_str = " ".join(self.binary)
		lines = [
			"Snapshot Summary:",
			f"  Binary: {binary_str}",
			f"  Working Directory: {self.working_directory}",
			f"  Execution Time: {self.execution_time_ms:.2f}ms",
			f"  Number of Arrays: {len(self.arrays)}",
		]

		if self.grid_size is not None:
			lines.append(
				f"  Grid Size: x={self.grid_size.get('x')}, y={self.grid_size.get('y')}, z={self.grid_size.get('z')}"
			)
		if self.block_size is not None:
			lines.append(
				f"  Block Size: x={self.block_size.get('x')}, y={self.block_size.get('y')}, z={self.block_size.get('z')}"
			)

		for i, arr in enumerate(self.arrays):
			lines.append(f"  Array {i}: shape={arr.shape}, dtype={arr.dtype}")

		return "\n".join(lines)
