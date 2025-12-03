# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Configuration classes for Accordo validation."""

from dataclasses import dataclass, field
from typing import Union


@dataclass
class KernelArg:
	"""Represents a kernel argument with semantic information.

	Args:
		name: Argument name (e.g., "result", "input", "count")
		type: C/C++ type string (e.g., "double*", "const float*", "int")

	Examples:
		>>> KernelArg(name="result", type="double*")
		>>> KernelArg(name="input", type="const double*")
		>>> KernelArg(name="count", type="unsigned long")

	Note:
		Output arguments are identified by checking for "*" without "const" in the type.
		This matches the existing IPC logic.
	"""

	name: str
	type: str

	@classmethod
	def from_string(cls, type_str: str, name: str = None) -> "KernelArg":
		"""Create KernelArg from a plain type string (backward compatibility).

		Args:
			type_str: C/C++ type string
			name: Optional argument name (auto-generated if not provided)

		Returns:
			KernelArg instance
		"""
		if name is None:
			name = f"arg_{id(type_str)}"  # Generate unique name
		return cls(name=name, type=type_str)

	@classmethod
	def from_dict(cls, d: dict) -> "KernelArg":
		"""Create KernelArg from a dictionary."""
		return cls(**d)


@dataclass
class ValidationConfig:
	"""Configuration for Accordo kernel validation.

	Args:
		kernel_name: Name of the kernel to validate
		kernel_args: List of kernel arguments (KernelArg, str, or dict)
		additional_includes: C++ include directives for custom types
		tolerance: Absolute tolerance for array comparison
		timeout_multiplier: Timeout = baseline_time_ms * timeout_multiplier
		log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")

	Examples:
		>>> config = ValidationConfig(
		...     kernel_name="my_kernel",
		...     kernel_args=[
		...         KernelArg(name="result", type="double*", direction="out"),
		...         KernelArg(name="input", type="const double*", direction="in"),
		...         "int"  # Plain string (backward compat)
		...     ],
		...     additional_includes=['"my_types.h"', '<hip/hip_bf16.h>'],
		...     tolerance=1e-6
		... )
	"""

	kernel_name: str
	kernel_args: list[Union[KernelArg, str, dict]]
	additional_includes: list[str] = field(default_factory=list)
	tolerance: float = 1e-6
	timeout_multiplier: float = 2.0
	log_level: str = "WARNING"

	def __post_init__(self):
		"""Normalize kernel_args to KernelArg instances."""
		normalized_args = []
		for i, arg in enumerate(self.kernel_args):
			if isinstance(arg, KernelArg):
				normalized_args.append(arg)
			elif isinstance(arg, str):
				# Convert plain string to KernelArg
				normalized_args.append(KernelArg.from_string(arg, name=f"arg{i}"))
			elif isinstance(arg, dict):
				# Convert dict to KernelArg
				if "name" not in arg:
					arg["name"] = f"arg{i}"
				normalized_args.append(KernelArg.from_dict(arg))
			else:
				raise TypeError(f"kernel_args must be KernelArg, str, or dict, got {type(arg)}")

		self.kernel_args = normalized_args

	def get_arg_types(self) -> list[str]:
		"""Get list of argument type strings (for backward compatibility)."""
		return [arg.type for arg in self.kernel_args]

	def get_arg_names(self) -> list[str]:
		"""Get list of argument names."""
		return [arg.name for arg in self.kernel_args]
