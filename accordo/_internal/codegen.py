# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Metadata generation for Accordo kernel arguments."""

import json
import logging


def _get_type_size(type_str: str) -> int:
	"""Get size in bytes for a C++ type.

	Args:
		type_str: C++ type string (e.g., "float*", "int", "double")

	Returns:
		Size in bytes
	"""
	# Pointers are always 8 bytes on 64-bit systems
	if "*" in type_str:
		return 8

	# Map common scalar types to sizes
	type_sizes = {
		"char": 1,
		"signed char": 1,
		"unsigned char": 1,
		"short": 2,
		"unsigned short": 2,
		"int": 4,
		"unsigned int": 4,
		"unsigned": 4,
		"long": 8,
		"unsigned long": 8,
		"long long": 8,
		"unsigned long long": 8,
		"float": 4,
		"double": 8,
		"bool": 1,
		"size_t": 8,
		"std::size_t": 8,
		"int8_t": 1,
		"uint8_t": 1,
		"int16_t": 2,
		"uint16_t": 2,
		"int32_t": 4,
		"uint32_t": 4,
		"int64_t": 8,
		"uint64_t": 8,
		"__half": 2,
		"__hip_bfloat16": 2,
	}

	# Remove const/volatile qualifiers
	type_clean = type_str.replace("const", "").replace("volatile", "").strip()

	# Look up in table
	if type_clean in type_sizes:
		return type_sizes[type_clean]

	# Default to 8 bytes if unknown (conservative estimate)
	logging.warning(f"Unknown type size for '{type_str}', assuming 8 bytes")
	return 8


def generate_kernel_metadata(args: list[str]) -> str:
	"""Generate JSON metadata file for kernel arguments.

	This replaces the old header file generation with runtime metadata.
	No rebuild of libaccordo.so is required when kernel signatures change.

	Args:
		args: List of argument type strings (e.g., ["double*", "const float*", "int"])

	Returns:
		Path to the generated metadata file
	"""
	import uuid

	metadata = {
		"version": 1,
		"description": "Accordo kernel argument metadata",
		"args": []
	}

	offset = 0
	for i, arg_type in enumerate(args):
		size = _get_type_size(arg_type)

		# Detect pointer and const qualifiers
		is_pointer = "*" in arg_type
		is_const = "const" in arg_type

		arg_info = {
			"index": i,
			"name": f"arg{i}",
			"type": arg_type,
			"size": size,
			"offset": offset,
			"is_pointer": is_pointer,
			"is_const": is_const,
			"is_output": is_pointer and not is_const  # Non-const pointers are outputs
		}

		metadata["args"].append(arg_info)
		offset += size

	metadata["total_size"] = offset

	# Write to JSON file with unique name to avoid stale files
	unique_id = uuid.uuid4().hex[:8]
	metadata_path = f"/tmp/accordo_metadata_{unique_id}.json"
	with open(metadata_path, "w") as f:
		json.dump(metadata, f, indent=2)

	logging.debug(f"Generated metadata file: {metadata_path}")
	logging.debug(f"Metadata content:\n{json.dumps(metadata, indent=2)}")

	return metadata_path
