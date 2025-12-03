# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Code generation for Accordo C++ header files."""

import logging


def generate_kernel_header(args: list[str], additional_includes: list[str] = None) -> str:
	"""Generate C++ header file for kernel arguments.

	Args:
		args: List of argument type strings (e.g., ["double*", "const float*", "int"])
		additional_includes: Optional list of additional include directives

	Returns:
		Path to the generated header file
	"""
	if additional_includes is None:
		additional_includes = []

	header_path = "/tmp/KernelArguments.hpp"
	member_names = [f"arg{i}" for i in range(len(args))]
	members = ";\n    ".join(f"{arg} {name}" for arg, name in zip(args, member_names)) + ";"
	as_tuple_members = ", ".join(member_names)

	# Build includes section
	includes_section = "#include <tuple>\n"
	includes_section += "#include <hip/hip_fp16.h> // for float16\n"
	includes_section += "#include <hip/hip_bf16.h> // for bfloat16\n"

	if additional_includes:
		includes_section += "\n// User-provided includes\n"
		for include in additional_includes:
			includes_section += f"#include {include}\n"

	header_content = f"""#pragma once
{includes_section}
struct KernelArguments {{
    {members}

    auto as_tuple() const {{
        return std::tie({as_tuple_members});
    }}
}};
"""

	with open(header_path, "w") as header_file:
		header_file.write(header_content)

	logging.debug(f"Generated header file: {header_path}")
	logging.debug(f"Header content: {header_content}")
	return header_path
