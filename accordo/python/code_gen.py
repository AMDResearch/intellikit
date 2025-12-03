################################################################################
# MIT License

# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

import logging


def generate_header(args: list[str]) -> str:
	header_path = "/tmp/KernelArguments.hpp"
	member_names = [f"arg{i}" for i in range(len(args))]
	members = ";\n    ".join(f"{arg} {name}" for arg, name in zip(args, member_names)) + ";"
	as_tuple_members = ", ".join(member_names)

	header_content = f"""#pragma once
#include <tuple>

// Datatypes
#include <hip/hip_fp16.h> // for float16
#include <hip/hip_bf16.h> // for bfloat16

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
