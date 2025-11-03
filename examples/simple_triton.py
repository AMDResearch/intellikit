#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Simple Triton Kernel Tracing Example

This example demonstrates how to use Nexus to trace a Triton kernel.
It creates a basic vector addition kernel using Triton and captures its assembly.
"""

from nexus import Nexus
import tempfile
import os
from pathlib import Path

def main():
    print("Tracing a simple Triton kernel...")

    # 1. Create a simple Triton script
    triton_script_content = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

if __name__ == "__main__":
    size = 2**10
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    output = add(x, y)
    print(f"Triton add completed. Output shape: {output.shape}")
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as f:
        f.write(triton_script_content)
        triton_file = Path(f.name)

    try:
        # 2. Use Nexus to run and trace the Triton script
        nexus = Nexus(log_level=1)  # Set log_level to 1 for info messages
        print(f"\nRunning Triton script with Nexus...")
        trace = nexus.run(["python3", str(triton_file)])

        # 3. Process and display the trace
        print(f"\nCaptured {len(trace)} kernel(s):")

        for kernel in trace:
            print(f"\n{'='*80}")
            print(f"Kernel: {kernel.name}")
            print(f"Signature: {kernel.signature}")
            print(f"{'='*80}")

            print(f"\nAssembly ({len(kernel.assembly)} instructions):")
            for i, asm_line in enumerate(kernel.assembly, 1):
                print(f"  {i:3d}. {asm_line}")

            print(f"\nHIP Source ({len(kernel.hip)} lines):")
            if kernel.hip:
                # Use actual source line numbers if available, otherwise enumerate
                if kernel.lines and len(kernel.lines) == len(kernel.hip):
                    for line_no, hip_line in zip(kernel.lines, kernel.hip):
                        print(f"  {line_no:3d}. {hip_line}")
                else:
                    for i, hip_line in enumerate(kernel.hip, 1):
                        print(f"  {i:3d}. {hip_line}")
            else:
                print("  (No HIP source captured)")

        print(f"\n{'='*80}")
        print("Example completed!")

        # Also save to JSON for reference
        trace.save("triton_trace.json")
        print(f"Trace saved to triton_trace.json")
        return 0

    finally:
        # Clean up temporary files
        if triton_file.exists():
            os.remove(triton_file)

if __name__ == "__main__":
    try:
        exit(main())
    except ImportError as e:
        if 'triton' in str(e).lower() or 'torch' in str(e).lower():
            print("ERROR: This example requires PyTorch and Triton to be installed.")
            print("Install them with: pip install torch triton")
            exit(1)
        raise
