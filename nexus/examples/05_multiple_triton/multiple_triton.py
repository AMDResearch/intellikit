#!/usr/bin/env python3
"""
Example: Tracing Multiple Triton Kernels

This example demonstrates tracing multiple Triton kernels in a single execution.
We'll run two different kernels: vector addition and vector multiplication.
"""

import sys
import tempfile
from pathlib import Path

from nexus import Nexus


def main():
    print("Tracing multiple Triton kernels...\n")

    # Triton script with two different kernels
    triton_script = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    '''Vector addition kernel'''
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def mul_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    '''Vector multiplication kernel'''
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)

if __name__ == "__main__":
    size = 1024
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')

    # Run first kernel (addition)
    add_result = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, add_result, size, BLOCK_SIZE=1024)

    # Run second kernel (multiplication)
    mul_result = torch.empty_like(x)
    mul_kernel[grid](x, y, mul_result, size, BLOCK_SIZE=1024)

    print("Both Triton kernels executed successfully")
"""

    # Write the script to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(triton_script)
        triton_file = Path(f.name)

    try:
        # 1. Create Nexus instance
        nexus = Nexus(log_level=1)

        # 2. Run the script with Nexus tracing
        print("Running Triton script with Nexus...\n")
        trace = nexus.run(["python3", str(triton_file)])

        if not trace:
            print("ERROR: No kernels were traced!")
            return 1

        # 3. Process and display the trace
        print(f"Captured {len(trace)} kernel(s):")

        for idx, kernel in enumerate(trace, 1):
            # Filter out PyTorch internal kernels to focus on our Triton kernels
            if 'add_kernel' in kernel.name or 'mul_kernel' in kernel.name:
                print(f"\n{'='*80}")
                print(f"Kernel #{idx}: {kernel.name}")
                print(f"Signature: {kernel.signature}")
                print(f"{'='*80}")

                print(f"\nAssembly ({len(kernel.assembly)} instructions):")
                # Show first 10 and last 5 instructions for brevity
                for i, asm_line in enumerate(kernel.assembly[:10], 1):
                    print(f"  {i:3d}. {asm_line}")
                if len(kernel.assembly) > 15:
                    print(f"  ... ({len(kernel.assembly) - 15} more instructions) ...")
                    for i, asm_line in enumerate(kernel.assembly[-5:], len(kernel.assembly) - 4):
                        print(f"  {i:3d}. {asm_line}")

                print(f"\nHIP Source ({len(kernel.hip)} lines):")
                if kernel.hip:
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
        trace.save("multiple_triton_trace.json")
        print(f"Trace saved to multiple_triton_trace.json")
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        if triton_file.exists():
            triton_file.unlink()


if __name__ == "__main__":
    sys.exit(main())

