#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Simple Source-Level GPU Profiling Example

This example demonstrates how to use Linex to profile a HIP kernel
and map cycle counts back to source code lines.
"""

from linex import Linex
import subprocess
import tempfile
from pathlib import Path


def main():
    print("Source-Level GPU Profiling with Linex...")

    # 1. Create temp directory for our example files (source + binary)
    example_dir = Path(tempfile.mkdtemp(prefix="linex_example_"))
    print(f"Example directory: {example_dir}")

    # 2. Create a simple HIP kernel source
    hip_source = """
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vector_add(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Some computation to generate interesting performance data
        float temp = a[idx] + b[idx];
        temp = temp * 2.0f;
        temp = temp - 1.0f;
        c[idx] = temp;
    }
}

int main() {
    // Use large size to ensure waves hit target CU
    const int N = 1024 * 1024;
    size_t bytes = N * sizeof(float);

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    // Launch kernel
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    hipLaunchKernelGGL(vector_add, grid, block, 0, 0, d_a, d_b, d_c, N);
    hipDeviceSynchronize();

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    printf("HIP kernel executed successfully!\\n");
    return 0;
}
"""

    # Save HIP source to file
    hip_file = example_dir / "vector_add.hip"
    hip_file.write_text(hip_source)
    print(f"HIP source written to: {hip_file}")

    exe_file = example_dir / "vector_add"

    # 3. Compile the HIP kernel with debug symbols
    compile_cmd = ["hipcc", str(hip_file), "-g", "-O2", "-o", str(exe_file)]
    print(f"Compiling HIP kernel: {' '.join(compile_cmd)}")
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if compile_result.returncode != 0:
        print(f"Compilation failed: {compile_result.stderr}")
        return 1
    print("Compilation successful.")

    # 4. Profile with Linex (profiling output goes to temp directory automatically)
    print("\nProfiling (this may take a moment)...")
    profiler = Linex()
    profiler.profile(
        command=str(exe_file),
        kernel_filter="vector_add",
    )

    # 5. Display results as a table
    print("\n" + "=" * 100)
    print("Source-Level Performance Results")
    print("=" * 100)

    # Filter to only our source file
    our_lines = [l for l in profiler.source_lines if "vector_add.hip" in l.file]

    # Read source file to show actual code
    source_lines_cache = {}
    try:
        with open(hip_file, "r") as f:
            source_lines_cache = {i + 1: line.rstrip() for i, line in enumerate(f.readlines())}
    except OSError:
        pass

    # Show table header
    print(
        f"\n{'HIP Source Line':<40} {'ISA Instruction':<35} {'Latency':>10} {'Stall':>10} {'Idle':>10} {'Hits':>8} {'Addr':>10}"
    )
    print(
        f"{'':40} {'':35} {'(cycles)':>10} {'(cycles)':>10} {'(cycles)':>10} {'(count)':>8} {'':>10}"
    )
    print("-" * 130)

    # Show each source line with its ISA instructions
    for line in our_lines[:5]:
        # Get actual source code text
        source_text = source_lines_cache.get(line.line_number, f"Line {line.line_number}")
        source_text = source_text[:37] + "..." if len(source_text) > 40 else source_text

        # First row: show source line with first ISA instruction
        first_inst = line.instructions[0] if line.instructions else None
        if first_inst:
            isa_text = first_inst.isa[:32] + "..." if len(first_inst.isa) > 35 else first_inst.isa
            print(
                f"{source_text:<40} {isa_text:<35} {first_inst.latency_cycles:>10,} {first_inst.stall_cycles:>10,} {first_inst.idle_cycles:>10,} {first_inst.execution_count:>8,} 0x{first_inst.instruction_address:08x}"
            )

        # Remaining instructions for this source line
        for inst in line.instructions[1:]:
            isa_text = inst.isa[:32] + "..." if len(inst.isa) > 35 else inst.isa
            print(
                f"{'':40} {isa_text:<35} {inst.latency_cycles:>10,} {inst.stall_cycles:>10,} {inst.idle_cycles:>10,} {inst.execution_count:>8,} 0x{inst.instruction_address:08x}"
            )

        print()

    return 0


if __name__ == "__main__":
    exit(main())
