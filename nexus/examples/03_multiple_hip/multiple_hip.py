#!/usr/bin/env python3
"""
Example: Tracing Multiple HIP Kernels

This example demonstrates tracing multiple HIP kernels in a single execution.
We'll compile and run a program with two kernels: vector addition and vector multiplication.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from nexus import Nexus


def main():
    print("Tracing multiple HIP kernels...\n")

    # HIP source code with two kernels
    hip_code = """
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_multiply(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    // Initialize with some data
    float *h_a = new float[n];
    float *h_b = new float[n];
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

    // Launch first kernel (addition)
    hipLaunchKernelGGL(vector_add, dim3(4), dim3(256), 0, 0, d_a, d_b, d_c, n);
    hipDeviceSynchronize();

    // Launch second kernel (multiplication)
    hipLaunchKernelGGL(vector_multiply, dim3(4), dim3(256), 0, 0, d_a, d_b, d_c, n);
    hipDeviceSynchronize();

    std::cout << "Both HIP kernels executed successfully" << std::endl;

    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    delete[] h_a;
    delete[] h_b;

    return 0;
}
"""

    # Write to temporary files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as f:
        f.write(hip_code)
        cpp_file = Path(f.name)

    exe_file = cpp_file.with_suffix("")

    try:
        # 1. Compile the HIP code with debug symbols
        print("Compiling HIP code...")
        compile_result = subprocess.run(
            ["hipcc", "-g", str(cpp_file), "-o", str(exe_file)], capture_output=True, text=True
        )

        if compile_result.returncode != 0:
            print("ERROR: Compilation failed!")
            print(f"STDOUT: {compile_result.stdout}")
            print(f"STDERR: {compile_result.stderr}")
            return 1

        print("Compilation successful!\n")

        # 2. Create Nexus instance and run the executable
        nexus = Nexus(log_level=1)

        print("Running HIP executable with Nexus...\n")
        trace = nexus.run([str(exe_file)])

        if not trace:
            print("ERROR: No kernels were traced!")
            return 1

        # 3. Process and display the trace
        print(f"Captured {len(trace)} kernel(s):")

        for idx, kernel in enumerate(trace, 1):
            print(f"\n{'=' * 80}")
            print(f"Kernel #{idx}: {kernel.name}")
            print(f"Signature: {kernel.signature}")
            print(f"{'=' * 80}")

            print(f"\nAssembly ({len(kernel.assembly)} instructions):")
            for i, asm_line in enumerate(kernel.assembly, 1):
                print(f"  {i:3d}. {asm_line}")

            print(f"\nHIP Source ({len(kernel.hip)} lines):")
            if kernel.hip:
                # Use actual source line numbers if available
                if kernel.lines and len(kernel.lines) == len(kernel.hip):
                    for line_no, hip_line in zip(kernel.lines, kernel.hip):
                        print(f"  {line_no:3d}. {hip_line}")
                else:
                    # Fallback to sequential numbering
                    for i, hip_line in enumerate(kernel.hip, 1):
                        print(f"  {i:3d}. {hip_line}")
            else:
                print("  (No HIP source captured)")

            if kernel.files:
                print("\nSource Files:")
                unique_files = sorted(set(kernel.files))
                for file in unique_files:
                    print(f"  - {file}")

        print(f"\n{'=' * 80}")
        print("Example completed!")

        # Save to JSON for reference
        trace.save("multiple_hip_trace.json")
        print("Trace saved to multiple_hip_trace.json")
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        if cpp_file.exists():
            cpp_file.unlink()
        if exe_file.exists():
            exe_file.unlink()


if __name__ == "__main__":
    sys.exit(main())
