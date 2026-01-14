#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Accordo Example: Template Kernel Validation

This example demonstrates validating C++ template kernels with different
type instantiations (float, double, int).
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from accordo import Accordo

# Template kernel - single instantiation per binary
FLOAT_KERNEL = """
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

template<typename T>
__global__ void scale_values(T* input, T* output, T factor, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * factor;
    }
}

int main() {
    const int N = 1024;
    float *d_in, *d_out;
    hipMalloc(&d_in, N * sizeof(float));
    hipMalloc(&d_out, N * sizeof(float));

    float* h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    hipMemcpy(d_in, h_in, N * sizeof(float), hipMemcpyHostToDevice);
    hipLaunchKernelGGL((scale_values<float>), dim3(4), dim3(256), 0, 0,
                       d_in, d_out, 2.0f, N);
    hipDeviceSynchronize();

    printf("Float kernel executed\\n");
    free(h_in);
    hipFree(d_in);
    hipFree(d_out);
    return 0;
}
"""

DOUBLE_KERNEL = """
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

template<typename T>
__global__ void scale_values(T* input, T* output, T factor, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * factor;
    }
}

int main() {
    const int N = 1024;
    double *d_in, *d_out;
    hipMalloc(&d_in, N * sizeof(double));
    hipMalloc(&d_out, N * sizeof(double));

    double* h_in = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) h_in[i] = (double)i;

    hipMemcpy(d_in, h_in, N * sizeof(double), hipMemcpyHostToDevice);
    hipLaunchKernelGGL((scale_values<double>), dim3(4), dim3(256), 0, 0,
                       d_in, d_out, 2.0, N);
    hipDeviceSynchronize();

    printf("Double kernel executed\\n");
    free(h_in);
    hipFree(d_in);
    hipFree(d_out);
    return 0;
}
"""


def compile_kernel(kernel_code, name, tmp_dir):
    """Compile kernel with debug symbols."""
    source_file = tmp_dir / f"{name}.hip"
    binary_file = tmp_dir / name

    source_file.write_text(kernel_code)
    print(f"Compiling {name}...", end=" ")

    cmd = ["hipcc", str(source_file), "-o", str(binary_file), "-O2", "-g"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("FAILED")
        print(result.stderr)
        sys.exit(1)

    print("OK")
    return binary_file


def main():
    print("=" * 80)
    print("Accordo Example: Template Kernel Validation")
    print("=" * 80)
    print()

    with tempfile.TemporaryDirectory(prefix="accordo_template_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Temp directory: {tmp_dir}")
        print()

        # Compile separate binaries for each template instantiation
        print("Step 1: Compiling kernels...")
        float_bin = compile_kernel(FLOAT_KERNEL, "scale_float", tmp_path)
        double_bin = compile_kernel(DOUBLE_KERNEL, "scale_double", tmp_path)
        print()

        # Test each template instantiation
        print("Step 2: Testing template instantiations...")
        print()

        test_configs = [
            ("float", float_bin, "scale_values"),
            ("double", double_bin, "scale_values"),
        ]

        results = {}

        for type_name, binary, kernel_base in test_configs:
            print(f"Testing {type_name} instantiation...")
            print(f"  Binary: {binary.name}")

            # Create validator
            try:
                validator = Accordo(binary=str(binary), kernel_name=kernel_base, working_directory=str(tmp_path))
                print(f"  Arguments: {[f'{name}:{type}' for name, type in validator.kernel_args]}")
            except Exception as e:
                print(f"  Failed to create validator: {e}")
                results[type_name] = None
                continue

            # Capture snapshots
            try:
                ref_snap = validator.capture_snapshot(binary=str(binary), timeout_seconds=10)
                opt_snap = validator.capture_snapshot(binary=str(binary), timeout_seconds=10)
                print(f"  Captured: {len(ref_snap.arrays)} array(s)")
            except Exception as e:
                print(f"  Failed to capture: {e}")
                results[type_name] = None
                continue

            # Compare
            result = validator.compare_snapshots(ref_snap, opt_snap, tolerance=1e-6)
            results[type_name] = result

            if result.is_valid:
                print("  ✓ PASS - Arrays matched within tolerance")
            else:
                print(f"  ✗ FAIL - {len(result.mismatches)} mismatch(es)")
            print()

        # Summary
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)

        valid_results = {k: v for k, v in results.items() if v is not None}
        passed = sum(1 for r in valid_results.values() if r.is_valid)
        total = len(valid_results)

        print(f"Template instantiations tested: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print()

        for type_name, result in results.items():
            if result is None:
                status = "✗ ERROR"
            elif result.is_valid:
                status = "✓ PASS"
            else:
                status = "✗ FAIL"
            print(f"  {type_name:10s} {status}")

        print("=" * 80)

        return 0 if all(r and r.is_valid for r in results.values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
