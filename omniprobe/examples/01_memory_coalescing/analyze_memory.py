#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Omniprobe Example: Memory Coalescing Analysis

Writes a HIP kernel with two deliberately different access patterns to a
temporary directory, compiles it, and uses Omniprobe to show which one wastes
memory bandwidth.

The kernel does the same arithmetic twice. Only the indexing differs:

  * ``coalesced_copy``   — adjacent threads read adjacent addresses. One cache
    line serves the whole wavefront.
  * ``strided_copy``     — adjacent threads read addresses STRIDE apart. Each
    thread pulls its own cache line, so most of every line is fetched and
    discarded. This is what ``excess_cache_lines`` counts.

Run it and the strided kernel should report substantially more excess cache
lines than the coalesced one.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

STRIDED_KERNEL = """
#include <hip/hip_runtime.h>
#include <stdio.h>

#define STRIDE 16

// Adjacent threads touch adjacent elements: one cache line per wavefront.
__global__ void coalesced_copy(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

// Adjacent threads touch elements STRIDE apart: one cache line per thread.
__global__ void strided_copy(const float* in, float* out, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * STRIDE;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

int main() {
    const int N = 1024 * 1024;
    const size_t bytes = N * sizeof(float);

    float *d_in, *d_out;
    hipMalloc(&d_in, bytes);
    hipMalloc(&d_out, bytes);

    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;
    }
    hipMemcpy(d_in, h_in, bytes, hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(coalesced_copy, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_in, d_out, N);
    hipLaunchKernelGGL(strided_copy, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_in, d_out, N);
    hipDeviceSynchronize();

    printf("Both kernels completed\\n");

    free(h_in);
    hipFree(d_in);
    hipFree(d_out);

    return 0;
}
"""


def main():
    print("=" * 80)
    print("Omniprobe Example: Memory Coalescing Analysis")
    print("=" * 80)
    print()

    with tempfile.TemporaryDirectory(prefix="omniprobe_example_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Temp directory: {tmp_dir}")
        print()

        print("Step 1: Writing kernel...")
        kernel_file = tmp_path / "strided.hip"
        kernel_file.write_text(STRIDED_KERNEL)
        print(f"  Wrote: {kernel_file}")
        print()

        # -g matters here: Omniprobe reports findings against source_file:line:column,
        # and without debug info those fields come back empty.
        # Omniprobe only sees kernels compiled with its instrumentation plugin, and
        # each analysis needs its own plugin; instrumentation_flags() returns the
        # -fpass-plugin and --offload-arch flags for this one.
        print("Step 2: Compiling kernel with Omniprobe instrumentation (and -g)...")
        binary_file = tmp_path / "strided"
        try:
            from omniprobe_ik import instrumentation_flags

            flags = instrumentation_flags("memory")
        except (ImportError, FileNotFoundError, RuntimeError) as e:
            print(f"  Cannot locate Omniprobe's instrumentation plugin: {e}")
            print("  Compiling without it; Omniprobe will find nothing to analyze.")
            flags = []
        cmd = ["hipcc", str(kernel_file), "-o", str(binary_file), "-O2", "-g", *flags]
        print(f"  {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Compilation failed:\n{result.stderr}")
            return 1
        print(f"  Compiled: {binary_file}")
        print()

        print("Step 3: Analyzing memory access patterns with Omniprobe...")

        try:
            from omniprobe_ik import Omniprobe

            probe = Omniprobe()
            result = probe.analyze_memory(command=str(binary_file))

            print()
            print("=" * 80)
            print("MEMORY ACCESS FINDINGS")
            print("=" * 80)

            if not result.kernels:
                print("\nNo kernels reported. Check that the binary launched on the GPU.")
                return 0

            for kernel in result.kernels:
                print(f"\nKernel: {kernel.kernel_name} (dispatch {kernel.dispatch_id})")

                total_excess = sum(a.excess_cache_lines for a in kernel.accesses)
                total_conflicts = sum(a.bank_conflicts for a in kernel.accesses)
                print(f"  Accesses analyzed:  {len(kernel.accesses)}")
                print(f"  Excess cache lines: {total_excess}")
                print(f"  Bank conflicts:     {total_conflicts}")

                # Worst offenders first — this is the actionable part.
                worst = sorted(
                    kernel.accesses,
                    key=lambda a: a.excess_cache_lines,
                    reverse=True,
                )
                for access in worst[:5]:
                    if access.excess_cache_lines == 0:
                        continue
                    loc = f"{access.source_file}:{access.line}:{access.column}"
                    print(f"    {loc} -> {access.excess_cache_lines} excess cache lines")

            print()
            print("=" * 80)
            print("Expect strided_copy to show far more excess cache lines than")
            print("coalesced_copy — same arithmetic, different indexing.")
            print("=" * 80)

        except ImportError as e:
            print(f"  Could not import the Omniprobe API: {e}")
            print("  Ensure Omniprobe is installed and on PYTHONPATH.")
            print("  Running kernel directly to verify compilation...")
            result = subprocess.run([str(binary_file)], capture_output=True, text=True)
            print(result.stdout)
            if result.returncode == 0:
                print("  Kernel executed successfully")

        except Exception as e:
            print(f"  Omniprobe analysis failed: {e}")
            print("  Running kernel directly to verify compilation...")
            result = subprocess.run([str(binary_file)], capture_output=True, text=True)
            print(result.stdout)
            if result.returncode == 0:
                print("  Kernel executed successfully")

        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
