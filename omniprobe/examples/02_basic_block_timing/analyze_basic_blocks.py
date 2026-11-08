#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Omniprobe Example: Basic-Block Execution

Writes a HIP kernel with a data-dependent branch to a temporary directory,
compiles it, and uses Omniprobe to show which basic blocks execute and how many
waves reach each.

The counts are exact: wave entries per block and the lanes active in them.
Basic-block analysis does not time blocks; for where time goes, use metrix
kernel timing or linex per-line cycles.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

BRANCHING_KERNEL = """
#include <hip/hip_runtime.h>
#include <stdio.h>

// Two sides of the branch cost very different amounts of work, so they show up
// as basic blocks with clearly different cycle counts.
__global__ void divergent_work(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    float acc = in[idx];

    if ((idx & 1) == 0) {
        // Cheap path: a couple of multiplies.
        acc = acc * 2.0f + 1.0f;
    } else {
        // Expensive path: a short loop of transcendentals.
        for (int i = 0; i < 32; i++) {
            acc = __sinf(acc) + __cosf(acc);
        }
    }

    out[idx] = acc;
}

int main() {
    const int N = 256 * 1024;
    const size_t bytes = N * sizeof(float);

    float *d_in, *d_out;
    hipMalloc(&d_in, bytes);
    hipMalloc(&d_out, bytes);

    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_in[i] = 0.5f;
    }
    hipMemcpy(d_in, h_in, bytes, hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(divergent_work, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_in, d_out, N);
    hipDeviceSynchronize();

    printf("Kernel completed\\n");

    free(h_in);
    hipFree(d_in);
    hipFree(d_out);

    return 0;
}
"""


def main():
    print("=" * 80)
    print("Omniprobe Example: Basic-Block Execution")
    print("=" * 80)
    print()

    with tempfile.TemporaryDirectory(prefix="omniprobe_example_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Temp directory: {tmp_dir}")
        print()

        print("Step 1: Writing kernel...")
        kernel_file = tmp_path / "divergent.hip"
        kernel_file.write_text(BRANCHING_KERNEL)
        print(f"  Wrote: {kernel_file}")
        print()

        # -g maps each basic block back to a source line; without it, bb.source_file
        # and bb.line are empty and you only get block ids.
        # Omniprobe only sees kernels compiled with its instrumentation plugin, and
        # each analysis needs its own plugin; instrumentation_flags() returns the
        # -fpass-plugin and --offload-arch flags for this one.
        print("Step 2: Compiling kernel with Omniprobe instrumentation (and -g)...")
        binary_file = tmp_path / "divergent"
        try:
            from omniprobe_ik import instrumentation_flags

            flags = instrumentation_flags("basic_blocks")
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

        print("Step 3: Timing basic blocks with Omniprobe...")

        try:
            from omniprobe_ik import Omniprobe

            probe = Omniprobe()
            result = probe.analyze_basic_blocks(command=str(binary_file))

            print()
            print("=" * 80)
            print("BASIC-BLOCK EXECUTION")
            print("=" * 80)

            if not result.kernels:
                print("\nNo kernels reported. Check that the binary launched on the GPU.")
                return 0

            for kernel in result.kernels:
                print(f"\nKernel: {kernel.kernel_name} (dispatch {kernel.dispatch_id})")
                print(f"  Basic blocks: {len(kernel.basic_blocks)}")
                print()
                print(f"  {'BB':>4}  {'entries':>8}  {'lanes':>9}  {'lanes/entry':>11}  location")
                print(f"  {'-' * 4}  {'-' * 8}  {'-' * 9}  {'-' * 11}  {'-' * 20}")

                # Most-executed blocks first. Counts, not cost.
                reached = sorted(kernel.basic_blocks, key=lambda b: b.wave_visits, reverse=True)
                for bb in reached[:10]:
                    loc = f"{bb.source_file}:{bb.line}" if bb.source_file else "(no debug info)"
                    per = bb.active_lanes / bb.wave_visits if bb.wave_visits else 0.0
                    print(
                        f"  {bb.bb_id:>4}  {bb.wave_visits:>8}  {bb.active_lanes:>9}  {per:>11.1f}  {loc}"
                    )

            print()
            print("=" * 80)
            print("Each side of the branch is entered by every wave with half its lanes")
            print("(32 lanes/entry): that is divergence. Counts, not time.")
            print("=" * 80)

        except ImportError as e:
            print(f"  Could not import the Omniprobe API: {e}")
            print(
                "  Put <prefix>/omniprobe/bin of an Omniprobe install on PATH; omniprobe-ik check diagnoses."
            )
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
