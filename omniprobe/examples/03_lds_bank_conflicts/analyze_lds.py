#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Omniprobe Example: LDS Bank Conflicts

Writes two versions of a kernel that stages data through a shared-memory (LDS)
tile and reads it back column-wise -- the access pattern of a matrix transpose.
In the first, each tile row is 32 floats wide, so a column is 32 words apart and
every lane of a half-wave hits the same LDS bank. The second pads each row to
33 floats, which spreads a column across all 32 banks.

Omniprobe attributes the conflicts to the exact source line. A kernel-level
profiler (e.g. metrix memory.lds_bank_conflicts) tells you conflicts exist;
this tells you which access causes them.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

LDS_KERNEL = """
#include <hip/hip_runtime.h>

#define ROWS 64

// Stage a 64x32 tile, then read it column-wise: lane l reads row l of column c.
template <int PITCH>
__device__ void transpose_tile(const float* in, float* out) {
    __shared__ float tile[ROWS * PITCH];
    int lane = threadIdx.x;
    const float* src = in + blockIdx.x * ROWS * 32;

    // Coalesced fill: consecutive lanes write consecutive words of a row.
    for (int k = lane; k < ROWS * 32; k += 64) {
        tile[(k / 32) * PITCH + (k % 32)] = src[k];
    }
    __syncthreads();

    float acc = 0.0f;
    for (int c = 0; c < 32; c++) {
        acc += tile[lane * PITCH + c];  // column read: the bank-conflict site
    }
    out[blockIdx.x * ROWS + lane] = acc;
}

__global__ void transpose_conflicting(const float* in, float* out) {
    transpose_tile<32>(in, out);
}

__global__ void transpose_padded(const float* in, float* out) {
    transpose_tile<33>(in, out);
}

int main() {
    const int blocks = 64;
    float *in, *out;
    hipMalloc(&in, sizeof(float) * blocks * 64 * 32);
    hipMalloc(&out, sizeof(float) * blocks * 64);
    hipMemset(in, 0, sizeof(float) * blocks * 64 * 32);
    hipLaunchKernelGGL(transpose_conflicting, dim3(blocks), dim3(64), 0, 0, in, out);
    hipLaunchKernelGGL(transpose_padded, dim3(blocks), dim3(64), 0, 0, in, out);
    hipDeviceSynchronize();
    return 0;
}
"""


def main():
    print("=" * 80)
    print("Omniprobe Example: LDS Bank Conflicts")
    print("=" * 80)

    with tempfile.TemporaryDirectory(prefix="omniprobe_example_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        kernel_file = tmp_path / "lds.hip"
        kernel_file.write_text(LDS_KERNEL)
        binary_file = tmp_path / "lds"

        # Memory analysis needs the address-message plugin; -g maps conflicts to lines.
        from omniprobe_ik import instrumentation_flags

        cmd = [
            "hipcc",
            str(kernel_file),
            "-o",
            str(binary_file),
            "-O2",
            "-g",
            *instrumentation_flags("memory"),
        ]
        print(f"\nCompiling: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}")
            return 1

        from omniprobe_ik import Omniprobe, summarize_memory

        print("Analyzing with Omniprobe...\n")
        analysis = Omniprobe().analyze_memory(command=str(binary_file))
        summary = summarize_memory(analysis)

        for kernel in ("transpose_conflicting", "transpose_padded"):
            sites = [s for s in summary["lds_sites"] if s["kernel"].startswith(kernel)]
            if not sites:
                print(f"{kernel}: no LDS bank conflicts")
                continue
            for s in sites:
                print(
                    f"{kernel}: {Path(s['source_file']).name}:{s['line']}:{s['column']} "
                    f"LDS {s['access_type']} -> {s['conflicts_per_execution']} extra cycles "
                    f"per wave execution ({s['bank_conflicts']} total)"
                )

        print("\nA column read of a 32-float-wide tile puts all 32 lanes of each half-wave")
        print("in one bank: 31 extra cycles per half, 62 per wave. Padding rows to 33")
        print("floats removes them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
