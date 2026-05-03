#!/usr/bin/env python3
"""
Profile a sample HIP program with every metric defined in counter_defs.yaml
for the running architecture, and print the per-kernel values as a table.

Useful for:
  - sanity-checking that all defined counters fire on this GPU
  - getting a one-shot dump of "everything metrix knows" for an LLM-feedback
    workflow

Usage:
    python3 all_counters_demo.py                     # default sample program
    python3 all_counters_demo.py /path/to/binary     # any HIP binary
    python3 all_counters_demo.py --kernel REGEX ...  # filter kernels
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from metrix import Metrix


SAMPLE_HIP = r"""
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void k_noop() {}

__global__ void k_copy(const float* __restrict__ src,
                       float* __restrict__ dst,
                       size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = gridDim.x * blockDim.x;
    for (; i < N; i += s) dst[i] = src[i];
}

__global__ void k_compute_heavy(const float* __restrict__ src,
                                float* __restrict__ dst,
                                size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = gridDim.x * blockDim.x;
    for (; i < N; i += s) {
        float v = src[i];
        #pragma unroll
        for (int j = 0; j < 320; ++j) v = fmaf(v, 1.0001f, 0.0001f);
        dst[i] = v;
    }
}

__global__ void k_lds_bank_conflict(int* out) {
    __shared__ int s[32 * 32];
    int tid = threadIdx.x;
    s[tid * 32 % (32 * 32)] = tid;        // strided write -> bank conflicts
    __syncthreads();
    out[tid] = s[(tid * 7) % (32 * 32)];
}

int main() {
    const size_t N = 64 * 1024 * 1024;
    const size_t bytes = N * sizeof(float);
    float *src, *dst;
    int *iout;
    hipMalloc(&src, bytes);
    hipMalloc(&dst, bytes);
    hipMalloc(&iout, 1024 * sizeof(int));
    hipMemset(src, 1, bytes);

    const int block = 256;
    const int grid  = (N + block - 1) / block;

    k_noop<<<1, 1>>>();
    k_copy<<<grid, block>>>(src, dst, N);
    k_compute_heavy<<<grid, block>>>(src, dst, N);
    k_lds_bank_conflict<<<1, 1024>>>(iout);

    hipDeviceSynchronize();
    hipFree(src); hipFree(dst); hipFree(iout);
    printf("All kernels completed\n");
    return 0;
}
"""


def find_counter_defs() -> Path:
    here = Path(__file__).resolve()
    candidate = here.parent.parent / "src" / "metrix" / "backends" / "counter_defs.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"counter_defs.yaml not found at {candidate}")


def metrics_for_arch(arch: str) -> list[str]:
    with open(find_counter_defs()) as f:
        data = yaml.safe_load(f)
    out = []
    for c in data["rocprofiler-sdk"]["counters"]:
        for d in c["definitions"]:
            if arch in d.get("architectures", []) and "unsupported_reason" not in d:
                out.append(c["name"])
                break
    return out


def compile_or_use_binary(arg: str | None, work: Path) -> Path:
    if arg:
        p = Path(arg).resolve()
        if not p.exists():
            sys.exit(f"binary not found: {p}")
        return p
    src = work / "demo.hip"
    src.write_text(SAMPLE_HIP)
    binp = work / "demo"
    res = subprocess.run(["hipcc", str(src), "-o", str(binp), "-O2"],
                         capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"hipcc failed:\n{res.stderr}")
    return binp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", nargs="?", default=None,
                    help="HIP binary to profile (default: built-in sample)")
    ap.add_argument("--kernel", default=None,
                    help="regex to filter kernel names")
    ap.add_argument("--num-replays", type=int, default=1)
    args = ap.parse_args()

    prof = Metrix()
    arch = prof.arch
    print(f"# architecture: {arch}")
    print(f"# peak VRAM BW: {prof.backend.device_specs.hbm_bandwidth_gbs:.1f} GB/s")

    metrics = metrics_for_arch(arch)
    print(f"# collecting {len(metrics)} metrics defined for {arch}:")
    for m in metrics:
        print(f"#   - {m}")
    print()

    with tempfile.TemporaryDirectory(prefix="metrix_all_counters_") as td:
        work = Path(td)
        binp = compile_or_use_binary(args.binary, work)
        print(f"# binary: {binp}")
        print()

        results = prof.profile(
            command=str(binp),
            metrics=metrics,
            num_replays=args.num_replays,
            kernel_filter=args.kernel,
            cwd=str(work),
        )

    name_w = max((len(m) for m in metrics), default=20)
    kernels = list(results.kernels)
    if not kernels:
        print("(no kernels captured)")
        return 1

    header = f"{'metric':<{name_w}}  " + "  ".join(
        f"{(k.name.split('(')[0])[:24]:>24s}" for k in kernels
    )
    print(header)
    print("-" * len(header))
    for m in metrics:
        cells = []
        for k in kernels:
            v = k.metrics.get(m)
            cells.append(f"{v.avg:>24,.4g}" if v is not None else f"{'-':>24s}")
        print(f"{m:<{name_w}}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
