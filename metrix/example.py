#!/usr/bin/env python3
"""
Metrix test script - profiles a tiny HIP kernel and prints all available counters.
Run on any AMD GPU to validate metrix works on that architecture.
"""
import sys
import subprocess
import tempfile
import os
from pathlib import Path


def compile_kernel(arch: str, workdir: str) -> str:
    """Compile a minimal vector_add HIP kernel."""
    src = os.path.join(workdir, "kernel.hip")
    binary = os.path.join(workdir, "vector_add")

    with open(src, "w") as f:
        f.write(r"""
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1024;
    float *a, *b, *c;
    hipMalloc(&a, N * sizeof(float));
    hipMalloc(&b, N * sizeof(float));
    hipMalloc(&c, N * sizeof(float));

    float ha[1024], hb[1024];
    for (int i = 0; i < N; i++) { ha[i] = i; hb[i] = i * 2; }
    hipMemcpy(a, ha, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(b, hb, N * sizeof(float), hipMemcpyHostToDevice);

    vector_add<<<(N+255)/256, 256>>>(a, b, c, N);
    hipDeviceSynchronize();

    float hc[1024];
    hipMemcpy(hc, c, N * sizeof(float), hipMemcpyDeviceToHost);
    printf("Result[0]=%.0f Result[1023]=%.0f\n", hc[0], hc[1023]);

    hipFree(a); hipFree(b); hipFree(c);
    return 0;
}
""")

    result = subprocess.run(
        ["hipcc", "--offload-arch=" + arch, "-o", binary, src],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"hipcc failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return binary


def main():
    from metrix import Metrix

    profiler = Metrix()
    arch = profiler.arch
    print(f"{'='*70}")
    print(f"  METRIX TEST — {profiler.backend.device_specs.name} ({arch})")
    print(f"{'='*70}")

    # List all available metrics
    all_metrics = profiler.list_metrics()
    print(f"\n  Available metrics ({len(all_metrics)}):")
    for m in sorted(all_metrics):
        try:
            info = profiler.get_metric_info(m)
            cat = info.get("category", "?")
            cat_name = cat.value if hasattr(cat, "value") else str(cat)
            unit = info.get("unit", "?")
            unit_name = unit.value if hasattr(unit, "value") else str(unit)
            print(f"    {m:45s}  [{cat_name:10s}]  ({unit_name})")
        except ValueError:
            print(f"    {m:45s}  [yaml-only ]  (no catalog entry)")

    # List profiles
    profiles = profiler.list_profiles()
    print(f"\n  Available profiles ({len(profiles)}): {', '.join(profiles)}")

    # Counter block limits
    limits = profiler.backend._get_counter_block_limits()
    if limits:
        print(f"\n  Counter block limits ({len(limits)} blocks):")
        for block, limit in sorted(limits.items()):
            print(f"    {block:10s}: {limit}")
    else:
        print("\n  Counter block limits: NONE (fallback chunking)")

    # Compile and profile
    with tempfile.TemporaryDirectory() as workdir:
        print(f"\n  Compiling vector_add for {arch}...")
        binary = compile_kernel(arch, workdir)
        print(f"  Compiled: {binary}")

        # 1) Time-only
        print(f"\n{'─'*70}")
        print("  TEST 1: Time-only profiling")
        print(f"{'─'*70}")
        results = profiler.profile(binary, time_only=True, num_replays=1)
        for k in results.kernels:
            print(f"    Kernel: {k.name}")
            print(f"    Duration: {k.duration_us.avg:.1f} μs")

        # 2) All metrics
        if all_metrics:
            print(f"\n{'─'*70}")
            print(f"  TEST 2: All {len(all_metrics)} metrics")
            print(f"{'─'*70}")
            try:
                results = profiler.profile(binary, num_replays=1)
                for k in results.kernels:
                    print(f"    Kernel: {k.name}")
                    print(f"    Duration: {k.duration_us.avg:.1f} μs")
                    for mname, mstat in sorted(k.metrics.items()):
                        print(f"      {mname:45s} = {mstat.avg:>12.4f}")
            except Exception as e:
                print(f"    ERROR: {e}")

        # 3) Each profile
        for pname in profiles:
            print(f"\n{'─'*70}")
            print(f"  TEST 3: Profile '{pname}'")
            print(f"{'─'*70}")
            try:
                results = profiler.profile(binary, profile=pname, num_replays=1)
                for k in results.kernels:
                    print(f"    Kernel: {k.name}")
                    for mname, mstat in sorted(k.metrics.items()):
                        print(f"      {mname:45s} = {mstat.avg:>12.4f}")
                if not results.kernels:
                    print(f"    (no kernels captured)")
            except Exception as e:
                print(f"    ERROR: {e}")

    print(f"\n{'='*70}")
    print(f"  DONE — {arch}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
