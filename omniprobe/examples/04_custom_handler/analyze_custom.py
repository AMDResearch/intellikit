#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Omniprobe Example: A Bespoke Message Handler

Omniprobe's stock analyzers answer fixed questions. When you need to count
something they do not report, write a message handler: a small C++ class that
Omniprobe calls for every message the instrumented kernel sends. This example
builds access_counter.cc, which counts memory-instruction executions per kernel
by address space and kind, the lanes that took part, and executions by
partially active waves, then checks the counts against values known in advance.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 1000 threads in blocks of 256: 16 waves, the last one with 40 active lanes.
# Each active thread does one global load and one global store.
COUNT_KERNEL = """
#include <hip/hip_runtime.h>
__global__ void copy_n(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}
int main() {
    float *a, *b;
    if (hipMalloc(&a, 4096 * sizeof(float)) != hipSuccess) return 1;
    if (hipMalloc(&b, 4096 * sizeof(float)) != hipSuccess) return 1;
    hipLaunchKernelGGL(copy_n, dim3(4), dim3(256), 0, 0, a, b, 1000);
    return hipDeviceSynchronize() == hipSuccess ? 0 : 1;
}
"""

EXPECTED_GLOBAL = {
    "load_executions": 16,
    "store_executions": 16,
    "lanes": 2000,
    "partial_wave_executions": 2,
}


def build(tmp: Path):
    """Build the handler and the instrumented application; return their paths."""
    from omniprobe_ik import handler_build_flags, instrumentation_flags

    handler = tmp / "libaccess_counter.so"
    flags = handler_build_flags()
    cmd = ["hipcc", "-std=c++20", "-fPIC", "-shared", "-O2", *flags["cflags"]]
    cmd += [str(HERE / "access_counter.cc"), "-o", str(handler), *flags["ldflags"]]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    app = tmp / "copy_n"
    (tmp / "copy_n.hip").write_text(COUNT_KERNEL)
    cmd = ["hipcc", "-O2", "-g", *instrumentation_flags("memory"), str(tmp / "copy_n.hip")]
    subprocess.run([*cmd, "-o", str(app)], check=True, capture_output=True, text=True)
    return handler, app


def counts_from(log: str) -> list[dict]:
    """The handler prints one 'ACCESS_COUNTER {json}' line per kernel and address space."""
    prefix = "ACCESS_COUNTER "
    return [json.loads(line[len(prefix) :]) for line in log.splitlines() if line.startswith(prefix)]


def main():
    from omniprobe_ik import run_analyzer

    with tempfile.TemporaryDirectory(prefix="omniprobe_example_") as tmp_dir:
        handler, app = build(Path(tmp_dir))
        run = run_analyzer(str(handler), str(app))
        # The omniprobe launcher forwards what the application and handlers print on
        # its stderr, so read both streams (run.log).
        rows = counts_from(run.log)

    for row in rows:
        print(json.dumps(row))
    glob = [r for r in rows if r["kernel"].startswith("copy_n") and r["space"] == "global"]
    if not glob:
        print("No global counts reported.")
        return 1
    got = {k: glob[0][k] for k in EXPECTED_GLOBAL}
    print(f"\nglobal counts: {got}\nexpected:      {EXPECTED_GLOBAL}")
    print("MATCH" if got == EXPECTED_GLOBAL else "MISMATCH")
    return 0 if got == EXPECTED_GLOBAL else 1


if __name__ == "__main__":
    sys.exit(main())
