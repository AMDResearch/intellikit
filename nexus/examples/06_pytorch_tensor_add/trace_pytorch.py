#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Nexus driver for Example 06: trace ``tensor_add.py`` and print kernels + assembly sample.

Structured like ``01_trace_kernel/trace.py``: banner, numbered steps, summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

from nexus import Nexus


def main() -> int:
    here = Path(__file__).resolve().parent
    script = here / "tensor_add.py"
    if not script.is_file():
        print(f"Missing {script}", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(script)]

    print("=" * 60)
    print("Nexus example: PyTorch tensor add")
    print("=" * 60)
    print(f"Step 1: Child command: {' '.join(cmd)}\n")

    nexus = Nexus(log_level=1)
    print("Step 2: Tracing with Nexus...")
    try:
        trace = nexus.run(cmd)
    except Exception as e:
        print(f"Tracing failed: {e}", file=sys.stderr)
        return 1

    print(f"  Captured {len(trace)} kernel(s)\n")

    print("Step 3: Kernel trace summary")
    print("=" * 60)
    for kernel in trace:
        print(f"\nKernel: {kernel.name}")
        if kernel.signature:
            print(f"  Signature: {kernel.signature}")
        asm = kernel.assembly
        print(f"  Assembly lines: {len(asm)}")
        hip = kernel.hip
        if hip:
            print(f"  HIP source lines: {len(hip)}")
        if asm:
            for i, line in enumerate(asm[:10], 1):
                print(f"    {i:2d}. {line}")
            if len(asm) > 10:
                print(f"    ... ({len(asm) - 10} more)")
        print()

    print("=" * 60)
    print("Tracing completed successfully")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
