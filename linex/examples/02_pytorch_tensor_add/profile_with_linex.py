#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Linex driver for Example 02: profile ``tensor_add.py`` and print hotspots or ISA.

Same idea as running the numbered SQTT example end-to-end: one script drives the tool.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    script = here / "tensor_add.py"
    if not script.is_file():
        print(f"Missing {script}", file=sys.stderr)
        return 1

    cmd = f"{sys.executable} {script}"

    print("=" * 60)
    print("Linex example: PyTorch tensor add")
    print("=" * 60)
    print("Step 1: Running workload under Linex (rocprofv3 + SQTT)...")
    print(f"Command: {cmd}\n")

    from linex import Linex

    profiler = Linex()
    profiler.profile(cmd, kernel_filter=None)

    print("Step 2: Reporting results...\n")
    if profiler.source_lines:
        print("Top source-line hotspots:\n")
        for i, line in enumerate(profiler.source_lines[:15], 1):
            print(
                f"{i:2}. {line.source_location}  "
                f"cycles={line.total_cycles:,}  stall={line.stall_percent:.1f}%  "
                f"insts={len(line.instructions)}"
            )
        print("\nDone.")
        return 0

    print("No aggregated source lines; sample ISA instructions:\n")
    for i, inst in enumerate(profiler.instructions[:15], 1):
        isa = inst.isa[:100] + ("..." if len(inst.isa) > 100 else "")
        print(f"{i:2}. {isa}  latency_cycles={inst.latency_cycles:,}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)
