#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Omniprobe Example: A Message Handler Written in Python

Builds a kernel whose answer is known in advance, runs address_counter.py on it
through Omniprobe's PythonHandler, and checks the counts. 1000 threads in blocks
of 256 run as 16 waves; each active thread does one global load and one global
store (2000 lane accesses), and every wave's 64 lanes also load a kernel argument
(16 * 64 = 1024), so the handler sees 3024 lane accesses.
"""

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "04_custom_handler"))

EXPECTED_LANE_ACCESSES = 2000 + 16 * 64


def counts_from(stdout: str) -> list[dict]:
    """The handler prints one 'ADDRESS_COUNTER {json}' line per dispatch."""
    prefix = "ADDRESS_COUNTER "
    return [
        json.loads(line[len(prefix) :]) for line in stdout.splitlines() if line.startswith(prefix)
    ]


def main():
    from analyze_custom import COUNT_KERNEL

    from omniprobe_ik import instrumentation_flags, run_analyzer
    import subprocess

    with tempfile.TemporaryDirectory(prefix="omniprobe_example_") as tmp_dir:
        tmp = Path(tmp_dir)
        (tmp / "copy_n.hip").write_text(COUNT_KERNEL)
        cmd = ["hipcc", "-O2", "-g", *instrumentation_flags("memory"), str(tmp / "copy_n.hip")]
        subprocess.run([*cmd, "-o", str(tmp / "copy_n")], check=True, capture_output=True)
        # A .py path runs the module through PythonHandler; its report is on stdout.
        run = run_analyzer(str(HERE / "address_counter.py"), str(tmp / "copy_n"))
        rows = counts_from(run.stdout)

    for row in rows:
        print(json.dumps(row))
    ok = len(rows) == 1 and rows[0]["lane_accesses"] == EXPECTED_LANE_ACCESSES
    print("PASS" if ok else f"FAIL: expected {EXPECTED_LANE_ACCESSES} lane accesses")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
