#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Omniprobe Example: A Message Handler Written in Python

Builds a kernel whose answer is known in advance, runs line_counter.py on it
through Omniprobe's PythonHandler, and checks the counts. copy_n runs 1000
threads in blocks of 256: 16 waves, the last with 40 active lanes. Line 5,
`out[i] = in[i]`, has one global read and one global write: 16 wave executions,
1000 lane accesses, 1 partial wave and 32 distinct 128-byte lines (4000 bytes)
each. Every wave's 64 lanes also load a kernel argument (constant space).
"""

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "04_custom_handler"))

EXPECTED = {
    ("read", "global"): (16, 1000, 1, 32),
    ("write", "global"): (16, 1000, 1, 32),
}
FIELDS = ("wave_executions", "lane_accesses", "partial_wave_executions", "distinct_128B_lines")


def counts_from(stdout: str) -> list[dict]:
    """The handler prints one 'LINE_COUNTER {json}' line per site and dispatch."""
    prefix = "LINE_COUNTER "
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
        run = run_analyzer(str(HERE / "line_counter.py"), str(tmp / "copy_n"))
        rows = counts_from(run.stdout)

    for row in rows:
        print(json.dumps(row))
    got = {
        (r["access_type"], r["memory_space"]): tuple(r[f] for f in FIELDS)
        for r in rows
        if r["location"].split(":")[0].endswith("copy_n.hip") and r["location"].split(":")[1] == "5"
    }
    ok = got == EXPECTED
    print("PASS" if ok else f"FAIL: expected {EXPECTED} for copy_n.hip:5, got {got}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
