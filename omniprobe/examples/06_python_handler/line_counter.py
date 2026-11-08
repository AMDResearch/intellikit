# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""A Python message handler for Omniprobe's PythonHandler (Omniprobe b9492cd or later).

Omniprobe imports this module once per process. handleAddressWave(msg) is called once
per wave-level memory instruction with the wave's ``addresses`` (one per active lane),
``exec``, ``lanes``, ``access_type`` ("read"/"write"/...), ``memory_space`` ("global",
"shared", "constant", ...), ``access_size`` (bytes), ``source_file``/``line``/``column``,
``block_idx``, ``wave_num``, ``cu_id``, ``se_id``, ``xcc_id``, ``timestamp``, ``kernel``
and ``dispatch_id``.

At the end of each dispatch it prints one line per source location, access type and
address space:
    LINE_COUNTER {"kernel": ..., "dispatch_id": ..., "location": "file:line:column",
                  "access_type": ..., "memory_space": ..., "wave_executions": ...,
                  "lane_accesses": ..., "partial_wave_executions": ..., "distinct_128B_lines": ...}
Prints need flush=True: unflushed output is lost when the run ends.
"""

import json
from collections import defaultdict

# dispatch_id -> (file, line, column, access_type, memory_space) -> counters
_sites = defaultdict(dict)


def handleAddressWave(msg):
    key = (msg["source_file"], msg["line"], msg["column"], msg["access_type"], msg["memory_space"])
    site = _sites[msg["dispatch_id"]].setdefault(
        key, {"waves": 0, "lanes": 0, "partial": 0, "lines": set()}
    )
    site["waves"] += 1
    site["lanes"] += len(msg["addresses"])
    site["partial"] += len(msg["addresses"]) < 64
    site["lines"].update(a // 128 for a in msg["addresses"])


def handleReport(msg):
    for (path, line, column, access_type, space), site in sorted(
        _sites.pop(msg["dispatch_id"], {}).items()
    ):
        row = {
            "kernel": msg["kernel"],
            "dispatch_id": msg["dispatch_id"],
            "location": f"{path}:{line}:{column}",
            "access_type": access_type,
            "memory_space": space,
            "wave_executions": site["waves"],
            "lane_accesses": site["lanes"],
            "partial_wave_executions": site["partial"],
            "distinct_128B_lines": len(site["lines"]),
        }
        print("LINE_COUNTER " + json.dumps(row), flush=True)
