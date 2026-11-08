# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""A Python message handler for Omniprobe's PythonHandler (Omniprobe 13ca707 or later).

Omniprobe imports this module once per process and calls handleAddressMessage once per
active lane of every instrumented memory instruction, with only ``address``, ``kernel``
and ``dispatch_id``. There is no source line, read/write flag, address space or wave
id: kernel-argument loads (every lane of every wave) are counted along with the
application's own loads and stores. For per-line or per-wave questions write a C++
handler (example 04).

At the end of each dispatch it prints one line:
    ADDRESS_COUNTER {"kernel": ..., "dispatch_id": ..., "lane_accesses": ...,
                     "distinct_addresses": ..., "distinct_128B_lines": ...}
"""

import json
from collections import defaultdict

_addresses = defaultdict(list)  # dispatch_id -> addresses, in arrival order


def handleAddressMessage(msg):
    _addresses[msg["dispatch_id"]].append(msg["address"])


def handleTimeMessage(msg):
    pass


def handleReport(msg):
    addresses = _addresses.pop(msg["dispatch_id"], [])
    row = {
        "kernel": msg["kernel"],
        "dispatch_id": msg["dispatch_id"],
        "lane_accesses": len(addresses),
        "distinct_addresses": len(set(addresses)),
        "distinct_128B_lines": len({a // 128 for a in addresses}),
    }
    print("ADDRESS_COUNTER " + json.dumps(row), flush=True)
