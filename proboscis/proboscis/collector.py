# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Reads and formats probe results from the shared probe buffer.

The C++ runtime writes results to a JSON file specified by PROBOSCIS_RESULTS.
This module parses those results into structured Python objects.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_results(results_path: str) -> Dict[str, Any]:
    """Load probe results from the JSON file written by the C++ runtime."""
    path = Path(results_path)
    if not path.exists():
        return {}

    content = path.read_text().strip()
    if not content:
        return {}

    return json.loads(content)


def format_memory_trace(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize memory trace records."""
    if not records:
        return {"total_accesses": 0, "loads": 0, "stores": 0, "atomics": 0}

    loads = sum(1 for r in records if r.get("op") == "load")
    stores = sum(1 for r in records if r.get("op") == "store")
    atomics = sum(1 for r in records if r.get("op") == "atomic")

    addresses = [r.get("address", 0) for r in records]
    sizes = [r.get("size", 0) for r in records]

    return {
        "total_accesses": len(records),
        "loads": loads,
        "stores": stores,
        "atomics": atomics,
        "min_address": hex(min(addresses)) if addresses else "0x0",
        "max_address": hex(max(addresses)) if addresses else "0x0",
        "address_range_bytes": max(addresses) - min(addresses) if addresses else 0,
        "unique_sizes": sorted(set(sizes)),
    }


def format_block_count(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize block count records."""
    if not records:
        return {"total_blocks": 0, "hottest_block": None}

    sorted_blocks = sorted(records, key=lambda r: r.get("count", 0), reverse=True)
    total_executions = sum(r.get("count", 0) for r in records)

    return {
        "total_blocks": len(records),
        "total_executions": total_executions,
        "hottest_block": sorted_blocks[0] if sorted_blocks else None,
        "coldest_block": sorted_blocks[-1] if sorted_blocks else None,
        "dead_blocks": [r for r in records if r.get("count", 0) == 0],
    }


def format_register_snapshot(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize register snapshot records."""
    if not records:
        return {}

    return {
        "vgpr_count": records[0].get("vgpr_count", 0),
        "sgpr_count": records[0].get("sgpr_count", 0),
        "occupancy": records[0].get("occupancy", 0.0),
        "spill_bytes": records[0].get("spill_bytes", 0),
        "is_spilling": records[0].get("spill_bytes", 0) > 0,
    }


FORMATTERS = {
    "memory_trace": format_memory_trace,
    "block_count": format_block_count,
    "register_snapshot": format_register_snapshot,
}


def format_results(probe_type: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format probe results with a type-appropriate summary."""
    formatter = FORMATTERS.get(probe_type)
    if formatter:
        return {"records": records, "summary": formatter(records)}
    return {"records": records, "summary": {}}
