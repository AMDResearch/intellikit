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
    """Summarize memory trace records.

    Handles two record formats:
    - Per-access records (from probe buffer): {op, address, size, thread_id}
    - Dispatch-level records (from interceptor): {dispatch_id, grid_size, workgroup_size, threads}
    """
    if not records:
        return {"total_accesses": 0, "loads": 0, "stores": 0, "atomics": 0}

    # Dispatch-level records (from the interceptor, before probe buffers are wired)
    if "dispatch_id" in records[0]:
        total_threads = sum(r.get("threads", 0) for r in records)
        return {
            "dispatch_count": len(records),
            "total_threads": total_threads,
            "grid_sizes": [r.get("grid_size") for r in records],
            "workgroup_sizes": [r.get("workgroup_size") for r in records],
        }

    # Per-access records (from probe buffer)
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


def format_instruction_analysis(analysis: Dict[str, Any], dispatches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine static instruction analysis with runtime dispatch data.

    Produces per-thread instruction counts and estimated total operations
    based on grid dimensions.
    """
    if not analysis:
        return {}

    total_threads = sum(d.get("threads", 0) for d in dispatches)
    dispatch_count = len(dispatches)

    total_loads = analysis.get("total_loads", 0)
    total_stores = analysis.get("total_stores", 0)
    atomics = analysis.get("atomics", 0)
    loads_gt_4b = analysis.get("loads_gt_4B", 0)
    stores_gt_4b = analysis.get("stores_gt_4B", 0)

    result = {
        "per_thread": {
            "loads": total_loads,
            "stores": total_stores,
            "atomics": atomics,
            "loads_gt_4B": loads_gt_4b,
            "stores_gt_4B": stores_gt_4b,
        },
        "by_type": {
            "global_loads": analysis.get("global_loads", 0),
            "global_stores": analysis.get("global_stores", 0),
            "flat_loads": analysis.get("flat_loads", 0),
            "flat_stores": analysis.get("flat_stores", 0),
            "buffer_loads": analysis.get("buffer_loads", 0),
            "buffer_stores": analysis.get("buffer_stores", 0),
            "ds_reads": analysis.get("ds_reads", 0),
            "ds_writes": analysis.get("ds_writes", 0),
        },
        "loads_by_size": analysis.get("loads_by_size", {}),
        "stores_by_size": analysis.get("stores_by_size", {}),
    }

    if total_threads > 0:
        result["estimated_totals"] = {
            "total_load_ops": total_loads * total_threads,
            "total_store_ops": total_stores * total_threads,
            "total_atomic_ops": atomics * total_threads,
            "total_load_ops_gt_4B": loads_gt_4b * total_threads,
            "total_threads": total_threads,
            "dispatch_count": dispatch_count,
        }

    return result


FORMATTERS = {
    "memory_trace": format_memory_trace,
    "block_count": format_block_count,
    "register_snapshot": format_register_snapshot,
}


def format_results(probe_type: str, records: List[Dict[str, Any]],
                   instruction_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Format probe results with a type-appropriate summary."""
    formatter = FORMATTERS.get(probe_type)
    result = {}
    if formatter:
        result = {"records": records, "summary": formatter(records)}
    else:
        result = {"records": records, "summary": {}}

    if instruction_analysis:
        result["instruction_analysis"] = format_instruction_analysis(
            instruction_analysis, records
        )

    return result
