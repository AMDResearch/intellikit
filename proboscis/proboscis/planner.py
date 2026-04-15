# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Translates natural language probe descriptions into structured ProbeSpecs.

Simple keyword matching — the MCP caller is already an LLM, so we don't
need another LLM call here. Just map intent to probe type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ProbeSpec:
    """Structured specification for a kernel instrumentation probe."""

    probe_type: str  # "memory_trace", "block_count", "register_snapshot"
    target_kernel: Optional[str] = None  # kernel name pattern, None = all
    filters: Dict[str, Any] = field(default_factory=dict)
    sample_rate: int = 1  # 1 = every invocation
    max_records: int = 10000

    def to_dict(self) -> dict:
        return {
            "probe_type": self.probe_type,
            "target_kernel": self.target_kernel,
            "filters": self.filters,
            "sample_rate": self.sample_rate,
            "max_records": self.max_records,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProbeSpec:
        return cls(
            probe_type=d["probe_type"],
            target_kernel=d.get("target_kernel"),
            filters=d.get("filters", {}),
            sample_rate=d.get("sample_rate", 1),
            max_records=d.get("max_records", 10000),
        )


# Probe type detection patterns (order matters — first match wins)
# Register/occupancy checked BEFORE block_count because "how many VGPRs"
# should match register_snapshot, not block_count's "how many" pattern.
_PROBE_PATTERNS = [
    # Memory access tracing
    (
        re.compile(
            r"memory|load|store|access|read.*write|write.*read|"
            r"flat_load|flat_store|global_load|global_store|"
            r"buffer_load|buffer_store|scratch|lds",
            re.IGNORECASE,
        ),
        "memory_trace",
    ),
    # Register pressure / occupancy (before block_count to avoid "how many" clash)
    (
        re.compile(
            r"register|vgpr|sgpr|occupancy|spill|pressure|"
            r"wave|wavefront",
            re.IGNORECASE,
        ),
        "register_snapshot",
    ),
    # Block frequency / hotspot profiling
    (
        re.compile(
            r"how many|count|frequency|hotspot|hot spot|"
            r"basic.?block|block.?count|coverage|profile|"
            r"execution.?count|call.?count|invoc",
            re.IGNORECASE,
        ),
        "block_count",
    ),
]

# Descriptions for each probe type
PROBE_DESCRIPTIONS = {
    "memory_trace": {
        "name": "Memory Access Trace",
        "description": (
            "Records memory load and store operations executed by the kernel. "
            "Captures the address, size, and type (load/store/atomic) of each access."
        ),
        "result_schema": {
            "address": "uint64 — virtual address accessed",
            "size": "uint32 — access size in bytes",
            "op": "string — 'load', 'store', or 'atomic'",
            "thread_id": "uint32 — flat thread index within the workgroup",
        },
    },
    "block_count": {
        "name": "Basic Block Frequency",
        "description": (
            "Counts how many times each basic block in the kernel is executed. "
            "Useful for identifying hotspots and dead code."
        ),
        "result_schema": {
            "block_id": "uint32 — basic block index",
            "count": "uint64 — execution count across all threads",
            "address": "uint64 — start address of the basic block",
        },
    },
    "register_snapshot": {
        "name": "Register Pressure Snapshot",
        "description": (
            "Captures VGPR and SGPR usage at instrumentation points. "
            "Helps identify register spilling and occupancy limiters."
        ),
        "result_schema": {
            "vgpr_count": "uint32 — VGPRs allocated",
            "sgpr_count": "uint32 — SGPRs allocated",
            "occupancy": "float — theoretical occupancy (0.0-1.0)",
            "spill_bytes": "uint32 — scratch memory used for spills",
        },
    },
}


def plan_probe(
    description: str,
    target_kernel: Optional[str] = None,
    sample_rate: int = 1,
    max_records: int = 10000,
) -> ProbeSpec:
    """
    Parse a natural language probe description into a ProbeSpec.

    Args:
        description: Natural language like "find memory accesses" or "count block executions"
        target_kernel: Optional kernel name pattern
        sample_rate: Sample every N-th dispatch
        max_records: Maximum records to collect

    Returns:
        ProbeSpec with the detected probe type
    """
    for pattern, probe_type in _PROBE_PATTERNS:
        if pattern.search(description):
            return ProbeSpec(
                probe_type=probe_type,
                target_kernel=target_kernel,
                sample_rate=sample_rate,
                max_records=max_records,
            )

    # Default to memory trace if nothing matches
    return ProbeSpec(
        probe_type="memory_trace",
        target_kernel=target_kernel,
        sample_rate=sample_rate,
        max_records=max_records,
    )


def list_probe_types() -> dict:
    """Return descriptions of all available probe types."""
    return PROBE_DESCRIPTIONS
