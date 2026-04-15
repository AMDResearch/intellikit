# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
AMDGPU assembly code generator for probe instrumentation.

Generates small assembly snippets that get injected into kernel code objects.
Each probe type maps to a template that writes structured data into the
probe buffer (passed via the hidden_proboscis_ctx pointer).

The probe buffer layout:
  [0:8]   - write_offset (atomic counter, incremented per record)
  [8:16]  - max_records
  [16:24] - record_size
  [24:..] - records array

Each record has a type-specific layout defined by the probe type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .planner import ProbeSpec


@dataclass
class ProbeBufferLayout:
    """Describes the structure of the probe result buffer."""
    header_size: int = 24  # write_offset(8) + max_records(8) + record_size(8)
    record_size: int = 0
    fields: list = None

    def __post_init__(self):
        if self.fields is None:
            self.fields = []

    def total_size(self, max_records: int) -> int:
        return self.header_size + self.record_size * max_records


@dataclass
class InstrumentationSnippet:
    """A generated assembly snippet for probe instrumentation."""
    probe_type: str
    assembly: str
    buffer_layout: ProbeBufferLayout
    vgpr_count: int  # additional VGPRs needed
    sgpr_count: int  # additional SGPRs needed
    insertion_point: str  # "entry", "before_mem", "block_start", "exit"


# ─── Probe Templates ─────────────────────────────────────────────────────────

def generate_memory_trace_snippet(
    ctx_sgpr: int = 0,
    scratch_vgpr_base: int = 0,
) -> InstrumentationSnippet:
    """
    Generate assembly for memory access tracing.

    Records: { address: u64, size: u32, op_type: u32, thread_id: u32, pad: u32 }
    Record size: 24 bytes

    The snippet atomically increments the write_offset in the probe buffer header,
    then writes the record at the computed offset.
    """
    layout = ProbeBufferLayout(
        record_size=24,
        fields=[
            {"name": "address", "type": "uint64", "offset": 0},
            {"name": "size", "type": "uint32", "offset": 8},
            {"name": "op_type", "type": "uint32", "offset": 12},
            {"name": "thread_id", "type": "uint32", "offset": 16},
            {"name": "pad", "type": "uint32", "offset": 20},
        ],
    )

    # Assembly template using probe context pointer from hidden arg
    # s[ctx_sgpr:ctx_sgpr+1] holds the probe buffer base address
    # v[scratch_vgpr_base+0..5] are scratch VGPRs
    s0, s1 = ctx_sgpr, ctx_sgpr + 1
    v0 = scratch_vgpr_base
    v1 = scratch_vgpr_base + 1
    v2 = scratch_vgpr_base + 2
    v3 = scratch_vgpr_base + 3

    asm = f"""\
    // === Proboscis: Memory Access Trace Probe ===
    // Atomically claim a record slot
    v_mov_b32       v{v0}, 1
    global_atomic_add_u64 v[{v0}:{v1}], v[{v0}:{v1}], s[{s0}:{s1}] offset:0 glc
    s_waitcnt       vmcnt(0)

    // Check if within bounds: compare slot index vs max_records
    // v[{v0}:{v1}] now holds the old write_offset (our slot index)
    s_load_dwordx2  s[{s0+2}:{s0+3}], s[{s0}:{s1}], 0x8  // max_records
    s_waitcnt       lgkmcnt(0)
    v_cmp_lt_u64    vcc, v[{v0}:{v1}], s[{s0+2}:{s0+3}]
    s_cbranch_vccz  proboscis_skip_{ctx_sgpr}

    // Compute record address: base + header_size + slot * record_size
    v_mul_lo_u32    v{v2}, v{v0}, {layout.record_size}
    v_add_co_u32    v{v2}, vcc, v{v2}, {layout.header_size}
    v_mov_b32       v{v3}, s{s0}
    v_add_co_u32    v{v2}, vcc, v{v2}, v{v3}
    v_mov_b32       v{v3}, s{s1}
    v_addc_co_u32   v{v3}, vcc, v{v3}, 0, vcc

    // Write record (address, size, op_type, thread_id)
    // The actual memory address being accessed should be in a register
    // that the instrumentation point provides — this is a placeholder
    global_store_dwordx2 v[{v2}:{v3}], v[{v0}:{v1}], off offset:0
    v_mov_b32       v{v0}, 0  // placeholder for size/op fields
    global_store_dword v[{v2}:{v3}], v{v0}, off offset:16

proboscis_skip_{ctx_sgpr}:
    // === End Proboscis Probe ===
"""

    return InstrumentationSnippet(
        probe_type="memory_trace",
        assembly=asm,
        buffer_layout=layout,
        vgpr_count=4,
        sgpr_count=4,
        insertion_point="before_mem",
    )


def generate_block_count_snippet(
    block_id: int,
    ctx_sgpr: int = 0,
    scratch_vgpr_base: int = 0,
) -> InstrumentationSnippet:
    """
    Generate assembly for basic block execution counting.

    Each block gets a u64 counter at a fixed offset in the probe buffer.
    Record layout: { block_id: u32, pad: u32, count: u64 } = 16 bytes
    """
    layout = ProbeBufferLayout(
        record_size=16,
        fields=[
            {"name": "block_id", "type": "uint32", "offset": 0},
            {"name": "pad", "type": "uint32", "offset": 4},
            {"name": "count", "type": "uint64", "offset": 8},
        ],
    )

    s0, s1 = ctx_sgpr, ctx_sgpr + 1
    v0 = scratch_vgpr_base

    # Fixed offset for this block's counter
    counter_offset = layout.header_size + block_id * layout.record_size + 8  # +8 for count field

    asm = f"""\
    // === Proboscis: Block Count Probe (block {block_id}) ===
    v_mov_b32       v{v0}, 1
    global_atomic_add_u64 v{v0}, v{v0}, s[{s0}:{s1}] offset:{counter_offset}
    // === End Proboscis Probe ===
"""

    return InstrumentationSnippet(
        probe_type="block_count",
        assembly=asm,
        buffer_layout=layout,
        vgpr_count=1,
        sgpr_count=2,
        insertion_point="block_start",
    )


def generate_register_snapshot_snippet(
    ctx_sgpr: int = 0,
    scratch_vgpr_base: int = 0,
) -> InstrumentationSnippet:
    """
    Generate assembly for register pressure snapshot.

    Records static register allocation info (from kernel descriptor,
    not dynamic). The C++ runtime fills this from the code object metadata
    rather than from inline assembly — this snippet is a no-op marker.
    """
    layout = ProbeBufferLayout(
        record_size=16,
        fields=[
            {"name": "vgpr_count", "type": "uint32", "offset": 0},
            {"name": "sgpr_count", "type": "uint32", "offset": 4},
            {"name": "occupancy_pct", "type": "uint32", "offset": 8},
            {"name": "spill_bytes", "type": "uint32", "offset": 12},
        ],
    )

    asm = """\
    // === Proboscis: Register Snapshot ===
    // (filled by C++ runtime from code object metadata)
    // === End Proboscis Probe ===
"""

    return InstrumentationSnippet(
        probe_type="register_snapshot",
        assembly=asm,
        buffer_layout=layout,
        vgpr_count=0,
        sgpr_count=0,
        insertion_point="entry",
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_probe(spec: ProbeSpec) -> InstrumentationSnippet:
    """Generate an instrumentation snippet for the given probe spec."""
    generators = {
        "memory_trace": generate_memory_trace_snippet,
        "block_count": lambda: generate_block_count_snippet(block_id=0),
        "register_snapshot": generate_register_snapshot_snippet,
    }

    gen = generators.get(spec.probe_type)
    if gen is None:
        raise ValueError(f"Unknown probe type: {spec.probe_type}")
    return gen()
