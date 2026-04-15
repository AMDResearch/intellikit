# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the AMDGPU assembly code generator."""

import pytest

from proboscis.codegen import (
    generate_block_count_snippet,
    generate_memory_trace_snippet,
    generate_probe,
    generate_register_snapshot_snippet,
)
from proboscis.planner import ProbeSpec


class TestMemoryTraceSnippet:
    def test_generates_assembly(self):
        snippet = generate_memory_trace_snippet()
        assert "Proboscis" in snippet.assembly
        assert "global_atomic_add" in snippet.assembly
        assert snippet.probe_type == "memory_trace"
        assert snippet.buffer_layout.record_size == 24
        assert snippet.vgpr_count > 0

    def test_insertion_point(self):
        snippet = generate_memory_trace_snippet()
        assert snippet.insertion_point == "before_mem"


class TestBlockCountSnippet:
    def test_generates_assembly(self):
        snippet = generate_block_count_snippet(block_id=5)
        assert "block 5" in snippet.assembly
        assert "global_atomic_add" in snippet.assembly
        assert snippet.probe_type == "block_count"
        assert snippet.buffer_layout.record_size == 16

    def test_different_blocks(self):
        s0 = generate_block_count_snippet(block_id=0)
        s1 = generate_block_count_snippet(block_id=1)
        assert s0.assembly != s1.assembly  # different offsets


class TestRegisterSnapshotSnippet:
    def test_generates_placeholder(self):
        snippet = generate_register_snapshot_snippet()
        assert snippet.probe_type == "register_snapshot"
        assert snippet.vgpr_count == 0  # no inline asm needed
        assert snippet.insertion_point == "entry"


class TestGenerateProbe:
    def test_memory_trace(self):
        spec = ProbeSpec(probe_type="memory_trace")
        snippet = generate_probe(spec)
        assert snippet.probe_type == "memory_trace"

    def test_block_count(self):
        spec = ProbeSpec(probe_type="block_count")
        snippet = generate_probe(spec)
        assert snippet.probe_type == "block_count"

    def test_register_snapshot(self):
        spec = ProbeSpec(probe_type="register_snapshot")
        snippet = generate_probe(spec)
        assert snippet.probe_type == "register_snapshot"

    def test_unknown_type_raises(self):
        spec = ProbeSpec(probe_type="quantum_teleportation")
        with pytest.raises(ValueError, match="Unknown probe type"):
            generate_probe(spec)
