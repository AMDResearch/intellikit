# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the natural language → ProbeSpec planner."""

import pytest

from proboscis.planner import ProbeSpec, plan_probe, list_probe_types


class TestPlanProbe:
    """Test natural language probe planning."""

    def test_memory_trace_keywords(self):
        for phrase in [
            "find memory accesses",
            "trace loads and stores",
            "show me the memory access pattern",
            "what global_load instructions are there",
            "buffer_store analysis",
        ]:
            spec = plan_probe(phrase)
            assert spec.probe_type == "memory_trace", f"Failed for: {phrase!r}"

    def test_block_count_keywords(self):
        for phrase in [
            "count basic block executions",
            "how many times is each block executed",
            "find hotspots",
            "block frequency profiling",
            "execution count per block",
        ]:
            spec = plan_probe(phrase)
            assert spec.probe_type == "block_count", f"Failed for: {phrase!r}"

    def test_register_snapshot_keywords(self):
        for phrase in [
            "check register pressure",
            "how many VGPRs are used",
            "SGPR allocation",
            "what's the occupancy",
            "is it spilling registers",
        ]:
            spec = plan_probe(phrase)
            assert spec.probe_type == "register_snapshot", f"Failed for: {phrase!r}"

    def test_default_to_memory_trace(self):
        spec = plan_probe("something completely unrelated")
        assert spec.probe_type == "memory_trace"

    def test_target_kernel_passthrough(self):
        spec = plan_probe("memory accesses", target_kernel="vec_add")
        assert spec.target_kernel == "vec_add"

    def test_sample_rate_passthrough(self):
        spec = plan_probe("memory accesses", sample_rate=10)
        assert spec.sample_rate == 10

    def test_max_records_passthrough(self):
        spec = plan_probe("memory accesses", max_records=5000)
        assert spec.max_records == 5000

    def test_to_dict_roundtrip(self):
        spec = ProbeSpec(
            probe_type="block_count",
            target_kernel="my_kernel",
            sample_rate=5,
            max_records=1000,
            filters={"min_count": 10},
        )
        d = spec.to_dict()
        restored = ProbeSpec.from_dict(d)
        assert restored.probe_type == spec.probe_type
        assert restored.target_kernel == spec.target_kernel
        assert restored.sample_rate == spec.sample_rate
        assert restored.max_records == spec.max_records
        assert restored.filters == spec.filters


class TestListProbeTypes:
    def test_returns_all_types(self):
        types = list_probe_types()
        assert "memory_trace" in types
        assert "block_count" in types
        assert "register_snapshot" in types

    def test_each_type_has_description(self):
        types = list_probe_types()
        for name, info in types.items():
            assert "name" in info
            assert "description" in info
            assert "result_schema" in info
