# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the Omniprobe IntelliKit API (mocked upstream)."""

from unittest.mock import MagicMock, patch

import pytest

from omniprobe_ik.api import (
    BasicBlock,
    BasicBlockResult,
    KernelBBResult,
    KernelMemoryResult,
    MemoryAccess,
    MemoryAnalysisResult,
    Omniprobe,
)


# ---------------------------------------------------------------------------
# Fixtures: fake upstream objects
# ---------------------------------------------------------------------------


def _make_upstream_memory_result():
    """Build a fake upstream MemoryAnalysisResult."""
    access = MagicMock()
    access.source_file = "test.hip"
    access.line = 42
    access.column = 8
    access.excess_cache_lines = 3
    access.bank_conflicts = 1
    access.memory_space = "global"
    access.access_type = "load"
    access.execution_count = 100

    kernel = MagicMock()
    kernel.kernel_name = "matmul_kernel"
    kernel.dispatch_id = 0
    kernel.accesses = [access]

    result = MagicMock()
    result.kernels = [kernel]
    return result


def _make_upstream_bb_result():
    """Build a fake upstream BasicBlockResult."""
    bb = MagicMock()
    bb.bb_id = 1
    bb.source_file = "test.hip"
    bb.line = 10
    bb.min_cycles = 100.0
    bb.max_cycles = 500.0
    bb.p25 = 150.0
    bb.p50 = 200.0
    bb.p75 = 300.0
    bb.p95 = 450.0
    bb.p99 = 490.0
    bb.wave_count = 64

    kernel = MagicMock()
    kernel.kernel_name = "matmul_kernel"
    kernel.dispatch_id = 0
    kernel.basic_blocks = [bb]

    result = MagicMock()
    result.kernels = [kernel]
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOmniprobeInit:
    """Test Omniprobe initialization."""

    def test_import_error_when_upstream_missing(self):
        with patch.dict("sys.modules", {"omniprobe": None, "omniprobe.api": None}):
            with pytest.raises(ImportError, match="omniprobe.api"):
                Omniprobe()


class TestAnalyzeMemory:
    """Test analyze_memory conversion."""

    @patch("omniprobe_ik.api.Omniprobe.__init__", return_value=None)
    def test_converts_upstream_result(self, mock_init):
        probe = Omniprobe.__new__(Omniprobe)
        probe._upstream = MagicMock()
        probe._upstream.analyze_memory.return_value = _make_upstream_memory_result()

        result = probe.analyze_memory("./app")

        assert isinstance(result, MemoryAnalysisResult)
        assert len(result.kernels) == 1

        kernel = result.kernels[0]
        assert isinstance(kernel, KernelMemoryResult)
        assert kernel.kernel_name == "matmul_kernel"
        assert kernel.dispatch_id == 0

        access = kernel.accesses[0]
        assert isinstance(access, MemoryAccess)
        assert access.source_file == "test.hip"
        assert access.line == 42
        assert access.bank_conflicts == 1
        assert access.excess_cache_lines == 3
        assert access.memory_space == "global"


class TestAnalyzeBasicBlocks:
    """Test analyze_basic_blocks conversion."""

    @patch("omniprobe_ik.api.Omniprobe.__init__", return_value=None)
    def test_converts_upstream_result(self, mock_init):
        probe = Omniprobe.__new__(Omniprobe)
        probe._upstream = MagicMock()
        probe._upstream.analyze_basic_blocks.return_value = _make_upstream_bb_result()

        result = probe.analyze_basic_blocks("./app")

        assert isinstance(result, BasicBlockResult)
        assert len(result.kernels) == 1

        kernel = result.kernels[0]
        assert isinstance(kernel, KernelBBResult)
        assert kernel.kernel_name == "matmul_kernel"

        bb = kernel.basic_blocks[0]
        assert isinstance(bb, BasicBlock)
        assert bb.bb_id == 1
        assert bb.min_cycles == 100.0
        assert bb.max_cycles == 500.0
        assert bb.p50 == 200.0
        assert bb.wave_count == 64


class TestDataclassDefaults:
    """Test that result dataclasses have correct defaults."""

    def test_memory_access_defaults(self):
        access = MemoryAccess(
            source_file="a.hip", line=1, column=0, excess_cache_lines=0, bank_conflicts=0
        )
        assert access.memory_space == ""
        assert access.access_type == ""
        assert access.execution_count == 0

    def test_basic_block_defaults(self):
        bb = BasicBlock(bb_id=0, source_file="a.hip", line=1, min_cycles=0, max_cycles=0)
        assert bb.p25 == 0.0
        assert bb.p50 == 0.0
        assert bb.p75 == 0.0
        assert bb.p95 == 0.0
        assert bb.p99 == 0.0
        assert bb.wave_count == 0

    def test_empty_results(self):
        mem = MemoryAnalysisResult()
        assert mem.kernels == []

        bb = BasicBlockResult()
        assert bb.kernels == []
