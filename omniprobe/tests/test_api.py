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
    NoInstrumentedKernelsError,
    Omniprobe,
    instrumentation_flags,
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
    bb.column = 3
    bb.wave_visits = 64
    bb.active_lanes = 4000

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
        with (
            patch.dict("sys.modules", {"omniprobe": None, "omniprobe.api": None}),
            patch("omniprobe_ik.api._add_upstream_source_to_path", return_value=False),
        ):
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
        assert bb.column == 3
        assert bb.wave_visits == 64
        assert bb.active_lanes == 4000


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
        bb = BasicBlock(bb_id=0, source_file="a.hip", line=1)
        assert bb.column == 0
        assert bb.wave_visits == 0
        assert bb.active_lanes == 0

    def test_empty_results(self):
        mem = MemoryAnalysisResult()
        assert mem.kernels == []

        bb = BasicBlockResult()
        assert bb.kernels == []


def _probe_returning(memory=None, basic_blocks=None):
    """An Omniprobe wrapper whose upstream returns the given results."""
    probe = Omniprobe.__new__(Omniprobe)
    probe._upstream = MagicMock()
    probe._upstream.analyze_memory.return_value = memory
    probe._upstream.analyze_basic_blocks.return_value = basic_blocks
    return probe


class TestNoInstrumentedKernels:
    """An empty upstream result is an error that explains the plugin requirement."""

    def test_empty_memory_result_raises(self, monkeypatch):
        monkeypatch.setenv("OMNIPROBE_PLUGIN_DIR", "/nonexistent")
        probe = _probe_returning(memory=MagicMock(kernels=[]))
        with pytest.raises(NoInstrumentedKernelsError) as excinfo:
            probe.analyze_memory("./app")
        message = str(excinfo.value)
        assert "-fpass-plugin" in message
        assert "libAMDGCNSubmitAddressMessages-rocm.so" in message
        assert "kernel_filter" not in message

    def test_empty_bb_result_names_bb_plugin_and_filter(self, monkeypatch):
        monkeypatch.setenv("OMNIPROBE_PLUGIN_DIR", "/nonexistent")
        probe = _probe_returning(basic_blocks=MagicMock(kernels=[]))
        with pytest.raises(NoInstrumentedKernelsError) as excinfo:
            probe.analyze_basic_blocks("./app", kernel_filter="foo.*")
        message = str(excinfo.value)
        assert "libAMDGCNSubmitBBStart-rocm.so" in message
        assert "'foo.*'" in message

    def test_is_a_runtime_error(self):
        assert issubclass(NoInstrumentedKernelsError, RuntimeError)


class TestInstrumentationFlags:
    """Locating the plugin and building the hipcc flags."""

    @staticmethod
    def _fake_tree(tmp_path):
        """A fake Omniprobe tree: bin/omniprobe plus lib/plugins/*.so."""
        (tmp_path / "bin").mkdir()
        exe = tmp_path / "bin" / "omniprobe"
        exe.write_text("#!/bin/sh\n")
        plugins = tmp_path / "lib" / "plugins"
        plugins.mkdir(parents=True)
        for name in ("libAMDGCNSubmitAddressMessages-rocm.so", "libAMDGCNSubmitBBStart-rocm.so"):
            (plugins / name).write_bytes(b"")
        return exe, plugins

    def test_memory_flags(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OMNIPROBE_PLUGIN_DIR", raising=False)
        exe, plugins = self._fake_tree(tmp_path)
        flags = instrumentation_flags("memory", offload_arch="gfx90a", omniprobe_path=str(exe))
        assert flags == [
            f"-fpass-plugin={plugins / 'libAMDGCNSubmitAddressMessages-rocm.so'}",
            "--offload-arch=gfx90a",
        ]

    def test_rdc_is_opt_in(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OMNIPROBE_PLUGIN_DIR", raising=False)
        exe, plugins = self._fake_tree(tmp_path)
        flags = instrumentation_flags(
            "memory", offload_arch="gfx90a", omniprobe_path=str(exe), rdc=True
        )
        assert flags == [
            f"-fpass-plugin={plugins / 'libAMDGCNSubmitAddressMessages-rocm.so'}",
            "-fgpu-rdc",
            "--offload-arch=gfx90a",
        ]

    def test_basic_blocks_uses_bbstart_and_accepts_hyphen(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OMNIPROBE_PLUGIN_DIR", raising=False)
        exe, plugins = self._fake_tree(tmp_path)
        flags = instrumentation_flags(
            "basic-blocks", offload_arch="gfx942", omniprobe_path=str(exe)
        )
        assert flags[0] == f"-fpass-plugin={plugins / 'libAMDGCNSubmitBBStart-rocm.so'}"

    def test_unresolved_symlink_is_tried_first(self, tmp_path, monkeypatch):
        """A build tree's bin/omniprobe is a symlink into the source tree."""
        monkeypatch.delenv("OMNIPROBE_PLUGIN_DIR", raising=False)
        build = tmp_path / "build"
        build.mkdir()
        _, plugins = self._fake_tree(build)
        (build / "bin" / "omniprobe").unlink()
        source = tmp_path / "src" / "omniprobe"
        source.parent.mkdir()
        source.write_text("#!/bin/sh\n")
        (build / "bin" / "omniprobe").symlink_to(source)
        flags = instrumentation_flags(
            "memory", offload_arch="gfx90a", omniprobe_path=str(build / "bin" / "omniprobe")
        )
        assert flags[0].startswith(f"-fpass-plugin={plugins}")

    def test_env_override(self, tmp_path, monkeypatch):
        _, plugins = self._fake_tree(tmp_path)
        monkeypatch.setenv("OMNIPROBE_PLUGIN_DIR", str(plugins))
        flags = instrumentation_flags("memory", offload_arch="gfx90a")
        assert flags[0] == f"-fpass-plugin={plugins / 'libAMDGCNSubmitAddressMessages-rocm.so'}"

    def test_missing_plugins_raise(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OMNIPROBE_PLUGIN_DIR", raising=False)
        (tmp_path / "bin").mkdir()
        exe = tmp_path / "bin" / "omniprobe"
        exe.write_text("")
        with pytest.raises(FileNotFoundError, match="OMNIPROBE_PLUGIN_DIR"):
            instrumentation_flags("memory", offload_arch="gfx90a", omniprobe_path=str(exe))

    def test_unknown_analysis(self, tmp_path, monkeypatch):
        _, plugins = self._fake_tree(tmp_path)
        monkeypatch.setenv("OMNIPROBE_PLUGIN_DIR", str(plugins))
        with pytest.raises(ValueError, match="unknown analysis"):
            instrumentation_flags("heatmap", offload_arch="gfx90a")

    def test_undetectable_arch_raises(self, tmp_path, monkeypatch):
        _, plugins = self._fake_tree(tmp_path)
        monkeypatch.setenv("OMNIPROBE_PLUGIN_DIR", str(plugins))
        monkeypatch.setattr("omniprobe_ik.api.detect_offload_arch", lambda: None)
        with pytest.raises(RuntimeError, match="offload_arch"):
            instrumentation_flags("memory")
