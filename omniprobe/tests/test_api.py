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


class TestKernelFilterMissed:
    """Parsing Omniprobe b71927b's end-of-run warning (text from src/interceptor.cc:311-321)."""

    LOG = (
        "omniprobe: WARNING: no dispatched kernel matched the kernel filter -k '^gstride', so "
        "none ran instrumented. The filter is an ECMAScript regex searched in the demangled "
        "name; the instrumented kernels dispatched were:\n"
        "  void gstride<1>(float const*, float*) [clone .kd] (1 dispatch)\n"
        "  via_table(float* const*) [clone .kd] (3 dispatches)\n"
        "  ... and 4 more\n"
        "Ending kernelDB\n"
    )

    def test_parses_filter_and_candidates(self):
        from omniprobe_ik.api import kernel_filter_missed

        assert kernel_filter_missed(self.LOG) == (
            "^gstride",
            [
                "void gstride<1>(float const*, float*) [clone .kd]",
                "via_table(float* const*) [clone .kd]",
            ],
        )

    def test_absent_warning_is_none(self):
        from omniprobe_ik.api import kernel_filter_missed

        assert kernel_filter_missed("Kernel filter excluded foo; running uninstrumented") is None
        assert kernel_filter_missed(None) is None

    def test_raised_instead_of_rebuild_advice(self):
        from omniprobe_ik import KernelFilterMatchedNothingError, NoInstrumentedKernelsError
        from omniprobe_ik.api import _raise_no_data

        with pytest.raises(KernelFilterMatchedNothingError) as err:
            _raise_no_data("memory", "./app", "^gstride", [], self.LOG)
        assert isinstance(err.value, NoInstrumentedKernelsError)
        assert "do not rebuild" in str(err.value) and "Rebuild" not in str(err.value)


class TestEnvironmentProblem:
    def test_22c4d95_kerneldb_error_is_environment(self):
        from omniprobe_ik.api import environment_problem

        log = (
            "HSA intercept registered\nomniprobe: ERROR: kernelDB could not read /tmp/x.co: boom\n"
        )
        assert "ROCM_PATH" in environment_problem(log)

    def test_13ca707_missing_objdump_is_environment(self):
        from omniprobe_ik.api import environment_problem

        log = (
            "HSA intercept registered\nsh: /opt/rocm/llvm/bin/llvm-objdump: No such file or "
            "directory\n[hsaInterceptor] Command failed\n"
        )
        assert "could not run /opt/rocm/llvm/bin/llvm-objdump" in environment_problem(log)


class TestUnattributedWarning:
    def test_kernel_without_any_line_is_blamed_on_g(self):
        from omniprobe_ik.api import unattributed_accesses_warning

        text = unattributed_accesses_warning({"k": 64}, set())
        assert "-g" in text and "not missing" not in text

    def test_kernel_with_attributed_accesses_is_not(self):
        from omniprobe_ik.api import unattributed_accesses_warning

        text = unattributed_accesses_warning({"k": 64}, {"k"})
        assert "-g is not missing" in text and "Rebuild with -g" not in text


class TestOmniprobeMessagesPassThrough:
    """Omniprobe's own warnings reach the caller, so an empty result explains itself."""

    FUTURE = (
        "Found instrumented alternative for k(int*) [clone .kd]\n"
        "omniprobe: WARNING: the random dispatch was excluded by the kernel filter\n"
        "  k(int*) [clone .kd] (3 dispatches)\n"
        "Ending kernelDB\n"
    )

    def test_messages_keep_continuation_lines(self):
        from omniprobe_ik.api import omniprobe_messages

        assert omniprobe_messages(self.FUTURE) == [
            "omniprobe: WARNING: the random dispatch was excluded by the kernel filter\n"
            "  k(int*) [clone .kd] (3 dispatches)"
        ]
        assert omniprobe_messages(None) == []

    def test_unknown_warning_is_in_the_error(self):
        from omniprobe_ik import NoInstrumentedKernelsError
        from omniprobe_ik.api import _raise_no_data

        with pytest.raises(NoInstrumentedKernelsError) as err:
            _raise_no_data("memory", "./app", None, [], self.FUTURE)
        assert "the random dispatch was excluded by the kernel filter" in str(err.value)

    def test_random_with_filter_is_named_as_the_cause(self):
        from omniprobe_ik import NoInstrumentedKernelsError
        from omniprobe_ik.api import _raise_no_data

        log = (
            "Found instrumented alternative for a(int*) [clone .kd]\n"
            "Kernel filter excluded a(int*) [clone .kd]; running uninstrumented\n"
            "Found instrumented alternative for b(int*) [clone .kd]\n"
        )
        with pytest.raises(NoInstrumentedKernelsError) as err:
            _raise_no_data("memory", "./app", "^b", [], log, "random")
        text = str(err.value)
        assert "dispatches='random'" in text and "do not rebuild" in text
        assert "Rebuild the application" not in text
        # Same log with dispatches="all" keeps the ordinary explanation.
        with pytest.raises(NoInstrumentedKernelsError) as err:
            _raise_no_data("memory", "./app", "^b", [], log, "all")
        assert "dispatches='random'" not in str(err.value)
