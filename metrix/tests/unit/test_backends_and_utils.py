# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for architecture detection, backend plumbing, and shared helpers.

None of these need a GPU: ``rocminfo``/``hipcc``/``rocprofv3`` are all replaced
at their call sites, so the failure paths (which are the interesting ones) can
actually be reached.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from metrix.backends import detect as detect_mod
from metrix.backends import device_info
from metrix.backends.detect import detect_gpu_arch, detect_or_default
from metrix.utils.common import split_counters_into_passes

# --------------------------------------------------------------------------
# detect
# --------------------------------------------------------------------------


def _completed(stdout="", stderr="", rc=0):
    return subprocess.CompletedProcess(
        args=["rocminfo"], returncode=rc, stdout=stdout, stderr=stderr
    )


def test_detect_parses_arch_from_rocminfo():
    out = "Agent 1\n  Name:  gfx942\n"
    with patch.object(subprocess, "run", return_value=_completed(stdout=out)):
        assert detect_gpu_arch() == "gfx942"


def test_detect_raises_when_rocminfo_returns_nonzero():
    with patch.object(subprocess, "run", return_value=_completed(stderr="nope", rc=1)):
        with pytest.raises(RuntimeError, match="rocminfo failed"):
            detect_gpu_arch()


def test_detect_raises_when_no_arch_in_output():
    with patch.object(subprocess, "run", return_value=_completed(stdout="nothing here")):
        with pytest.raises(RuntimeError, match="No AMD GPU architecture"):
            detect_gpu_arch()


def test_detect_raises_when_rocminfo_missing():
    with patch.object(subprocess, "run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="rocminfo not found"):
            detect_gpu_arch()


def test_detect_raises_on_timeout():
    with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("rocminfo", 5)):
        with pytest.raises(RuntimeError, match="timed out"):
            detect_gpu_arch()


def test_detect_or_default_prefers_explicit_arch():
    # Explicit request must short-circuit before any subprocess call.
    with patch.object(detect_mod, "detect_gpu_arch") as probe:
        assert detect_or_default("gfx1201") == "gfx1201"
        probe.assert_not_called()


def test_detect_or_default_autodetects_when_unset():
    with patch.object(detect_mod, "detect_gpu_arch", return_value="gfx950"):
        assert detect_or_default() == "gfx950"


def test_detect_or_default_falls_back_to_gfx942():
    with patch.object(detect_mod, "detect_gpu_arch", side_effect=RuntimeError("no gpu")):
        assert detect_or_default() == "gfx942"


# --------------------------------------------------------------------------
# split_counters_into_passes
# --------------------------------------------------------------------------


def test_empty_counters_yield_one_empty_pass():
    # Timing-only mode still needs exactly one (empty) pass.
    assert split_counters_into_passes([]) == [[]]


def test_no_block_limits_returns_single_pass_when_small():
    counters = ["A", "B"]
    assert split_counters_into_passes(counters) == [counters]


def test_no_block_limits_chunks_by_max_per_pass():
    counters = [f"C{i}" for i in range(7)]
    passes = split_counters_into_passes(counters, max_per_pass=3)
    assert [len(p) for p in passes] == [3, 3, 1]
    assert sum(passes, []) == counters


def test_simple_chunking_logs_when_logger_supplied():
    logger = MagicMock()
    split_counters_into_passes([f"C{i}" for i in range(5)], max_per_pass=2, logger=logger)
    logger.info.assert_called_once()


def test_block_limits_without_mapper_is_an_error():
    with pytest.raises(ValueError, match="get_counter_block must be provided"):
        split_counters_into_passes(["A"], block_limits={"SQ": 2})


def test_block_aware_packing_respects_per_block_limit():
    counters = ["SQ_1", "SQ_2", "SQ_3"]
    passes = split_counters_into_passes(
        counters,
        block_limits={"SQ": 2},
        get_counter_block=lambda c: c.split("_")[0],
    )
    # SQ allows 2 per pass, so 3 counters need 2 passes.
    assert [len(p) for p in passes] == [2, 1]
    assert sorted(sum(passes, [])) == sorted(counters)


def test_block_aware_packing_interleaves_blocks():
    counters = ["SQ_1", "SQ_2", "TA_1"]
    passes = split_counters_into_passes(
        counters,
        block_limits={"SQ": 2, "TA": 2},
        get_counter_block=lambda c: c.split("_")[0],
    )
    # Both blocks fit within their limits, so one pass suffices.
    assert len(passes) == 1
    assert sorted(passes[0]) == sorted(counters)


def test_unknown_block_uses_default_limit():
    counters = [f"XX_{i}" for i in range(5)]
    passes = split_counters_into_passes(
        counters,
        block_limits={"SQ": 8},
        get_counter_block=lambda c: c.split("_")[0],
        default_block_limit=2,
        max_per_pass=8,
    )
    assert [len(p) for p in passes] == [2, 2, 1]


def test_max_per_pass_caps_total_across_blocks():
    counters = ["SQ_1", "SQ_2", "TA_1", "TA_2"]
    passes = split_counters_into_passes(
        counters,
        block_limits={"SQ": 4, "TA": 4},
        get_counter_block=lambda c: c.split("_")[0],
        max_per_pass=2,
    )
    assert all(len(p) <= 2 for p in passes)
    assert sorted(sum(passes, [])) == sorted(counters)


def test_block_aware_packing_logs_when_logger_supplied():
    logger = MagicMock()
    split_counters_into_passes(
        ["SQ_1"],
        block_limits={"SQ": 1},
        get_counter_block=lambda c: "SQ",
        logger=logger,
    )
    logger.debug.assert_called()
    logger.info.assert_called_once()


def test_every_counter_survives_packing():
    counters = [f"{b}_{i}" for b in ("SQ", "TA", "TCC") for i in range(5)]
    passes = split_counters_into_passes(
        counters,
        block_limits={"SQ": 2, "TA": 1, "TCC": 3},
        get_counter_block=lambda c: c.split("_")[0],
        max_per_pass=4,
    )
    # Nothing may be dropped or duplicated by the bin-packer.
    assert sorted(sum(passes, [])) == sorted(counters)


# --------------------------------------------------------------------------
# gfx backends
# --------------------------------------------------------------------------

BACKENDS = [
    ("gfx90a", "GFX90aBackend"),
    ("gfx942", "GFX942Backend"),
    ("gfx950", "GFX950Backend"),
    ("gfx1030", "GFX1030Backend"),
    ("gfx1100", "GFX1100Backend"),
    ("gfx1151", "GFX1151Backend"),
    ("gfx1201", "GFX1201Backend"),
]


def _make_backend(module_name, class_name):
    """Build a backend with device probing stubbed out.

    Uses a real ``DeviceSpecs`` rather than a mock: the base class calls
    ``dataclasses.fields()`` on it to build expression variables, which a
    ``MagicMock`` cannot satisfy.
    """
    import importlib

    from metrix.backends.base import DeviceSpecs

    mod = importlib.import_module(f"metrix.backends.{module_name}")
    specs = DeviceSpecs(
        arch=module_name,
        name=f"test-{module_name}",
        num_cu=64,
        max_waves_per_cu=32,
        wavefront_size=64,
        base_clock_mhz=1700.0,
        hbm_bandwidth_gbs=3200.0,
        l2_size_mb=8.0,
        lds_size_per_cu_kb=64.0,
    )
    with patch.object(mod, "query_device_specs", return_value=specs):
        return mod, getattr(mod, class_name)()


@pytest.mark.parametrize("module_name,class_name", BACKENDS)
def test_backend_reports_its_own_arch(module_name, class_name):
    _mod, backend = _make_backend(module_name, class_name)
    assert backend.device_specs.arch == module_name


@pytest.mark.parametrize("module_name,class_name", BACKENDS)
def test_backend_block_limits_are_positive_ints(module_name, class_name):
    _mod, backend = _make_backend(module_name, class_name)
    limits = backend._get_counter_block_limits()
    assert limits, f"{class_name} declared no block limits"
    assert all(isinstance(v, int) and v > 0 for v in limits.values())


@pytest.mark.parametrize("module_name,class_name", BACKENDS)
def test_backend_groups_counters_without_dropping_any(module_name, class_name):
    _mod, backend = _make_backend(module_name, class_name)
    limits = backend._get_counter_block_limits()
    block = next(iter(limits))
    counters = [f"{block}_{i}" for i in range(limits[block] + 1)]
    passes = backend._get_counter_groups(counters)
    assert sorted(sum(passes, [])) == sorted(counters)


@pytest.mark.parametrize("module_name,class_name", BACKENDS)
def test_backend_run_rocprof_delegates_to_wrapper(module_name, class_name):
    import sys

    _mod, backend = _make_backend(module_name, class_name)
    # The RDNA backends subclass GFX1201Backend and inherit _run_rocprof, so
    # ROCProfV3Wrapper must be patched in whichever module defines it.
    defining_mod = sys.modules[type(backend)._run_rocprof.__module__]
    sentinel = [object()]
    wrapper = MagicMock()
    wrapper.profile.return_value = sentinel
    with patch.object(defining_mod, "ROCProfV3Wrapper", return_value=wrapper) as ctor:
        got = backend._run_rocprof("./app", ["SQ_WAVES"], kernel_filter="gemm.*")
    assert got is sentinel
    ctor.assert_called_once()
    assert wrapper.profile.call_args.kwargs["kernel_filter"] == "gemm.*"


def test_empty_counter_list_still_yields_one_pass():
    _mod, backend = _make_backend("gfx942", "GFX942Backend")
    assert backend._get_counter_groups([]) == [[]]


# --------------------------------------------------------------------------
# device_info
# --------------------------------------------------------------------------


def test_find_hip_source_returns_packaged_file():
    src = device_info._find_hip_source()
    # gpu_query.hip ships as package data, so this must resolve.
    assert src is not None
    assert src.name == "gpu_query.hip"


def test_find_hip_source_returns_none_when_absent():
    with (
        patch.object(Path, "is_file", return_value=False),
        patch("metrix.backends.__path__", []),
    ):
        assert device_info._find_hip_source() is None


def test_compile_requires_hipcc(tmp_path):
    device_info._compiled_binary = None
    with patch.object(device_info.shutil, "which", return_value=None):
        with pytest.raises(RuntimeError, match="hipcc not found"):
            device_info._compile_gpu_query(tmp_path / "gpu_query.hip")


def test_compile_reports_hipcc_failure(tmp_path):
    device_info._compiled_binary = None
    failed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")
    with (
        patch.object(device_info.shutil, "which", return_value="/usr/bin/hipcc"),
        patch.object(device_info.subprocess, "run", return_value=failed),
    ):
        with pytest.raises(RuntimeError, match="hipcc failed"):
            device_info._compile_gpu_query(tmp_path / "gpu_query.hip")


def test_compile_reports_timeout(tmp_path):
    device_info._compiled_binary = None
    with (
        patch.object(device_info.shutil, "which", return_value="/usr/bin/hipcc"),
        patch.object(
            device_info.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired("hipcc", 120),
        ),
    ):
        with pytest.raises(RuntimeError, match="timed out"):
            device_info._compile_gpu_query(tmp_path / "gpu_query.hip")


def test_run_gpu_query_errors_when_source_missing():
    with patch.object(device_info, "_find_hip_source", return_value=None):
        with pytest.raises(RuntimeError, match="Cannot find gpu_query.hip"):
            device_info._run_gpu_query()


def test_run_gpu_query_parses_json(tmp_path):
    payload = [{"arch": "gfx942", "cu_count": 304}]
    ok = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(payload), stderr="")
    with (
        patch.object(device_info, "_find_hip_source", return_value=tmp_path / "s.hip"),
        patch.object(device_info, "_compile_gpu_query", return_value=tmp_path / "bin"),
        patch.object(device_info.subprocess, "run", return_value=ok),
    ):
        assert device_info._run_gpu_query() == payload


def test_run_gpu_query_passes_device_id(tmp_path):
    ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
    with (
        patch.object(device_info, "_find_hip_source", return_value=tmp_path / "s.hip"),
        patch.object(device_info, "_compile_gpu_query", return_value=tmp_path / "bin"),
        patch.object(device_info.subprocess, "run", return_value=ok) as run,
    ):
        device_info._run_gpu_query(device_id=3)
    assert run.call_args[0][0][-1] == "3"


def test_run_gpu_query_reports_nonzero_exit(tmp_path):
    bad = subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="no device")
    with (
        patch.object(device_info, "_find_hip_source", return_value=tmp_path / "s.hip"),
        patch.object(device_info, "_compile_gpu_query", return_value=tmp_path / "bin"),
        patch.object(device_info.subprocess, "run", return_value=bad),
    ):
        with pytest.raises(RuntimeError, match="gpu_query failed"):
            device_info._run_gpu_query()


def test_run_gpu_query_reports_bad_json(tmp_path):
    ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="not json", stderr="")
    with (
        patch.object(device_info, "_find_hip_source", return_value=tmp_path / "s.hip"),
        patch.object(device_info, "_compile_gpu_query", return_value=tmp_path / "bin"),
        patch.object(device_info.subprocess, "run", return_value=ok),
    ):
        with pytest.raises(RuntimeError, match="invalid JSON"):
            device_info._run_gpu_query()


def test_run_gpu_query_reports_missing_binary(tmp_path):
    with (
        patch.object(device_info, "_find_hip_source", return_value=tmp_path / "s.hip"),
        patch.object(device_info, "_compile_gpu_query", return_value=tmp_path / "bin"),
        patch.object(device_info.subprocess, "run", side_effect=FileNotFoundError),
    ):
        with pytest.raises(RuntimeError, match="gpu_query failed"):
            device_info._run_gpu_query()
