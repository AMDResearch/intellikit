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
from metrix.backends.base import CounterBackend, DeviceSpecs, ProfileResult
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
    ("gfx1103", "GFX1103Backend"),
    ("gfx1150", "GFX1150Backend"),
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


def _gpu_payload(arch, memory_clock_rate_khz, memory_bus_width_bits):
    """A gpu_query record, varying only the fields the bandwidth math reads."""
    return {
        "name": "AMD Radeon Graphics",
        "gcn_arch_name": arch,
        "num_cu": 6,
        "wavefront_size": 32,
        "max_threads_per_multiprocessor": 2048,
        "clock_rate_khz": 2799000,
        "memory_clock_rate_khz": memory_clock_rate_khz,
        "memory_bus_width_bits": memory_bus_width_bits,
        "l2_cache_size_bytes": 2 * 1024 * 1024,
        "max_shared_memory_per_multiprocessor": 64 * 1024,
    }


# Parameter is `gfx_arch`, not `arch`: the autouse skip_arch_mismatch fixture
# skips any test with an `arch` parameter that differs from the local GPU, which
# would reduce this whole table to the one row matching whatever card is in the
# machine. The bandwidth math is pure and _run_gpu_query is mocked, so every row
# can and must run everywhere.
@pytest.mark.parametrize(
    "gfx_arch, mem_clock_khz, bus_bits, expected_gbs",
    [
        # gfx1103 (Phoenix / Radeon 780M) reads system memory, not GDDR6, and
        # the same part ships with either type. DDR5-5600 dual channel =
        # 2800 MHz x 2 x 128-bit / 8, measured on a Ryzen 9 7940HS; the GDDR6
        # fallback would claim 716.8 GB/s.
        ("gfx1103", 2800000, 128, 89.6),
        # ...and the LPDDR5X-7500 package of that same 7940HS, which a fixed
        # 2x multiplier would have put at 30 GB/s instead of 120.
        ("gfx1103", 937500, 128, 120.0),
        # gfx1150 (Strix Point) likewise: LPDDR5X-7500 as validated, and the
        # DDR5 board, which a fixed 8x multiplier would have put at 358.4 GB/s.
        ("gfx1150", 937000, 128, 119.936),
        ("gfx1150", 2800000, 128, 89.6),
        # gfx1151 (Strix Halo), LPDDR5X-8000 over 256-bit, as validated.
        ("gfx1151", 1000000, 256, 256.0),
        # Discrete RDNA must still take the 16x GDDR6 path.
        ("gfx1100", 2500000, 384, 1920.0),
        # The MCLK test must stay scoped to APUs: this RX 6800 XT reports the
        # very same 1000 MHz as the gfx1151 above, yet is GDDR6 and needs 16x.
        # Reading memory type from the clock alone would call it LPDDR5X and
        # halve its peak bandwidth.
        ("gfx1030", 1000000, 256, 512.0),
    ],
)
def test_query_device_specs_memory_bandwidth(gfx_arch, mem_clock_khz, bus_bits, expected_gbs):
    payload = [_gpu_payload(gfx_arch, mem_clock_khz, bus_bits)]
    with patch.object(device_info, "_run_gpu_query", return_value=payload):
        specs = device_info.query_device_specs(gfx_arch)
    assert specs.arch == gfx_arch
    assert specs.hbm_bandwidth_gbs == pytest.approx(expected_gbs)


# --------------------------------------------------------------------------
# CounterBackend._merge_dispatches
# --------------------------------------------------------------------------


class _DummyBackend(CounterBackend):
    """Minimal concrete backend, just enough to exercise _merge_dispatches."""

    def _get_device_specs(self):
        return DeviceSpecs(arch="dummy", name="dummy")

    def _run_rocprof(self, *args, **kwargs):
        raise NotImplementedError


def _dispatch(dispatch_id, duration_ns, counters):
    return ProfileResult(
        dispatch_id=dispatch_id,
        kernel_name="k",
        gpu_id=0,
        duration_ns=duration_ns,
        grid_size=(1, 1, 1),
        workgroup_size=(1, 1, 1),
        counters=dict(counters),
    )


@pytest.mark.parametrize(
    "durations,flop_counts",
    [
        ([1000, 2000], [100, 200]),
        ([1000, 1000, 1000], [50, 50, 50]),
        ([500, 1500, 2000, 4000], [10, 30, 20, 60]),
        ([1, 1], [1, 1]),
    ],
)
def test_merge_dispatches_keeps_rate_metrics_consistent(durations, flop_counts):
    """A kernel launched multiple times per run merges to a single average
    dispatch whose counter/duration ratio must match the same rate computed
    from the raw per-dispatch data -- summing one side of the ratio but
    averaging the other silently inflates rate metrics (GFLOPS, bandwidth %)
    by roughly the dispatch count.
    """
    dispatches = [
        _dispatch(i, d, {"SQ_INSTS_VALU_ADD_F32": f})
        for i, (d, f) in enumerate(zip(durations, flop_counts))
    ]

    merged = _DummyBackend()._merge_dispatches(dispatches)

    expected_rate = sum(flop_counts) / sum(durations)
    actual_rate = merged.counters["SQ_INSTS_VALU_ADD_F32"] / merged.duration_ns
    assert actual_rate == pytest.approx(expected_rate)


def test_merge_dispatches_averages_duration_and_counters():
    """Pinning test for the specific mechanism: both duration_ns and the
    counters on the merged result describe a single average dispatch, so
    neither side of a rate metric carries the dispatch count.
    """
    dispatches = [
        _dispatch(0, 1000, {"C": 1}),
        _dispatch(1, 2000, {"C": 4}),
        _dispatch(2, 3000, {"C": 7}),
    ]

    merged = _DummyBackend()._merge_dispatches(dispatches)

    assert merged.duration_ns == 2000
    assert merged.counters["C"] == pytest.approx(4.0)


def test_merge_dispatches_single_dispatch_is_a_no_op():
    (dispatch,) = [_dispatch(0, 1234, {"SQ_INSTS_VALU_ADD_F32": 42})]

    merged = _DummyBackend()._merge_dispatches([dispatch])

    assert merged.duration_ns == 1234
    assert merged.counters["SQ_INSTS_VALU_ADD_F32"] == 42


def test_merge_dispatches_keeps_scale_of_counters_missing_from_some_passes():
    """Multi-pass profiling: each pass may only report a subset of counters,
    e.g. rocprofv3 splitting counters that can't be collected in a single
    pass across separate replays of the kernel. Each counter is averaged over
    the dispatches that actually reported it, so a counter seen once keeps
    its value instead of being diluted by the passes that never measured it.
    """
    dispatches = [
        _dispatch(0, 1000, {"A": 10}),
        _dispatch(1, 1000, {"B": 20}),
    ]

    merged = _DummyBackend()._merge_dispatches(dispatches)

    assert merged.counters["A"] == 10
    assert merged.counters["B"] == 20
    assert merged.duration_ns == 1000


@pytest.mark.parametrize("counter_name", ["GpuBusyPercent", "L2CacheHit", "VALUUtil", "MemoryBusy"])
def test_merge_dispatches_averages_utilization_style_counters(counter_name):
    """Counters that are already ratios/percentages need no special casing:
    averaging every counter keeps two 50% utilization samples at ~50% instead
    of turning them into 100%.
    """
    dispatches = [
        _dispatch(0, 1000, {counter_name: 40.0}),
        _dispatch(1, 3000, {counter_name: 60.0}),
    ]

    merged = _DummyBackend()._merge_dispatches(dispatches)

    assert merged.counters[counter_name] == pytest.approx(50.0)


def test_merge_dispatches_sets_num_dispatches():
    dispatches = [_dispatch(i, 1000, {"C": 1}) for i in range(4)]

    merged = _DummyBackend()._merge_dispatches(dispatches)

    assert merged._num_dispatches == 4


def test_merge_dispatches_rejects_empty_list():
    with pytest.raises(ValueError, match="empty dispatch list"):
        _DummyBackend()._merge_dispatches([])
