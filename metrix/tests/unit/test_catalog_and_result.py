# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the metric catalog helpers and the profiler result placeholders."""

from __future__ import annotations

import pytest

from metrix.metrics import METRIC_CATALOG, METRIC_PROFILES
from metrix.metrics.catalog import (
    get_metric_info,
    get_metrics_by_category,
    list_all_metrics,
    list_all_profiles,
)
from metrix.profiler.engine import Profiler
from metrix.profiler.result import CollectionResult, KernelDispatch

KNOWN_METRIC = "memory.hbm_bandwidth_utilization"


# --------------------------------------------------------------------------
# catalog helpers
# --------------------------------------------------------------------------


def test_get_metrics_by_category_filters_to_that_category():
    category = METRIC_CATALOG[KNOWN_METRIC]["category"].value
    found = get_metrics_by_category(category)
    assert KNOWN_METRIC in found
    assert all(METRIC_CATALOG[m]["category"].value == category for m in found)


def test_get_metrics_by_unknown_category_is_empty():
    assert get_metrics_by_category("no-such-category") == []


def test_get_metric_info_returns_the_catalog_entry():
    assert get_metric_info(KNOWN_METRIC) is METRIC_CATALOG[KNOWN_METRIC]


def test_get_metric_info_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric_info("memory.not_a_real_metric")


def test_list_all_metrics_matches_catalog():
    assert list_all_metrics() == list(METRIC_CATALOG.keys())


def test_list_all_profiles_matches_profiles():
    assert list_all_profiles() == list(METRIC_PROFILES.keys())


def test_every_profile_references_only_real_metrics():
    # A profile naming a metric that does not exist would fail at runtime.
    for name, definition in METRIC_PROFILES.items():
        for metric in definition["metrics"]:
            assert metric in METRIC_CATALOG, f"profile '{name}' references unknown '{metric}'"


# --------------------------------------------------------------------------
# profiler.result
# --------------------------------------------------------------------------


def test_kernel_dispatch_holds_its_fields():
    d = KernelDispatch(
        dispatch_id=1,
        kernel_name="gemm",
        device_id=0,
        grid_size=(64, 1, 1),
        block_size=(256, 1, 1),
        duration_ns=1234,
        raw_counters={"SQ_WAVES": 8.0},
        metrics={KNOWN_METRIC: 0.5},
    )
    assert d.kernel_name == "gemm"
    assert d.grid_size == (64, 1, 1)
    # Optional source info defaults to absent.
    assert d.source_file is None
    assert d.source_line is None


def test_collection_result_starts_empty():
    assert CollectionResult().dispatches == []


def test_collection_result_query_returns_dispatches():
    result = CollectionResult()
    result.dispatches.append("d1")
    # query() is a passthrough until filtering is implemented.
    assert result.query() == ["d1"]
    assert result.query(kernel_pattern="anything") == ["d1"]


def test_collection_result_exporters_are_still_stubs(tmp_path):
    result = CollectionResult()
    # Documented as unimplemented: they must no-op rather than raise.
    assert result.to_json(tmp_path / "out.json") is None
    assert result.to_dataframe() is None


# --------------------------------------------------------------------------
# profiler.engine
# --------------------------------------------------------------------------


def test_profiler_records_requested_arch():
    assert Profiler(device_arch="gfx942").device_arch == "gfx942"


def test_profiler_arch_defaults_to_none():
    assert Profiler().device_arch is None


def test_profiler_profile_is_not_implemented():
    # The real engine lives in the backends; this placeholder must say so
    # loudly rather than silently returning nothing.
    with pytest.raises(NotImplementedError):
        Profiler().profile("./app")
