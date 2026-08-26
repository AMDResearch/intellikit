# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the Metrix MCP server tool bodies and entry point.

The existing ``test_mcp_server.py`` checks that the catalog the tools expose is
sane. These cover the code paths themselves — result marshalling, the
no-GPU fallback, and ``main()`` — none of which need a GPU.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from metrix.mcp import server as mcp_server
from metrix.metrics import METRIC_CATALOG

KNOWN_METRIC = "memory.hbm_bandwidth_utilization"


def _unwrap(tool):
    """Return the underlying function.

    ``@mcp.tool()`` yields a ``FunctionTool`` in some fastmcp versions and the
    bare function in others; ``.fn`` is the escape hatch when it is wrapped.
    """
    return getattr(tool, "fn", tool)


profile_metrics = _unwrap(mcp_server.profile_metrics)
list_available_metrics = _unwrap(mcp_server.list_available_metrics)


class FakeStat:
    def __init__(self, avg, unit=""):
        self.avg = avg
        self.unit = unit


class FakeKernel:
    def __init__(self, name, duration_avg=None, metrics=None, with_metrics=True):
        self.name = name
        # An object with no .avg exercises the hasattr fallback to 0.0.
        self.duration_us = FakeStat(duration_avg) if duration_avg is not None else object()
        if with_metrics:
            self.metrics = metrics or {}


class FakeResults:
    def __init__(self, kernels):
        self.kernels = kernels


class FakeProfiler:
    def __init__(self, kernels=None, available=None):
        self._kernels = kernels if kernels is not None else []
        self._available = available or [KNOWN_METRIC]
        self.profile_calls = []

    def list_metrics(self):
        return list(self._available)

    def profile(self, command, metrics=None):
        self.profile_calls.append({"command": command, "metrics": metrics})
        return FakeResults(self._kernels)


# --------------------------------------------------------------------------
# profile_metrics
# --------------------------------------------------------------------------


def test_profile_metrics_marshals_kernel_fields():
    kernel = FakeKernel("gemm", 12.5, {KNOWN_METRIC: FakeStat(87.5, "%")})
    with patch.object(mcp_server, "Metrix", return_value=FakeProfiler([kernel])):
        out = profile_metrics("./app", [KNOWN_METRIC])

    assert len(out["kernels"]) == 1
    entry = out["kernels"][0]
    assert entry["name"] == "gemm"
    assert entry["duration_us_avg"] == 12.5
    assert entry["metrics"][KNOWN_METRIC] == {"avg": 87.5, "unit": "%"}


def test_profile_metrics_defaults_to_all_available_metrics():
    profiler = FakeProfiler([], available=[KNOWN_METRIC, "memory.l2_hit_rate"])
    with patch.object(mcp_server, "Metrix", return_value=profiler):
        profile_metrics("./app")
    # metrics=None must be resolved via list_metrics before profiling.
    assert profiler.profile_calls[0]["metrics"] == [KNOWN_METRIC, "memory.l2_hit_rate"]


def test_profile_metrics_passes_command_through():
    profiler = FakeProfiler([])
    with patch.object(mcp_server, "Metrix", return_value=profiler):
        profile_metrics("python train.py --size 1024", [KNOWN_METRIC])
    assert profiler.profile_calls[0]["command"] == "python train.py --size 1024"


def test_profile_metrics_duration_without_avg_becomes_zero():
    kernel = FakeKernel("nodur", duration_avg=None)
    with patch.object(mcp_server, "Metrix", return_value=FakeProfiler([kernel])):
        out = profile_metrics("./app", [KNOWN_METRIC])
    assert out["kernels"][0]["duration_us_avg"] == 0.0


def test_profile_metrics_skips_metrics_absent_from_kernel():
    kernel = FakeKernel("gemm", 1.0, {})
    with patch.object(mcp_server, "Metrix", return_value=FakeProfiler([kernel])):
        out = profile_metrics("./app", [KNOWN_METRIC])
    assert out["kernels"][0]["metrics"] == {}


def test_profile_metrics_handles_kernel_without_metrics_attr():
    kernel = FakeKernel("bare", 1.0, with_metrics=False)
    with patch.object(mcp_server, "Metrix", return_value=FakeProfiler([kernel])):
        out = profile_metrics("./app", [KNOWN_METRIC])
    assert out["kernels"][0]["metrics"] == {}


def test_profile_metrics_metric_without_avg_becomes_zero():
    class NoAvg:
        unit = "%"

    kernel = FakeKernel("gemm", 1.0, {KNOWN_METRIC: NoAvg()})
    with patch.object(mcp_server, "Metrix", return_value=FakeProfiler([kernel])):
        out = profile_metrics("./app", [KNOWN_METRIC])
    assert out["kernels"][0]["metrics"][KNOWN_METRIC]["avg"] == 0.0


def test_profile_metrics_empty_result_is_empty_list():
    with patch.object(mcp_server, "Metrix", return_value=FakeProfiler([])):
        assert profile_metrics("./app", [KNOWN_METRIC]) == {"kernels": []}


def test_profile_metrics_handles_multiple_kernels():
    kernels = [FakeKernel("a", 1.0, {}), FakeKernel("b", 2.0, {})]
    with patch.object(mcp_server, "Metrix", return_value=FakeProfiler(kernels)):
        out = profile_metrics("./app", [KNOWN_METRIC])
    assert [k["name"] for k in out["kernels"]] == ["a", "b"]


# --------------------------------------------------------------------------
# list_available_metrics
# --------------------------------------------------------------------------


def test_list_metrics_falls_back_to_catalog_without_a_gpu():
    # Backend construction failing must not break discovery.
    with patch.object(mcp_server, "Metrix", side_effect=RuntimeError("no GPU")):
        out = list_available_metrics()
    assert out["metrics"] == sorted(METRIC_CATALOG.keys())


def test_list_metrics_prefers_backend_over_catalog():
    with patch.object(mcp_server, "Metrix", return_value=FakeProfiler(available=["z.b", "a.a"])):
        out = list_available_metrics()
    assert out["metrics"] == ["a.a", "z.b"]


def test_yaml_only_metric_is_categorised_by_prefix():
    # Not in METRIC_CATALOG, so the category comes from the name prefix.
    with patch.object(mcp_server, "Metrix", return_value=FakeProfiler(available=["custom.thing"])):
        out = list_available_metrics()
    assert out["by_category"]["custom"] == ["custom.thing"]


def test_undotted_unknown_metric_falls_into_other():
    with patch.object(mcp_server, "Metrix", return_value=FakeProfiler(available=["weird"])):
        out = list_available_metrics()
    assert out["by_category"]["other"] == ["weird"]


def test_catalog_metric_uses_its_declared_category():
    with patch.object(mcp_server, "Metrix", return_value=FakeProfiler(available=[KNOWN_METRIC])):
        out = list_available_metrics()
    expected = METRIC_CATALOG[KNOWN_METRIC]["category"].value
    assert out["by_category"][expected] == [KNOWN_METRIC]


def test_list_metrics_includes_usage_note():
    with patch.object(mcp_server, "Metrix", return_value=FakeProfiler(available=[KNOWN_METRIC])):
        assert "profile_metrics" in list_available_metrics()["note"]


# --------------------------------------------------------------------------
# main()
# --------------------------------------------------------------------------


def test_main_defaults_to_stdio():
    with (
        patch("sys.argv", ["metrix-mcp"]),
        patch.object(mcp_server.mcp, "run") as run,
    ):
        mcp_server.main()
    run.assert_called_once_with(transport="stdio")


def test_main_explicit_stdio():
    with (
        patch("sys.argv", ["metrix-mcp", "--transport", "stdio"]),
        patch.object(mcp_server.mcp, "run") as run,
    ):
        mcp_server.main()
    run.assert_called_once_with(transport="stdio")


def test_main_http_uses_documented_defaults():
    with (
        patch("sys.argv", ["metrix-mcp", "--transport", "http"]),
        patch.object(mcp_server.mcp, "run") as run,
    ):
        mcp_server.main()
    run.assert_called_once_with(
        transport="streamable-http", host="127.0.0.1", port=8000, path="/metrix"
    )


def test_main_http_honours_overrides():
    with (
        patch(
            "sys.argv",
            [
                "metrix-mcp",
                "--transport",
                "http",
                "--host",
                "0.0.0.0",
                "--port",
                "9001",
                "--path",
                "/custom",
            ],
        ),
        patch.object(mcp_server.mcp, "run") as run,
    ):
        mcp_server.main()
    run.assert_called_once_with(
        transport="streamable-http", host="0.0.0.0", port=9001, path="/custom"
    )


def test_main_rejects_unknown_transport():
    with (
        patch("sys.argv", ["metrix-mcp", "--transport", "carrier-pigeon"]),
        pytest.raises(SystemExit) as exc,
    ):
        mcp_server.main()
    assert exc.value.code == 2


def test_port_is_parsed_as_int():
    with (
        patch("sys.argv", ["metrix-mcp", "--transport", "http", "--port", "1234"]),
        patch.object(mcp_server.mcp, "run") as run,
    ):
        mcp_server.main()
    assert run.call_args.kwargs["port"] == 1234


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def test_both_tools_are_registered():
    """Guard against a tool silently failing to register.

    Driven with ``asyncio.run`` rather than a pytest async plugin so it needs
    no extra test dependency. Fails loudly rather than skipping if the fastmcp
    listing API moves — a permanently-skipped test contributes nothing while
    looking green.
    """
    import asyncio
    import inspect

    lister = getattr(mcp_server.mcp, "list_tools", None) or getattr(
        mcp_server.mcp, "get_tools", None
    )
    if lister is None:
        pytest.fail("fastmcp exposes neither list_tools() nor get_tools()")

    tools = lister()
    if inspect.isawaitable(tools):
        tools = asyncio.run(tools)

    # dict keyed by name in some versions, iterable of tool objects in others
    names = set(tools) if isinstance(tools, dict) else {getattr(t, "name", t) for t in tools}
    assert {"profile_metrics", "list_available_metrics"} <= names
