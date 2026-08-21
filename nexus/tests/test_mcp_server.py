# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the Nexus MCP server.

``Nexus`` is replaced at the server's call site, so these exercise the tool
bodies and ``main()`` without a GPU or a built ``libnexus.so``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nexus import Kernel, Trace
from nexus.mcp import server as mcp_server


def _unwrap(tool):
    """``@mcp.tool()`` yields a FunctionTool in some fastmcp versions."""
    return getattr(tool, "fn", tool)


extract_kernel_code = _unwrap(mcp_server.extract_kernel_code)
list_kernels = _unwrap(mcp_server.list_kernels)


def _trace(**kernels):
    """Build a real Trace from raw kernel payloads."""
    return Trace({"kernels": kernels})


class FakeNexus:
    """Stands in for Nexus, recording how it was constructed and called."""

    last = None

    def __init__(self, log_level=1, extra_search_prefix=None):
        self.log_level = log_level
        self.commands = []
        self.trace = _trace()
        FakeNexus.last = self

    def run(self, command, **kw):
        self.commands.append(command)
        return self.trace


def _install(trace):
    """Patch Nexus in the server module, returning a context manager."""

    def factory(log_level=1, **kw):
        n = FakeNexus(log_level=log_level)
        n.trace = trace
        return n

    return patch.object(mcp_server, "Nexus", side_effect=factory)


# --------------------------------------------------------------------------
# extract_kernel_code
# --------------------------------------------------------------------------


def test_extract_reports_name_and_instruction_count():
    trace = _trace(vector_add={"assembly": ["a", "b", "c"], "hip": ["src"]})
    with _install(trace):
        out = extract_kernel_code(["python", "app.py"])
    k = out["kernels"][0]
    assert k["name"] == "vector_add"
    assert k["num_instructions"] == 3


def test_extract_includes_hip_source_when_present():
    with _install(_trace(k={"hip": ["__global__ void k() {}"]})):
        out = extract_kernel_code(["true"])
    assert out["kernels"][0]["hip_source"] == ["__global__ void k() {}"]


def test_extract_omits_hip_source_when_absent():
    with _install(_trace(k={"assembly": ["x"]})):
        out = extract_kernel_code(["true"])
    assert "hip_source" not in out["kernels"][0]


def test_extract_omits_assembly_by_default():
    with _install(_trace(k={"assembly": ["x", "y"]})):
        out = extract_kernel_code(["true"])
    assert "assembly" not in out["kernels"][0]


def test_extract_includes_assembly_when_requested():
    with _install(_trace(k={"assembly": ["x", "y"]})):
        out = extract_kernel_code(["true"], include_assembly=True)
    assert out["kernels"][0]["assembly"] == ["x", "y"]
    assert "assembly_truncated" not in out["kernels"][0]


def test_extract_truncates_long_assembly_at_100_lines():
    with _install(_trace(k={"assembly": [f"i{n}" for n in range(150)]})):
        out = extract_kernel_code(["true"], include_assembly=True)
    k = out["kernels"][0]
    assert len(k["assembly"]) == 100
    assert k["assembly_truncated"] is True


def test_extract_does_not_flag_truncation_at_exactly_100():
    with _install(_trace(k={"assembly": [f"i{n}" for n in range(100)]})):
        out = extract_kernel_code(["true"], include_assembly=True)
    k = out["kernels"][0]
    assert len(k["assembly"]) == 100
    assert "assembly_truncated" not in k


def test_extract_zero_instructions_for_kernel_without_assembly():
    with _install(_trace(k={})):
        out = extract_kernel_code(["true"])
    assert out["kernels"][0]["num_instructions"] == 0


def test_extract_language_defaults_to_unknown():
    with _install(_trace(k={"assembly": ["x"]})):
        out = extract_kernel_code(["true"])
    assert out["kernels"][0]["language"] == "unknown"


def test_extract_empty_trace_yields_no_kernels():
    with _install(_trace()):
        assert extract_kernel_code(["true"]) == {"kernels": []}


def test_extract_handles_multiple_kernels():
    with _install(_trace(a={"assembly": ["x"]}, b={"assembly": ["y", "z"]})):
        out = extract_kernel_code(["true"])
    assert sorted(k["name"] for k in out["kernels"]) == ["a", "b"]


def test_extract_forwards_log_level():
    with _install(_trace()) as ctor:
        extract_kernel_code(["true"], log_level=2)
    assert ctor.call_args.kwargs["log_level"] == 2


def test_extract_passes_command_through():
    trace = _trace()
    with _install(trace):
        extract_kernel_code(["python", "-c", "pass"])
    assert FakeNexus.last.commands[0] == ["python", "-c", "pass"]


# --------------------------------------------------------------------------
# list_kernels
# --------------------------------------------------------------------------


def test_list_reports_total_and_names():
    with _install(_trace(a={"assembly": ["x"]}, b={})):
        out = list_kernels(["true"])
    assert out["total_kernels"] == 2
    assert sorted(k["name"] for k in out["kernels"]) == ["a", "b"]


def test_list_flags_source_and_assembly_presence():
    with _install(_trace(withsrc={"hip": ["s"], "assembly": ["x"]}, bare={})):
        out = list_kernels(["true"])
    by_name = {k["name"]: k for k in out["kernels"]}
    assert by_name["withsrc"]["has_source"] is True
    assert by_name["withsrc"]["has_assembly"] is True
    assert by_name["bare"]["has_source"] is False
    assert by_name["bare"]["has_assembly"] is False


def test_list_empty_trace():
    with _install(_trace()):
        out = list_kernels(["true"])
    assert out == {"total_kernels": 0, "kernels": []}


def test_list_uses_quiet_log_level():
    # Listing should not emit tracer chatter into the MCP channel.
    with _install(_trace()) as ctor:
        list_kernels(["true"])
    assert ctor.call_args.kwargs["log_level"] == 0


# --------------------------------------------------------------------------
# main()
# --------------------------------------------------------------------------


def test_main_defaults_to_stdio():
    with patch("sys.argv", ["nexus-mcp"]), patch.object(mcp_server.mcp, "run") as run:
        mcp_server.main()
    run.assert_called_once_with(transport="stdio")


def test_main_explicit_stdio():
    with (
        patch("sys.argv", ["nexus-mcp", "--transport", "stdio"]),
        patch.object(mcp_server.mcp, "run") as run,
    ):
        mcp_server.main()
    run.assert_called_once_with(transport="stdio")


def test_main_http_uses_documented_defaults():
    with (
        patch("sys.argv", ["nexus-mcp", "--transport", "http"]),
        patch.object(mcp_server.mcp, "run") as run,
    ):
        mcp_server.main()
    run.assert_called_once_with(
        transport="streamable-http", host="127.0.0.1", port=8000, path="/nexus"
    )


def test_main_http_honours_overrides():
    argv = [
        "nexus-mcp",
        "--transport",
        "http",
        "--host",
        "0.0.0.0",
        "--port",
        "9100",
        "--path",
        "/custom",
    ]
    with patch("sys.argv", argv), patch.object(mcp_server.mcp, "run") as run:
        mcp_server.main()
    run.assert_called_once_with(
        transport="streamable-http", host="0.0.0.0", port=9100, path="/custom"
    )


def test_main_port_is_parsed_as_int():
    with (
        patch("sys.argv", ["nexus-mcp", "--transport", "http", "--port", "1234"]),
        patch.object(mcp_server.mcp, "run") as run,
    ):
        mcp_server.main()
    assert run.call_args.kwargs["port"] == 1234


def test_main_rejects_unknown_transport():
    with (
        patch("sys.argv", ["nexus-mcp", "--transport", "carrier-pigeon"]),
        pytest.raises(SystemExit) as exc,
    ):
        mcp_server.main()
    assert exc.value.code == 2


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def test_both_tools_are_registered():
    """Fail loudly rather than skip if the fastmcp listing API moves.

    Driven with ``asyncio.run`` so it needs no pytest async plugin; a
    permanently-skipped test contributes nothing while looking green.
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
    names = set(tools) if isinstance(tools, dict) else {getattr(t, "name", t) for t in tools}
    assert {"extract_kernel_code", "list_kernels"} <= names


def test_kernel_objects_reach_the_tools_intact():
    """Guard the Trace -> tool boundary with a real Kernel, not a stub."""
    trace = _trace(vector_add={"assembly": ["a"], "hip": ["src"]})
    assert isinstance(next(iter(trace)), Kernel)
    with _install(trace):
        out = list_kernels(["true"])
    assert out["kernels"][0]["has_source"] is True
