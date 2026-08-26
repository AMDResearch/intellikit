# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Unit tests for the Linex MCP server tools.

These exercise the MCP tool layer without a GPU by substituting a fake
profiler.  The fixtures build real ``SourceLine``/``InstructionData``
objects rather than mocks, so a field rename in ``linex.api`` breaks
these tests instead of silently passing.
"""

from unittest.mock import patch

import pytest

from linex import InstructionData, SourceLine
from linex.mcp import server as mcp_server


def _tool(obj):
    """Return the underlying function whether or not FastMCP wrapped it."""
    return getattr(obj, "fn", obj)


profile_application = _tool(mcp_server.profile_application)
analyze_instruction_hotspots = _tool(mcp_server.analyze_instruction_hotspots)


def _inst(isa, latency, stall, idle=0, count=1, index=0, addr=0x1000):
    return InstructionData(
        isa=isa,
        instruction_index=index,
        source_location="kernel.hip:42",
        code_object_id=1,
        instruction_address=addr,
        execution_count=count,
        latency_cycles=latency,
        stall_cycles=stall,
        idle_cycles=idle,
    )


def _line(line_number, total_cycles, stall_cycles, instructions, idle=0, count=1):
    return SourceLine(
        file="kernel.hip",
        line_number=line_number,
        source_location=f"kernel.hip:{line_number}",
        execution_count=count,
        total_cycles=total_cycles,
        stall_cycles=stall_cycles,
        idle_cycles=idle,
        instructions=instructions,
    )


class FakeLinex:
    """Stand-in for Linex that records profile() calls and serves canned data."""

    calls = []

    def __init__(self, *args, **kwargs):
        self.source_lines = list(_FAKE_LINES)
        self.instructions = [i for line in _FAKE_LINES for i in line.instructions]

    def profile(self, command, kernel_filter=None, **kwargs):
        FakeLinex.calls.append({"command": command, "kernel_filter": kernel_filter})


_FAKE_LINES = [
    _line(
        10,
        1000,
        400,
        [
            _inst("v_mov_b32 v0, v1", latency=100, stall=10, addr=0x1000),
            _inst("global_load_dwordx4 v[0:3], v[4:5]", latency=800, stall=380, addr=0x1004),
            _inst("s_waitcnt vmcnt(0)", latency=100, stall=10, addr=0x1008),
        ],
    ),
    _line(20, 500, 100, [_inst("v_add_f32 v2, v0, v1", latency=500, stall=100, addr=0x2000)]),
    _line(30, 250, 0, [_inst("s_endpgm", latency=250, stall=0, addr=0x3000)]),
]


@pytest.fixture(autouse=True)
def _fake_profiler():
    FakeLinex.calls = []
    with patch.object(mcp_server, "Linex", FakeLinex):
        yield


class TestProfileApplication:
    def test_returns_totals_and_hotspots(self):
        result = profile_application("./app")
        assert result["total_source_lines"] == 3
        assert result["total_instructions"] == 5
        assert len(result["hotspots"]) == 3

    def test_hotspot_fields_match_source_lines(self):
        hot = profile_application("./app")["hotspots"][0]
        assert hot["rank"] == 1
        assert hot["file"] == "kernel.hip"
        assert hot["line_number"] == 10
        assert hot["source_location"] == "kernel.hip:10"
        assert hot["total_cycles"] == 1000
        assert hot["stall_cycles"] == 400
        assert hot["stall_percent"] == 40.0
        assert hot["idle_cycles"] == 0
        assert hot["execution_count"] == 1
        assert hot["num_instructions"] == 3

    def test_ranks_are_sequential(self):
        hotspots = profile_application("./app")["hotspots"]
        assert [h["rank"] for h in hotspots] == [1, 2, 3]

    def test_top_n_truncates(self):
        result = profile_application("./app", top_n=2)
        assert len(result["hotspots"]) == 2
        # totals still reflect everything profiled, not just the truncated view
        assert result["total_source_lines"] == 3

    def test_top_n_larger_than_available(self):
        assert len(profile_application("./app", top_n=99)["hotspots"]) == 3

    def test_stall_percent_is_rounded(self):
        # 400/1000 -> 40.0 exactly; verify it is a float rounded to 2dp
        hot = profile_application("./app")["hotspots"][0]
        assert isinstance(hot["stall_percent"], float)
        assert hot["stall_percent"] == round(hot["stall_percent"], 2)

    def test_zero_cycle_line_does_not_divide_by_zero(self):
        with patch.object(FakeLinex, "__init__", _init_with([_line(1, 0, 0, [])])):
            assert profile_application("./app")["hotspots"][0]["stall_percent"] == 0.0

    def test_command_and_filter_forwarded(self):
        profile_application("python train.py", kernel_filter="gemm.*")
        assert FakeLinex.calls == [{"command": "python train.py", "kernel_filter": "gemm.*"}]

    def test_kernel_filter_defaults_to_none(self):
        profile_application("./app")
        assert FakeLinex.calls[0]["kernel_filter"] is None

    def test_empty_profile(self):
        with patch.object(FakeLinex, "__init__", _init_with([])):
            result = profile_application("./app")
        assert result["total_source_lines"] == 0
        assert result["total_instructions"] == 0
        assert result["hotspots"] == []


class TestAnalyzeInstructionHotspots:
    def test_returns_analysis_per_line(self):
        result = analyze_instruction_hotspots("./app")
        assert len(result["hotspot_analysis"]) == 3

    def test_instructions_sorted_by_latency_desc(self):
        first = analyze_instruction_hotspots("./app")["hotspot_analysis"][0]
        latencies = [i["latency_cycles"] for i in first["instructions"]]
        assert latencies == sorted(latencies, reverse=True)
        assert latencies[0] == 800

    def test_instruction_fields(self):
        inst = analyze_instruction_hotspots("./app")["hotspot_analysis"][0]["instructions"][0]
        assert inst["isa"] == "global_load_dwordx4 v[0:3], v[4:5]"
        assert inst["latency_cycles"] == 800
        assert inst["stall_cycles"] == 380
        assert inst["stall_percent"] == 47.5
        assert inst["idle_cycles"] == 0
        assert inst["execution_count"] == 1
        assert inst["instruction_address"] == "0x00001004"

    def test_instruction_address_is_zero_padded_hex(self):
        for line in analyze_instruction_hotspots("./app")["hotspot_analysis"]:
            for inst in line["instructions"]:
                assert inst["instruction_address"].startswith("0x")
                assert len(inst["instruction_address"]) == 10

    def test_line_level_fields(self):
        line = analyze_instruction_hotspots("./app")["hotspot_analysis"][0]
        assert line["source_location"] == "kernel.hip:10"
        assert line["total_cycles"] == 1000
        assert line["stall_percent"] == 40.0

    def test_top_lines_truncates(self):
        result = analyze_instruction_hotspots("./app", top_lines=1)
        assert len(result["hotspot_analysis"]) == 1

    def test_top_instructions_per_line_truncates(self):
        result = analyze_instruction_hotspots("./app", top_instructions_per_line=2)
        assert len(result["hotspot_analysis"][0]["instructions"]) == 2

    def test_command_and_filter_forwarded(self):
        analyze_instruction_hotspots("./app", kernel_filter="attention")
        assert FakeLinex.calls == [{"command": "./app", "kernel_filter": "attention"}]

    def test_line_with_no_instructions(self):
        with patch.object(FakeLinex, "__init__", _init_with([_line(1, 100, 50, [])])):
            result = analyze_instruction_hotspots("./app")
        assert result["hotspot_analysis"][0]["instructions"] == []

    def test_empty_profile(self):
        with patch.object(FakeLinex, "__init__", _init_with([])):
            assert analyze_instruction_hotspots("./app") == {"hotspot_analysis": []}


class TestMain:
    def test_default_transport_is_stdio(self):
        with patch.object(mcp_server.mcp, "run") as run, patch("sys.argv", ["linex-mcp"]):
            mcp_server.main()
        run.assert_called_once_with(transport="stdio")

    def test_explicit_stdio(self):
        with (
            patch.object(mcp_server.mcp, "run") as run,
            patch("sys.argv", ["linex-mcp", "--transport", "stdio"]),
        ):
            mcp_server.main()
        run.assert_called_once_with(transport="stdio")

    def test_http_transport_uses_defaults(self):
        with (
            patch.object(mcp_server.mcp, "run") as run,
            patch("sys.argv", ["linex-mcp", "--transport", "http"]),
        ):
            mcp_server.main()
        run.assert_called_once_with(
            transport="streamable-http", host="127.0.0.1", port=8000, path="/linex"
        )

    def test_http_transport_honours_overrides(self):
        argv = [
            "linex-mcp",
            "--transport",
            "http",
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--path",
            "/custom",
        ]
        with patch.object(mcp_server.mcp, "run") as run, patch("sys.argv", argv):
            mcp_server.main()
        run.assert_called_once_with(
            transport="streamable-http", host="0.0.0.0", port=9001, path="/custom"
        )

    def test_invalid_transport_rejected(self):
        with patch("sys.argv", ["linex-mcp", "--transport", "carrier-pigeon"]):
            with pytest.raises(SystemExit):
                mcp_server.main()


class TestToolRegistration:
    def test_both_tools_registered_with_mcp(self):
        names = _registered_tools()
        assert {"profile_application", "analyze_instruction_hotspots"} <= names

    def test_server_name(self):
        assert mcp_server.mcp.name == "IntelliKit Linex"


def _registered_tools():
    """Fetch registered tool names across FastMCP versions."""
    import asyncio
    import inspect

    for attr in ("list_tools", "get_tools"):
        getter = getattr(mcp_server.mcp, attr, None)
        if getter is not None:
            break
    else:
        pytest.fail("FastMCP exposes neither list_tools() nor get_tools()")
    result = getter()
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    tools = result.values() if hasattr(result, "values") else result
    names = set()
    for t in tools:
        name = getattr(t, "name", None)
        names.add(name if name is not None else str(t))
    return names


def _init_with(lines):
    """Build a FakeLinex.__init__ that serves the given source lines."""

    def _init(self, *args, **kwargs):
        self.source_lines = list(lines)
        self.instructions = [i for line in lines for i in line.instructions]

    return _init
