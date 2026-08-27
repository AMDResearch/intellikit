# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the rocminfo MCP server tools.

These exercise the MCP tool layer without ROCm or a GPU by substituting a
fake ``Rocminfo`` over the module-level singleton.  The fixtures build real
``AgentInfo``/``RocminfoResult`` objects rather than mocks, so a field rename
in ``rocm_mcp.sysinfo.rocminfo`` breaks these tests instead of silently
passing.

Every tool is covered on three paths: the success path, the empty path (no
GPUs / no agents at all), and the failure path, where the collaborator raises
and the tool has to both return the error string and report it through
``ctx.error``.
"""

import asyncio
import inspect
from unittest.mock import patch

import pytest

from rocm_mcp.sysinfo import AgentInfo, DeviceType, RocminfoResult, rocminfo_mcp


def _tool(obj):  # noqa: ANN001, ANN202
    """Return the underlying function whether or not FastMCP wrapped it."""
    return getattr(obj, "fn", obj)


get_gpu_architecture = _tool(rocminfo_mcp.get_gpu_architecture)
get_all_agents = _tool(rocminfo_mcp.get_all_agents)


class FakeContext:
    """Minimal stand-in for the FastMCP Context that records log calls."""

    def __init__(self) -> None:
        """Start with empty info and error logs."""
        self.infos: list[str] = []
        self.errors: list[str] = []

    async def info(self, message: str) -> None:
        """Record an informational message."""
        self.infos.append(message)

    async def error(self, message: str) -> None:
        """Record an error message."""
        self.errors.append(message)


class FakeRocminfo:
    """Stand-in for Rocminfo that serves canned agents or raises."""

    def __init__(
        self, result: RocminfoResult | None = None, error: Exception | None = None
    ) -> None:
        """Serve ``result`` from get_agents(), or raise ``error`` if given."""
        self.result = result
        self.error = error
        self.calls = 0

    def get_agents(self) -> RocminfoResult:
        """Return the canned result, or raise the canned error."""
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def _agent(
    number: int,
    name: str,
    device_type: DeviceType,
    *,
    marketing_name: str = "AMD Device",
    vendor_name: str = "AMD",
    uuid: str = "GPU-XX",
    compute_units: int | None = 304,
    max_clock_freq: int | None = 2100,
    profile: str | None = "BASE_PROFILE",
) -> AgentInfo:
    """Build a real AgentInfo so a field rename fails the test."""
    return AgentInfo(
        agent_number=number,
        name=name,
        uuid=uuid,
        marketing_name=marketing_name,
        vendor_name=vendor_name,
        device_type=device_type,
        compute_units=compute_units,
        max_clock_freq=max_clock_freq,
        profile=profile,
    )


_CPU = _agent(1, "AMD EPYC 9654", DeviceType.CPU, marketing_name="AMD EPYC 9654", uuid="CPU-XX")
_GPU0 = _agent(2, "gfx942", DeviceType.GPU, marketing_name="AMD Instinct MI300X")
_GPU1 = _agent(3, "gfx942", DeviceType.GPU, marketing_name="AMD Instinct MI300X")


def _result(*agents: AgentInfo) -> RocminfoResult:
    """Wrap agents in a real RocminfoResult."""
    return RocminfoResult(agents=list(agents), raw_output="")


def _call(tool, fake: FakeRocminfo) -> tuple[str, FakeContext]:  # noqa: ANN001
    """Run an async tool against a patched rocminfo singleton."""
    ctx = FakeContext()
    with patch.object(rocminfo_mcp, "rocminfo", fake):
        return asyncio.run(tool(ctx)), ctx


class TestGetGpuArchitecture:
    """Cover the get_gpu_architecture tool."""

    def test_lists_only_gpu_agents(self) -> None:
        """CPU agents are filtered out and GPUs are renumbered from zero."""
        out, ctx = _call(get_gpu_architecture, FakeRocminfo(_result(_CPU, _GPU0, _GPU1)))
        assert "GPU 0:" in out
        assert "GPU 1:" in out
        assert "AMD EPYC 9654" not in out
        assert ctx.errors == []

    def test_reports_architecture_fields(self) -> None:
        """Architecture, marketing name, vendor, CUs and clock are all rendered."""
        out, _ = _call(get_gpu_architecture, FakeRocminfo(_result(_GPU0)))
        assert "  Architecture: gfx942" in out
        assert "  Marketing Name: AMD Instinct MI300X" in out
        assert "  Vendor: AMD" in out
        assert "  Compute Units: 304" in out
        assert "  Max Clock Frequency: 2100 MHz" in out

    def test_optional_fields_omitted_when_unset(self) -> None:
        """Compute units and clock lines disappear when the parser found nothing."""
        gpu = _agent(2, "gfx90a", DeviceType.GPU, compute_units=None, max_clock_freq=None)
        out, _ = _call(get_gpu_architecture, FakeRocminfo(_result(gpu)))
        assert "Compute Units" not in out
        assert "Max Clock Frequency" not in out

    def test_no_gpus_among_agents(self) -> None:
        """A CPU-only system reports no GPUs rather than an empty string."""
        out, ctx = _call(get_gpu_architecture, FakeRocminfo(_result(_CPU)))
        assert out == "No GPUs found in the system."
        # The empty result is not an error, so nothing is reported to the client.
        assert ctx.errors == []

    def test_no_agents_at_all(self) -> None:
        """An entirely empty agent list takes the same no-GPU path."""
        out, _ = _call(get_gpu_architecture, FakeRocminfo(_result()))
        assert out == "No GPUs found in the system."

    def test_failure_is_returned_and_reported(self) -> None:
        """A raising collaborator yields the error string and calls ctx.error."""
        boom = RuntimeError("rocminfo execution failed")
        out, ctx = _call(get_gpu_architecture, FakeRocminfo(error=boom))
        assert out == "Failed to get GPU architecture: rocminfo execution failed"
        assert ctx.errors == [out]

    def test_missing_executable_is_reported(self) -> None:
        """FileNotFoundError from the wrapper is surfaced the same way."""
        out, ctx = _call(get_gpu_architecture, FakeRocminfo(error=FileNotFoundError("no rocminfo")))
        assert out.startswith("Failed to get GPU architecture:")
        assert ctx.errors == [out]


class TestGetAllAgents:
    """Cover the get_all_agents tool."""

    def test_renders_every_agent(self) -> None:
        """Both CPU and GPU agents appear, keyed by their own agent number."""
        out, ctx = _call(get_all_agents, FakeRocminfo(_result(_CPU, _GPU0)))
        assert "Agent 1:" in out
        assert "Agent 2:" in out
        assert ctx.errors == []

    def test_reports_agent_fields(self) -> None:
        """Name, device type value, marketing name, vendor, UUID and profile render."""
        out, _ = _call(get_all_agents, FakeRocminfo(_result(_GPU0)))
        assert "  Name: gfx942" in out
        assert "  Type: GPU" in out
        assert "  Marketing Name: AMD Instinct MI300X" in out
        assert "  Vendor: AMD" in out
        assert "  UUID: GPU-XX" in out
        assert "  Profile: BASE_PROFILE" in out
        assert "  Compute Units: 304" in out
        assert "  Max Clock Frequency: 2100 MHz" in out

    def test_device_type_uses_enum_value(self) -> None:
        """The rendered type comes from DeviceType.value, not the member name."""
        unknown = _agent(9, "mystery", DeviceType.Unknown)
        out, _ = _call(get_all_agents, FakeRocminfo(_result(unknown)))
        assert f"  Type: {DeviceType.Unknown.value}" in out

    def test_optional_fields_omitted_when_unset(self) -> None:
        """Profile, compute units and clock lines are skipped when falsy."""
        bare = _agent(
            4,
            "gfx1100",
            DeviceType.GPU,
            compute_units=None,
            max_clock_freq=None,
            profile=None,
        )
        out, _ = _call(get_all_agents, FakeRocminfo(_result(bare)))
        assert "Profile" not in out
        assert "Compute Units" not in out
        assert "Max Clock Frequency" not in out

    def test_no_agents(self) -> None:
        """An empty agent list reports that, and is not treated as an error."""
        out, ctx = _call(get_all_agents, FakeRocminfo(_result()))
        assert out == "No HSA agents found in the system."
        assert ctx.errors == []

    def test_failure_is_returned_and_reported(self) -> None:
        """A raising collaborator yields the error string and calls ctx.error."""
        out, ctx = _call(get_all_agents, FakeRocminfo(error=RuntimeError("boom")))
        assert out == "Failed to get agent information: boom"
        assert ctx.errors == [out]


class TestMain:
    """Cover the transport dispatch in main()."""

    def test_default_transport_is_stdio(self) -> None:
        """With no arguments the server runs over stdio."""
        with patch.object(rocminfo_mcp.mcp, "run") as run, patch("sys.argv", ["rocminfo-mcp"]):
            rocminfo_mcp.main()
        run.assert_called_once_with(transport="stdio")

    def test_explicit_stdio(self) -> None:
        """An explicit --transport stdio behaves like the default."""
        with (
            patch.object(rocminfo_mcp.mcp, "run") as run,
            patch("sys.argv", ["rocminfo-mcp", "--transport", "stdio"]),
        ):
            rocminfo_mcp.main()
        run.assert_called_once_with(transport="stdio")

    def test_http_transport_uses_defaults(self) -> None:
        """HTTP falls back to loopback, port 8000 and the rocminfo path."""
        with (
            patch.object(rocminfo_mcp.mcp, "run") as run,
            patch("sys.argv", ["rocminfo-mcp", "--transport", "http"]),
        ):
            rocminfo_mcp.main()
        run.assert_called_once_with(
            transport="streamable-http", host="127.0.0.1", port=8000, path="/rocm_mcp/rocminfo"
        )

    def test_http_transport_honours_overrides(self) -> None:
        """Host, port and path overrides reach mcp.run()."""
        argv = [
            "rocminfo-mcp",
            "--transport",
            "http",
            "--host",
            "0.0.0.0",  # noqa: S104
            "--port",
            "9100",
            "--path",
            "/custom",
        ]
        with patch.object(rocminfo_mcp.mcp, "run") as run, patch("sys.argv", argv):
            rocminfo_mcp.main()
        run.assert_called_once_with(
            transport="streamable-http",
            host="0.0.0.0",  # noqa: S104
            port=9100,
            path="/custom",
        )

    def test_invalid_transport_rejected(self) -> None:
        """A transport outside the declared choices is rejected by argparse."""
        with (
            patch("sys.argv", ["rocminfo-mcp", "--transport", "carrier-pigeon"]),
            pytest.raises(SystemExit),
        ):
            rocminfo_mcp.main()


class TestToolRegistration:
    """Assert the tools are actually registered with the real FastMCP server."""

    def test_both_tools_registered_with_mcp(self) -> None:
        """Both tool names show up in the server's tool listing."""
        assert {"get_gpu_architecture", "get_all_agents"} <= _registered_tools()

    def test_server_name(self) -> None:
        """The server advertises itself as rocminfo."""
        assert rocminfo_mcp.mcp.name == "rocminfo"


def _registered_tools() -> set[str]:
    """Fetch registered tool names across FastMCP versions."""
    for attr in ("list_tools", "get_tools"):
        getter = getattr(rocminfo_mcp.mcp, attr, None)
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
