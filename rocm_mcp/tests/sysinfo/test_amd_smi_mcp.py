# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the amd-smi MCP server tools.

These exercise the MCP tool layer without a GPU by substituting a fake
``AmdSmi`` over the module-level singleton.  The fixtures build the real
``GpuStaticInfo``/``GpuMetrics``/``GpuFirmwareInfo``/... dataclasses rather
than mocks, so a field rename in ``rocm_mcp.sysinfo.amd_smi`` breaks these
tests instead of silently passing.

Every tool is covered on the success path, the empty path (no GPUs came
back), and the failure path, where the collaborator raises and the tool has
to both return the error string and report it through ``ctx.error``.  The
tools that take a ``gpu_index`` additionally have an ``IndexError`` path that
is deliberately *not* reported through ``ctx.error`` -- an out-of-range index
is a caller mistake, not a server fault -- and that distinction is asserted.
"""

import asyncio
import inspect
from unittest.mock import patch

import pytest

from rocm_mcp.sysinfo import (
    DriverInformationResult,
    FirmwareEntry,
    GpuBadPageInfo,
    GpuFirmwareInfo,
    GpuInfo,
    GpuMetrics,
    GpuProcessInfo,
    GpuStaticInfo,
    ProcessEntry,
    amd_smi_mcp,
)


def _tool(obj):  # noqa: ANN001, ANN202
    """Return the underlying function whether or not FastMCP wrapped it."""
    return getattr(obj, "fn", obj)


get_driver_information = _tool(amd_smi_mcp.get_driver_information)
list_gpus = _tool(amd_smi_mcp.list_gpus)
get_gpu_static_info = _tool(amd_smi_mcp.get_gpu_static_info)
get_gpu_metrics = _tool(amd_smi_mcp.get_gpu_metrics)
get_gpu_firmware_info = _tool(amd_smi_mcp.get_gpu_firmware_info)
get_gpu_process_info = _tool(amd_smi_mcp.get_gpu_process_info)
get_gpu_bad_pages = _tool(amd_smi_mcp.get_gpu_bad_pages)


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


class FakeAmdSmi:
    """Stand-in for AmdSmi that serves one canned result or raises."""

    def __init__(self, result: object = None, error: Exception | None = None) -> None:
        """Serve ``result`` from every query, or raise ``error`` if given."""
        self.result = result
        self.error = error
        self.calls: list[int | None] = []

    def _serve(self, gpu_index: int | None = None) -> object:
        """Record the requested index and serve the canned result or error."""
        self.calls.append(gpu_index)
        if self.error is not None:
            raise self.error
        return self.result

    def get_driver_information(self) -> object:
        """Serve the canned driver information."""
        return self._serve()

    def list_gpus(self) -> object:
        """Serve the canned GPU list."""
        return self._serve()

    def get_gpu_static_info(self, gpu_index: int | None = None) -> object:
        """Serve the canned static info."""
        return self._serve(gpu_index)

    def get_gpu_metrics(self, gpu_index: int | None = None) -> object:
        """Serve the canned metrics."""
        return self._serve(gpu_index)

    def get_gpu_firmware_info(self, gpu_index: int | None = None) -> object:
        """Serve the canned firmware info."""
        return self._serve(gpu_index)

    def get_gpu_process_info(self, gpu_index: int | None = None) -> object:
        """Serve the canned process info."""
        return self._serve(gpu_index)

    def get_gpu_bad_page_info(self, gpu_index: int | None = None) -> object:
        """Serve the canned bad page info."""
        return self._serve(gpu_index)


def _call(tool, fake: FakeAmdSmi, **kwargs) -> tuple[str, FakeContext]:  # noqa: ANN001
    """Run an async tool against a patched amd_smi singleton."""
    ctx = FakeContext()
    with patch.object(amd_smi_mcp, "amd_smi", fake):
        return asyncio.run(tool(ctx, **kwargs)), ctx


def _static(index: int = 0) -> GpuStaticInfo:
    """Build a real GpuStaticInfo so a field rename fails the test."""
    return GpuStaticInfo(
        gpu_index=index,
        bdf="0000:03:00.0",
        market_name="Instinct MI300X",
        vendor_id="0x1002",
        device_id="0x74a0",
        rev_id="0x00",
        asic_serial="0x1234567890ABCDEF",
        num_compute_units=304,
        vram_type="HBM3",
        vram_vendor="SK Hynix",
        vram_size_mb=196608,
        model_name="MI300X OAM",
        power_cap_w=750,
        default_power_cap_w=750,
        min_power_cap_w=0,
        max_power_cap_w=760,
    )


def _metrics(index: int = 0) -> GpuMetrics:
    """Build a real GpuMetrics so a field rename fails the test."""
    return GpuMetrics(
        gpu_index=index,
        bdf="0000:03:00.0",
        gfx_activity_pct=85,
        umc_activity_pct=42,
        mm_activity_pct=10,
        temp_edge_c=55,
        temp_hotspot_c=72,
        temp_vram_c=48,
        current_power_w=320,
        gfx_voltage_mv=900,
        soc_voltage_mv=850,
        mem_voltage_mv=1200,
        gfx_clock_mhz=2100,
        gfx_max_clock_mhz=2400,
        mem_clock_mhz=1600,
        mem_max_clock_mhz=2000,
        vram_used_mb=65536,
        vram_total_mb=196608,
    )


def _firmware(index: int = 0, entries: list[FirmwareEntry] | None = None) -> GpuFirmwareInfo:
    """Build a real GpuFirmwareInfo so a field rename fails the test."""
    return GpuFirmwareInfo(
        gpu_index=index,
        bdf="0000:03:00.0",
        vbios_name="MI300X VBIOS",
        vbios_version="022.040.000.001",
        vbios_build_date="2024/01/01",
        vbios_part_number="113-MI300X",
        firmware_list=[FirmwareEntry(name="SMU", version="85.101.0")]
        if entries is None
        else entries,
    )


def _processes(index: int = 0, procs: list[ProcessEntry] | None = None) -> GpuProcessInfo:
    """Build a real GpuProcessInfo so a field rename fails the test."""
    return GpuProcessInfo(
        gpu_index=index,
        bdf="0000:03:00.0",
        processes=[ProcessEntry(pid=4242, name="python3", vram_usage_mb=512)]
        if procs is None
        else procs,
    )


def _bad_pages(index: int = 0) -> GpuBadPageInfo:
    """Build a real GpuBadPageInfo so a field rename fails the test."""
    return GpuBadPageInfo(
        gpu_index=index,
        bdf="0000:03:00.0",
        correctable_ecc_count=3,
        uncorrectable_ecc_count=1,
        deferred_ecc_count=0,
        bad_page_count=2,
    )


_OUT_OF_RANGE = IndexError("GPU index 9 out of range (0-0)")


class TestGetDriverInformation:
    """Cover the get_driver_information tool."""

    def test_renders_all_driver_fields(self) -> None:
        """Name, version and date come from the real DriverInformationResult."""
        driver = DriverInformationResult(version="6.16.13", name="amdgpu", date="2025/01/01 00:00")
        out, ctx = _call(get_driver_information, FakeAmdSmi(driver))
        assert out == (
            "Driver Name: amdgpu\nDriver Version: 6.16.13\nDriver Date: 2025/01/01 00:00"
        )
        assert ctx.errors == []

    def test_failure_is_returned_and_reported(self) -> None:
        """A raising collaborator yields the error string and calls ctx.error."""
        out, ctx = _call(get_driver_information, FakeAmdSmi(error=RuntimeError("no driver")))
        assert out == "Failed to get driver version: no driver"
        assert ctx.errors == [out]


class TestListGpus:
    """Cover the list_gpus tool."""

    def test_lists_every_gpu(self) -> None:
        """Index, BDF and UUID are rendered for each GPU."""
        gpus = [
            GpuInfo(gpu_index=0, bdf="0000:03:00.0", uuid="GPU-aaaa"),
            GpuInfo(gpu_index=1, bdf="0000:04:00.0", uuid="GPU-bbbb"),
        ]
        out, ctx = _call(list_gpus, FakeAmdSmi(gpus))
        assert "GPU 0:\n  BDF: 0000:03:00.0\n  UUID: GPU-aaaa" in out
        assert "GPU 1:\n  BDF: 0000:04:00.0\n  UUID: GPU-bbbb" in out
        assert ctx.errors == []

    def test_no_gpus(self) -> None:
        """An empty GPU list renders as an empty string, not an error."""
        out, ctx = _call(list_gpus, FakeAmdSmi([]))
        assert out == ""
        assert ctx.errors == []

    def test_failure_is_returned_and_reported(self) -> None:
        """A raising collaborator yields the error string and calls ctx.error."""
        out, ctx = _call(list_gpus, FakeAmdSmi(error=RuntimeError("no processors")))
        assert out == "Failed to list GPUs: no processors"
        assert ctx.errors == [out]


class TestGetGpuStaticInfo:
    """Cover the get_gpu_static_info tool."""

    def test_renders_static_fields(self) -> None:
        """Market name, IDs, CUs, VRAM, model and power caps all render."""
        out, ctx = _call(get_gpu_static_info, FakeAmdSmi([_static()]))
        assert out.startswith("GPU 0 (0000:03:00.0):")
        assert "  Market Name: Instinct MI300X" in out
        assert "  Vendor/Device/Rev ID: 0x1002/0x74a0/0x00" in out
        assert "  ASIC Serial: 0x1234567890ABCDEF" in out
        assert "  Compute Units: 304" in out
        assert "  VRAM: 196608 MB HBM3 (SK Hynix)" in out
        assert "  Model: MI300X OAM" in out
        assert "  Power Cap: 750W (default 750W, range 0-760W)" in out
        assert ctx.errors == []

    def test_all_gpus_by_default(self) -> None:
        """With no index the tool asks for every GPU and renders them all."""
        fake = FakeAmdSmi([_static(0), _static(1)])
        out, _ = _call(get_gpu_static_info, fake)
        assert fake.calls == [None]
        assert "GPU 0 (" in out
        assert "GPU 1 (" in out

    def test_gpu_index_is_forwarded(self) -> None:
        """An explicit index reaches the wrapper unchanged."""
        fake = FakeAmdSmi([_static(1)])
        _call(get_gpu_static_info, fake, gpu_index=1)
        assert fake.calls == [1]

    def test_no_gpus(self) -> None:
        """An empty result renders as an empty string, not an error."""
        out, ctx = _call(get_gpu_static_info, FakeAmdSmi([]))
        assert out == ""
        assert ctx.errors == []

    def test_index_error_is_returned_without_ctx_error(self) -> None:
        """An out-of-range index is a caller mistake, so it is not reported."""
        out, ctx = _call(get_gpu_static_info, FakeAmdSmi(error=_OUT_OF_RANGE), gpu_index=9)
        assert out == "Error: GPU index 9 out of range (0-0)"
        assert ctx.errors == []

    def test_failure_is_returned_and_reported(self) -> None:
        """A raising collaborator yields the error string and calls ctx.error."""
        out, ctx = _call(get_gpu_static_info, FakeAmdSmi(error=RuntimeError("boom")))
        assert out == "Failed to get GPU static info: boom"
        assert ctx.errors == [out]


class TestGetGpuMetrics:
    """Cover the get_gpu_metrics tool."""

    def test_renders_metric_fields(self) -> None:
        """Activity, temperature, power, voltage, clocks and VRAM all render."""
        out, ctx = _call(get_gpu_metrics, FakeAmdSmi([_metrics()]))
        assert out.startswith("GPU 0 (0000:03:00.0):")
        assert "  Activity: GFX 85%, UMC 42%, MM 10%" in out
        assert "  Temperature: Edge 55C, Hotspot 72C, VRAM 48C" in out
        assert "  Power: 320W" in out
        assert "  Voltage: GFX 900mV, SoC 850mV, Mem 1200mV" in out
        assert "  Clocks: GFX 2100/2400 MHz, MEM 1600/2000 MHz" in out
        assert "  VRAM: 65536/196608 MB" in out
        assert ctx.errors == []

    def test_gpu_index_is_forwarded(self) -> None:
        """An explicit index reaches the wrapper unchanged."""
        fake = FakeAmdSmi([_metrics(2)])
        _call(get_gpu_metrics, fake, gpu_index=2)
        assert fake.calls == [2]

    def test_no_gpus(self) -> None:
        """An empty result renders as an empty string, not an error."""
        out, ctx = _call(get_gpu_metrics, FakeAmdSmi([]))
        assert out == ""
        assert ctx.errors == []

    def test_index_error_is_returned_without_ctx_error(self) -> None:
        """An out-of-range index is a caller mistake, so it is not reported."""
        out, ctx = _call(get_gpu_metrics, FakeAmdSmi(error=_OUT_OF_RANGE), gpu_index=9)
        assert out == "Error: GPU index 9 out of range (0-0)"
        assert ctx.errors == []

    def test_failure_is_returned_and_reported(self) -> None:
        """A raising collaborator yields the error string and calls ctx.error."""
        out, ctx = _call(get_gpu_metrics, FakeAmdSmi(error=RuntimeError("boom")))
        assert out == "Failed to get GPU metrics: boom"
        assert ctx.errors == [out]


class TestGetGpuFirmwareInfo:
    """Cover the get_gpu_firmware_info tool."""

    def test_renders_vbios_and_firmware_entries(self) -> None:
        """The VBIOS line and every firmware entry are rendered."""
        entries = [
            FirmwareEntry(name="SMU", version="85.101.0"),
            FirmwareEntry(name="VCN", version="0.0.6"),
        ]
        out, ctx = _call(get_gpu_firmware_info, FakeAmdSmi([_firmware(entries=entries)]))
        assert out.startswith("GPU 0 (0000:03:00.0):")
        assert "  VBIOS: MI300X VBIOS v022.040.000.001 (2024/01/01, 113-MI300X)" in out
        assert "    SMU: 85.101.0" in out
        assert "    VCN: 0.0.6" in out
        assert ctx.errors == []

    def test_empty_firmware_list_renders_placeholder(self) -> None:
        """A GPU with no firmware entries shows the (none) placeholder."""
        out, _ = _call(get_gpu_firmware_info, FakeAmdSmi([_firmware(entries=[])]))
        assert "  Firmware:\n    (none)" in out

    def test_gpu_index_is_forwarded(self) -> None:
        """An explicit index reaches the wrapper unchanged."""
        fake = FakeAmdSmi([_firmware(3)])
        _call(get_gpu_firmware_info, fake, gpu_index=3)
        assert fake.calls == [3]

    def test_no_gpus(self) -> None:
        """An empty result renders as an empty string, not an error."""
        out, ctx = _call(get_gpu_firmware_info, FakeAmdSmi([]))
        assert out == ""
        assert ctx.errors == []

    def test_index_error_is_returned_without_ctx_error(self) -> None:
        """An out-of-range index is a caller mistake, so it is not reported."""
        out, ctx = _call(get_gpu_firmware_info, FakeAmdSmi(error=_OUT_OF_RANGE), gpu_index=9)
        assert out == "Error: GPU index 9 out of range (0-0)"
        assert ctx.errors == []

    def test_failure_is_returned_and_reported(self) -> None:
        """A raising collaborator yields the error string and calls ctx.error."""
        out, ctx = _call(get_gpu_firmware_info, FakeAmdSmi(error=RuntimeError("boom")))
        assert out == "Failed to get GPU firmware info: boom"
        assert ctx.errors == [out]


class TestGetGpuProcessInfo:
    """Cover the get_gpu_process_info tool."""

    def test_renders_every_process(self) -> None:
        """PID, name and VRAM usage are rendered for each process."""
        procs = [
            ProcessEntry(pid=4242, name="python3", vram_usage_mb=512),
            ProcessEntry(pid=99, name="rocprof", vram_usage_mb=8),
        ]
        out, ctx = _call(get_gpu_process_info, FakeAmdSmi([_processes(procs=procs)]))
        assert out.startswith("GPU 0 (0000:03:00.0):")
        assert "    PID 4242 (python3): 512 MB VRAM" in out
        assert "    PID 99 (rocprof): 8 MB VRAM" in out
        assert ctx.errors == []

    def test_idle_gpu_renders_placeholder(self) -> None:
        """A GPU with no processes shows the (no processes) placeholder."""
        out, _ = _call(get_gpu_process_info, FakeAmdSmi([_processes(procs=[])]))
        assert out == "GPU 0 (0000:03:00.0):\n    (no processes)"

    def test_gpu_index_is_forwarded(self) -> None:
        """An explicit index reaches the wrapper unchanged."""
        fake = FakeAmdSmi([_processes(1)])
        _call(get_gpu_process_info, fake, gpu_index=1)
        assert fake.calls == [1]

    def test_no_gpus(self) -> None:
        """An empty result renders as an empty string, not an error."""
        out, ctx = _call(get_gpu_process_info, FakeAmdSmi([]))
        assert out == ""
        assert ctx.errors == []

    def test_index_error_is_returned_without_ctx_error(self) -> None:
        """An out-of-range index is a caller mistake, so it is not reported."""
        out, ctx = _call(get_gpu_process_info, FakeAmdSmi(error=_OUT_OF_RANGE), gpu_index=9)
        assert out == "Error: GPU index 9 out of range (0-0)"
        assert ctx.errors == []

    def test_failure_is_returned_and_reported(self) -> None:
        """A raising collaborator yields the error string and calls ctx.error."""
        out, ctx = _call(get_gpu_process_info, FakeAmdSmi(error=RuntimeError("boom")))
        assert out == "Failed to get GPU process info: boom"
        assert ctx.errors == [out]


class TestGetGpuBadPages:
    """Cover the get_gpu_bad_pages tool."""

    def test_renders_ecc_counts_and_bad_pages(self) -> None:
        """Correctable, uncorrectable, deferred and retired page counts render."""
        out, ctx = _call(get_gpu_bad_pages, FakeAmdSmi([_bad_pages()]))
        assert out.startswith("GPU 0 (0000:03:00.0):")
        assert "  ECC Errors: 3 correctable, 1 uncorrectable, 0 deferred" in out
        assert "  Bad Pages: 2" in out
        assert ctx.errors == []

    def test_gpu_index_is_forwarded(self) -> None:
        """An explicit index reaches the wrapper unchanged."""
        fake = FakeAmdSmi([_bad_pages(0)])
        _call(get_gpu_bad_pages, fake, gpu_index=0)
        assert fake.calls == [0]

    def test_no_gpus(self) -> None:
        """An empty result renders as an empty string, not an error."""
        out, ctx = _call(get_gpu_bad_pages, FakeAmdSmi([]))
        assert out == ""
        assert ctx.errors == []

    def test_index_error_is_returned_without_ctx_error(self) -> None:
        """An out-of-range index is a caller mistake, so it is not reported."""
        out, ctx = _call(get_gpu_bad_pages, FakeAmdSmi(error=_OUT_OF_RANGE), gpu_index=9)
        assert out == "Error: GPU index 9 out of range (0-0)"
        assert ctx.errors == []

    def test_failure_is_returned_and_reported(self) -> None:
        """A raising collaborator yields the error string and calls ctx.error."""
        out, ctx = _call(get_gpu_bad_pages, FakeAmdSmi(error=RuntimeError("boom")))
        assert out == "Failed to get GPU bad page info: boom"
        assert ctx.errors == [out]


class TestMain:
    """Cover the transport dispatch in main()."""

    def test_default_transport_is_stdio(self) -> None:
        """With no arguments the server runs over stdio."""
        with patch.object(amd_smi_mcp.mcp, "run") as run, patch("sys.argv", ["amd-smi-mcp"]):
            amd_smi_mcp.main()
        run.assert_called_once_with(transport="stdio")

    def test_explicit_stdio(self) -> None:
        """An explicit --transport stdio behaves like the default."""
        with (
            patch.object(amd_smi_mcp.mcp, "run") as run,
            patch("sys.argv", ["amd-smi-mcp", "--transport", "stdio"]),
        ):
            amd_smi_mcp.main()
        run.assert_called_once_with(transport="stdio")

    def test_http_transport_uses_defaults(self) -> None:
        """HTTP falls back to loopback, port 8001 and the amd_smi path."""
        with (
            patch.object(amd_smi_mcp.mcp, "run") as run,
            patch("sys.argv", ["amd-smi-mcp", "--transport", "http"]),
        ):
            amd_smi_mcp.main()
        run.assert_called_once_with(
            transport="streamable-http", host="127.0.0.1", port=8001, path="/rocm_mcp/amd_smi"
        )

    def test_http_transport_honours_overrides(self) -> None:
        """Host, port and path overrides reach mcp.run()."""
        argv = [
            "amd-smi-mcp",
            "--transport",
            "http",
            "--host",
            "0.0.0.0",  # noqa: S104
            "--port",
            "9200",
            "--path",
            "/custom",
        ]
        with patch.object(amd_smi_mcp.mcp, "run") as run, patch("sys.argv", argv):
            amd_smi_mcp.main()
        run.assert_called_once_with(
            transport="streamable-http",
            host="0.0.0.0",  # noqa: S104
            port=9200,
            path="/custom",
        )

    def test_invalid_transport_rejected(self) -> None:
        """A transport outside the declared choices is rejected by argparse."""
        with (
            patch("sys.argv", ["amd-smi-mcp", "--transport", "carrier-pigeon"]),
            pytest.raises(SystemExit),
        ):
            amd_smi_mcp.main()


class TestToolRegistration:
    """Assert the tools are actually registered with the real FastMCP server."""

    def test_all_tools_registered_with_mcp(self) -> None:
        """Every tool name shows up in the server's tool listing."""
        expected = {
            "get_driver_information",
            "list_gpus",
            "get_gpu_static_info",
            "get_gpu_metrics",
            "get_gpu_firmware_info",
            "get_gpu_process_info",
            "get_gpu_bad_pages",
        }
        assert expected <= _registered_tools()

    def test_server_name(self) -> None:
        """The server advertises itself as amd-smi."""
        assert amd_smi_mcp.mcp.name == "amd-smi"


def _registered_tools() -> set[str]:
    """Fetch registered tool names across FastMCP versions."""
    for attr in ("list_tools", "get_tools"):
        getter = getattr(amd_smi_mcp.mcp, attr, None)
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
