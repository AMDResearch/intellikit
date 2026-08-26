"""Tests for the Kerncap MCP server — the three tools and the entry point.

The tools are the surface agents actually call, so they are driven directly
rather than through the CLI.  ``Kerncap`` is stubbed at its call site, which
is what lets these run with no GPU; the values it returns are real
``KernelStat`` / ``ExtractResult`` / ``ValidationResult`` objects so that
renaming a field breaks these tests instead of passing silently.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kerncap.extract import ExtractResult
from kerncap.profiler import KernelStat
from kerncap.validator import ValidationResult


class FakeFastMCP:
    """Small FastMCP stub so the server module can be imported in isolation."""

    def __init__(self, *_args, **_kwargs):
        self.run = MagicMock()

    def tool(self):
        def decorator(func):
            return func

        return decorator


@pytest.fixture
def server_module(monkeypatch):
    """Load the server module with lightweight dependency stubs."""
    server_path = Path(__file__).resolve().parents[2] / "kerncap" / "mcp" / "server.py"
    spec = importlib.util.spec_from_file_location("test_kerncap_mcp_server", server_path)
    assert spec is not None, f"Could not create module spec for {server_path}"
    module = importlib.util.module_from_spec(spec)

    fake_fastmcp = types.ModuleType("fastmcp")
    fake_fastmcp.FastMCP = FakeFastMCP

    monkeypatch.setitem(sys.modules, "fastmcp", fake_fastmcp)

    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module


@pytest.mark.parametrize(
    ("argv", "expected_run_kwargs"),
    [
        (["kerncap-mcp"], {"transport": "stdio"}),
        (
            [
                "kerncap-mcp",
                "--transport",
                "http",
                "--host",
                "0.0.0.0",
                "--port",
                "9002",
                "--path",
                "/custom-kerncap",
            ],
            {
                "transport": "streamable-http",
                "host": "0.0.0.0",
                "port": 9002,
                "path": "/custom-kerncap",
            },
        ),
    ],
)
def test_main_dispatches_transport_options(server_module, monkeypatch, argv, expected_run_kwargs):
    """Kerncap MCP should map CLI transport options to the FastMCP runtime."""
    monkeypatch.setattr(sys, "argv", argv)

    server_module.main()

    server_module.mcp.run.assert_called_once_with(**expected_run_kwargs)


def test_main_rejects_an_unknown_transport(server_module, monkeypatch):
    """argparse must reject the value rather than silently doing nothing."""
    monkeypatch.setattr(sys, "argv", ["kerncap-mcp", "--transport", "grpc"])

    with pytest.raises(SystemExit):
        server_module.main()

    server_module.mcp.run.assert_not_called()


def test_main_http_defaults(server_module, monkeypatch):
    """Bare ``--transport http`` must bind loopback:8000/kerncap."""
    monkeypatch.setattr(sys, "argv", ["kerncap-mcp", "--transport", "http"])

    server_module.main()

    server_module.mcp.run.assert_called_once_with(
        transport="streamable-http", host="127.0.0.1", port=8000, path="/kerncap"
    )


@pytest.fixture
def fake_kerncap():
    """Patch ``kerncap.Kerncap`` and hand back the instance the tools will use.

    The tools do ``from kerncap import Kerncap`` inside the function body, so
    the name resolves from the module at call time and patching it here is
    what the tools actually see.
    """
    instance = MagicMock()
    with patch("kerncap.Kerncap", return_value=instance) as cls:
        yield cls, instance


# --------------------------------------------------------------------------
# profile_kernels
# --------------------------------------------------------------------------


def test_profile_kernels_maps_every_reported_field(server_module, fake_kerncap):
    _cls, kc = fake_kerncap
    kc.profile.return_value = [
        KernelStat(
            name="matmul_kernel",
            calls=1024,
            total_duration_ns=580_000_000,
            avg_duration_ns=566_406,
            percentage=42.3,
            min_duration_ns=480_000,
            max_duration_ns=720_000,
            stddev_ns=45_000.0,
        )
    ]

    out = server_module.profile_kernels(["./my_app", "--flag"])

    kc.profile.assert_called_once_with(["./my_app", "--flag"], output_path=None)
    assert out == {
        "kernels": [
            {
                "name": "matmul_kernel",
                "calls": 1024,
                "total_duration_ns": 580_000_000,
                "avg_duration_ns": 566_406,
                "percentage": 42.3,
            }
        ]
    }


def test_profile_kernels_does_not_leak_the_distribution_fields(server_module, fake_kerncap):
    """min/max/stddev exist on KernelStat but are deliberately not returned.

    If they are ever added, this test should be updated rather than deleted —
    it exists so the tool's wire format cannot drift unnoticed.
    """
    _cls, kc = fake_kerncap
    kc.profile.return_value = [
        KernelStat(
            name="k",
            calls=1,
            total_duration_ns=10,
            avg_duration_ns=10,
            percentage=100.0,
            min_duration_ns=5,
            max_duration_ns=15,
            stddev_ns=2.5,
        )
    ]

    entry = server_module.profile_kernels(["./app"])["kernels"][0]

    assert set(entry) == {
        "name",
        "calls",
        "total_duration_ns",
        "avg_duration_ns",
        "percentage",
    }


def test_profile_kernels_preserves_rank_order(server_module, fake_kerncap):
    """The tool must not re-sort — run_profile already ranked these."""
    _cls, kc = fake_kerncap
    kc.profile.return_value = [
        KernelStat(
            name="slow", calls=1, total_duration_ns=900, avg_duration_ns=900, percentage=90.0
        ),
        KernelStat(
            name="fast", calls=1, total_duration_ns=100, avg_duration_ns=100, percentage=10.0
        ),
    ]

    names = [k["name"] for k in server_module.profile_kernels(["./app"])["kernels"]]

    assert names == ["slow", "fast"]


def test_profile_kernels_forwards_output_path(server_module, fake_kerncap):
    _cls, kc = fake_kerncap
    kc.profile.return_value = []

    server_module.profile_kernels(["./app"], output_path="/tmp/profile.json")

    kc.profile.assert_called_once_with(["./app"], output_path="/tmp/profile.json")


def test_profile_kernels_empty_result(server_module, fake_kerncap):
    _cls, kc = fake_kerncap
    kc.profile.return_value = []

    assert server_module.profile_kernels(["./app"]) == {"kernels": []}


def test_profile_kernels_propagates_failures(server_module, fake_kerncap):
    """Errors must surface to the agent, not be swallowed into an empty list."""
    _cls, kc = fake_kerncap
    kc.profile.side_effect = RuntimeError("rocprofv3 not found")

    with pytest.raises(RuntimeError, match="rocprofv3 not found"):
        server_module.profile_kernels(["./app"])


# --------------------------------------------------------------------------
# extract_kernel
# --------------------------------------------------------------------------


def test_extract_kernel_forwards_arguments_and_maps_the_result(server_module, fake_kerncap):
    _cls, kc = fake_kerncap
    kc.extract.return_value = ExtractResult(
        output_dir="/iso/attn",
        capture_dir="/iso/attn/capture",
        language="triton",
        has_source=True,
        generated_files=["kernel_variant.py", "reproducer.py"],
    )

    out = server_module.extract_kernel(
        kernel_name="attn_fwd",
        cmd="./bench --fa",
        source_dir="./src",
        output="/iso/attn",
        language="triton",
        dispatch=2,
    )

    kc.extract.assert_called_once_with(
        kernel_name="attn_fwd",
        cmd="./bench --fa",
        source_dir="./src",
        output="/iso/attn",
        language="triton",
        dispatch=2,
    )
    assert out == {
        "output_dir": "/iso/attn",
        "language": "triton",
        "has_source": True,
        "generated_files": ["kernel_variant.py", "reproducer.py"],
    }


def test_extract_kernel_defaults(server_module, fake_kerncap):
    _cls, kc = fake_kerncap
    kc.extract.return_value = ExtractResult(output_dir="/o", capture_dir="/o/capture")

    server_module.extract_kernel(kernel_name="vec", cmd="./app")

    assert kc.extract.call_args.kwargs == {
        "kernel_name": "vec",
        "cmd": "./app",
        "source_dir": None,
        "output": None,
        "language": None,
        "dispatch": -1,
    }


def test_extract_kernel_does_not_expose_capture_dir(server_module, fake_kerncap):
    """capture_dir is an internal artifact path; the tool returns output_dir."""
    _cls, kc = fake_kerncap
    kc.extract.return_value = ExtractResult(
        output_dir="/o", capture_dir="/o/capture", language="hip"
    )

    out = server_module.extract_kernel(kernel_name="vec", cmd="./app")

    assert set(out) == {"output_dir", "language", "has_source", "generated_files"}


def test_extract_kernel_reports_a_sourceless_capture(server_module, fake_kerncap):
    """A capture with no source trail is a valid result, not an error."""
    _cls, kc = fake_kerncap
    kc.extract.return_value = ExtractResult(
        output_dir="/o",
        capture_dir="/o/capture",
        language="hip",
        has_source=False,
        generated_files=["Makefile"],
    )

    out = server_module.extract_kernel(kernel_name="Tensile_gemm", cmd="./app")

    assert out["has_source"] is False
    assert out["language"] == "hip"


def test_extract_kernel_propagates_failures(server_module, fake_kerncap):
    _cls, kc = fake_kerncap
    kc.extract.side_effect = RuntimeError("no dispatch matched")

    with pytest.raises(RuntimeError, match="no dispatch matched"):
        server_module.extract_kernel(kernel_name="nope", cmd="./app")


# --------------------------------------------------------------------------
# validate_reproducer
# --------------------------------------------------------------------------


def test_validate_reproducer_forwards_arguments_and_maps_the_result(server_module, fake_kerncap):
    _cls, kc = fake_kerncap
    kc.validate.return_value = ValidationResult(
        passed=True,
        details=["4 of 4 regions identical"],
        max_error=0.0,
        mode="byte-exact",
    )

    out = server_module.validate_reproducer(
        reproducer_dir="/iso/gemm",
        tolerance=1e-4,
        rtol=1e-3,
        hsaco="/iso/gemm/candidate.hsaco",
    )

    kc.validate.assert_called_once_with(
        reproducer_dir="/iso/gemm",
        tolerance=1e-4,
        rtol=1e-3,
        hsaco="/iso/gemm/candidate.hsaco",
    )
    assert out == {
        "passed": True,
        "max_error": 0.0,
        "details": ["4 of 4 regions identical"],
    }


def test_validate_reproducer_defaults(server_module, fake_kerncap):
    _cls, kc = fake_kerncap
    kc.validate.return_value = ValidationResult(passed=True, details=[])

    server_module.validate_reproducer(reproducer_dir="/iso/gemm")

    assert kc.validate.call_args.kwargs == {
        "reproducer_dir": "/iso/gemm",
        "tolerance": 1e-6,
        "rtol": 1e-5,
        "hsaco": None,
    }


def test_validate_reproducer_reports_failure_without_raising(server_module, fake_kerncap):
    """A failed validation is a result, not a tool error."""
    _cls, kc = fake_kerncap
    kc.validate.return_value = ValidationResult(
        passed=False,
        details=["region_0x2000.bin: DIFFERS"],
        max_error=0.25,
        mode="numeric",
    )

    out = server_module.validate_reproducer(reproducer_dir="/iso/gemm")

    assert out["passed"] is False
    assert out["max_error"] == 0.25
    assert out["details"] == ["region_0x2000.bin: DIFFERS"]


def test_validate_reproducer_reports_nan_max_error(server_module, fake_kerncap):
    """NaN is a meaningful outcome the validator sets deliberately."""
    import math

    _cls, kc = fake_kerncap
    kc.validate.return_value = ValidationResult(
        passed=False, details=["3 NaN elements"], max_error=float("nan"), mode="numeric"
    )

    out = server_module.validate_reproducer(reproducer_dir="/iso/attn")

    assert math.isnan(out["max_error"])


def test_validate_reproducer_omits_region_lines(server_module, fake_kerncap):
    """region_lines is CLI presentation detail and stays off the wire."""
    _cls, kc = fake_kerncap
    kc.validate.return_value = ValidationResult(
        passed=True,
        details=["ok"],
        mode="byte-exact",
        region_lines=["region_0x1000.bin: PASS (identical)"],
    )

    out = server_module.validate_reproducer(reproducer_dir="/iso/gemm")

    assert set(out) == {"passed", "max_error", "details"}


def test_validate_reproducer_propagates_failures(server_module, fake_kerncap):
    _cls, kc = fake_kerncap
    kc.validate.side_effect = FileNotFoundError("capture/ not found")

    with pytest.raises(FileNotFoundError, match="capture/ not found"):
        server_module.validate_reproducer(reproducer_dir="/nope")


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def registered_tool_names(server) -> set:
    """Return the names of tools registered on ``server``.

    Fails loudly rather than skipping if the fastmcp listing API moves — a
    permanently-skipped test contributes nothing while looking green.
    """
    if hasattr(server, "get_tools"):
        tools = asyncio.run(server.get_tools())
    elif hasattr(server, "list_tools"):
        tools = asyncio.run(server.list_tools())
    else:
        pytest.fail("fastmcp exposes neither get_tools nor list_tools; update this test")

    if isinstance(tools, dict):
        return set(tools)
    return {t.name for t in tools}


def test_all_three_tools_are_registered():
    """Driven against the real fastmcp, so a dropped decorator is caught."""
    from kerncap.mcp import server as real_server

    assert registered_tool_names(real_server.mcp) == {
        "profile_kernels",
        "extract_kernel",
        "validate_reproducer",
    }


def test_server_name():
    from kerncap.mcp import server as real_server

    assert real_server.mcp.name == "IntelliKit Kerncap"
