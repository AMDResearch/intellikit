# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the uProf profiler MCP server tool.

These exercise the MCP tool layer without AMD uProf by substituting a fake
profiler over the module-level singleton.  The fake returns a real
``UProfProfilerResult`` pointing at a report file created under ``tmp_path``,
so a field rename in ``uprof_mcp.uprof_profiler`` breaks these tests instead
of silently passing, and the tool genuinely has to open and read the report.

The tool is covered on the success path with an explicit output directory,
the success path that falls back to a temporary directory, the empty path
(the report exists but has nothing in it), and the failure paths -- the
profiler raised, or the report it named cannot be read.  Every failure has to
both return the error string and report it through ``ctx.error``, and that is
asserted.
"""

import asyncio
import inspect
from pathlib import Path
from unittest.mock import patch

import pytest

from uprof_mcp import uprof_profiler_mcp
from uprof_mcp.uprof_profiler import UProfProfilerResult

REPORT = "Function,Samples\nmain,1024\nkernel,512\n"


def _tool(obj):  # noqa: ANN001, ANN202
    """Return the underlying function whether or not FastMCP wrapped it."""
    return getattr(obj, "fn", obj)


profile_for_hotspots = _tool(uprof_profiler_mcp.profile_for_hotspots)


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


class FakeUProfProfiler:
    """Stand-in for UProfProfiler that records find_hotspots() calls."""

    def __init__(
        self, result: UProfProfilerResult | None = None, error: Exception | None = None
    ) -> None:
        """Serve ``result`` from find_hotspots(), or raise ``error`` if given."""
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def find_hotspots(
        self,
        output_dir: str | Path,
        executable: str | Path,
        executable_args: list[str] | None,
    ) -> UProfProfilerResult:
        """Record the arguments and serve the canned result or error."""
        self.calls.append(
            {
                "output_dir": output_dir,
                "executable": executable,
                "executable_args": executable_args,
            }
        )
        if self.error is not None:
            raise self.error
        return self.result


def _report(tmp_path: Path, text: str = REPORT) -> UProfProfilerResult:
    """Write a report file and wrap it in a real UProfProfilerResult."""
    report_path = tmp_path / "AMDuProf-app.csv"
    report_path.write_text(text)
    return UProfProfilerResult(results_path=tmp_path / "datafiles", report_path=report_path)


def _call(fake: FakeUProfProfiler, **kwargs) -> tuple[str, FakeContext]:
    """Run the tool against a patched profiler singleton."""
    ctx = FakeContext()
    with patch.object(uprof_profiler_mcp, "profiler", fake):
        return asyncio.run(profile_for_hotspots(ctx, **kwargs)), ctx


class TestProfileForHotspots:
    """Cover the profile_for_hotspots tool."""

    def test_returns_the_report_contents(self, tmp_path: Path) -> None:
        """The tool reads the report uProf produced and returns it verbatim."""
        out_dir = tmp_path / "results"
        out, ctx = _call(
            FakeUProfProfiler(_report(tmp_path)),
            executable="./app",
            executable_arguments=[],
            output_dir=out_dir,
        )
        assert out == REPORT
        assert ctx.errors == []

    def test_reports_completion_through_ctx_info(self, tmp_path: Path) -> None:
        """A successful run is announced with the report path."""
        result = _report(tmp_path)
        _, ctx = _call(
            FakeUProfProfiler(result),
            executable="./app",
            executable_arguments=[],
            output_dir=tmp_path / "results",
        )
        assert ctx.infos == [f"Profiling of ./app completed with results in {result.report_path}."]

    def test_executable_and_arguments_are_forwarded(self, tmp_path: Path) -> None:
        """The executable and its arguments reach the profiler unchanged."""
        fake = FakeUProfProfiler(_report(tmp_path))
        _call(
            fake,
            executable="./app",
            executable_arguments=["--size", "1024"],
            output_dir=tmp_path / "results",
        )
        assert fake.calls[0]["executable"] == "./app"
        assert fake.calls[0]["executable_args"] == ["--size", "1024"]

    def test_output_directory_is_created(self, tmp_path: Path) -> None:
        """A missing output directory, including parents, is created first."""
        out_dir = tmp_path / "nested" / "results"
        fake = FakeUProfProfiler(_report(tmp_path))
        _call(fake, executable="./app", executable_arguments=[], output_dir=out_dir)
        assert out_dir.is_dir()
        assert fake.calls[0]["output_dir"] == out_dir

    def test_string_output_directory_is_coerced_to_path(self, tmp_path: Path) -> None:
        """A string output directory reaches the profiler as a Path."""
        out_dir = tmp_path / "results"
        fake = FakeUProfProfiler(_report(tmp_path))
        _call(fake, executable="./app", executable_arguments=[], output_dir=str(out_dir))
        assert fake.calls[0]["output_dir"] == out_dir
        assert isinstance(fake.calls[0]["output_dir"], Path)

    def test_falls_back_to_a_kept_temporary_directory(self, tmp_path: Path) -> None:
        """With no output directory a temporary one that survives the run is used."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        mkdtemp_calls: list[int] = []
        fake = FakeUProfProfiler(_report(tmp_path))
        ctx = FakeContext()

        def _fake_mkdtemp() -> str:
            mkdtemp_calls.append(1)
            return str(scratch)

        with (
            patch.object(uprof_profiler_mcp, "profiler", fake),
            patch("tempfile.mkdtemp", _fake_mkdtemp),
        ):
            out = asyncio.run(
                profile_for_hotspots(ctx, executable="./app", executable_arguments=[])
            )
        assert out == REPORT
        assert fake.calls[0]["output_dir"] == str(scratch)
        # mkdtemp, not TemporaryDirectory(delete=False): the latter is 3.12+ and this
        # package supports >=3.10. The directory must outlive the call either way --
        # the report is read out of it after we return.
        assert mkdtemp_calls == [1]

    def test_empty_report(self, tmp_path: Path) -> None:
        """An empty report is returned as an empty string, not an error."""
        out, ctx = _call(
            FakeUProfProfiler(_report(tmp_path, text="")),
            executable="./app",
            executable_arguments=[],
            output_dir=tmp_path / "results",
        )
        assert out == ""
        assert ctx.errors == []

    def test_profiler_failure_is_returned_and_reported(self, tmp_path: Path) -> None:
        """A raising profiler yields the error string and calls ctx.error."""
        out, ctx = _call(
            FakeUProfProfiler(error=RuntimeError("Profiling report not found")),
            executable="./app",
            executable_arguments=[],
            output_dir=tmp_path / "results",
        )
        assert out == "Profiling of ./app failed: Profiling report not found"
        assert ctx.errors == [out]

    def test_missing_uprof_is_returned_and_reported(self, tmp_path: Path) -> None:
        """A missing uProf CLI surfaces the same way as any other failure."""
        error = FileNotFoundError("uprof executable not found at '/opt/uprof/AMDuProfCLI'")
        out, ctx = _call(
            FakeUProfProfiler(error=error),
            executable="./app",
            executable_arguments=[],
            output_dir=tmp_path / "results",
        )
        assert out.startswith("Profiling of ./app failed:")
        assert ctx.errors == [out]

    def test_unreadable_report_is_returned_and_reported(self, tmp_path: Path) -> None:
        """A report path that does not exist is reported rather than raised."""
        result = UProfProfilerResult(
            results_path=tmp_path / "datafiles",
            report_path=tmp_path / "does-not-exist.csv",
        )
        out, ctx = _call(
            FakeUProfProfiler(result),
            executable="./app",
            executable_arguments=[],
            output_dir=tmp_path / "results",
        )
        assert out.startswith("Profiling of ./app failed:")
        assert ctx.errors == [out]


class TestMain:
    """Cover the transport dispatch in main()."""

    def test_default_transport_is_stdio(self) -> None:
        """With no arguments the server runs over stdio."""
        with (
            patch.object(uprof_profiler_mcp.mcp, "run") as run,
            patch("sys.argv", ["uprof-profiler-mcp"]),
        ):
            uprof_profiler_mcp.main()
        run.assert_called_once_with(transport="stdio")

    def test_explicit_stdio(self) -> None:
        """An explicit --transport stdio behaves like the default."""
        with (
            patch.object(uprof_profiler_mcp.mcp, "run") as run,
            patch("sys.argv", ["uprof-profiler-mcp", "--transport", "stdio"]),
        ):
            uprof_profiler_mcp.main()
        run.assert_called_once_with(transport="stdio")

    def test_http_transport_uses_defaults(self) -> None:
        """HTTP falls back to loopback, port 8000 and the uprof_mcp path."""
        with (
            patch.object(uprof_profiler_mcp.mcp, "run") as run,
            patch("sys.argv", ["uprof-profiler-mcp", "--transport", "http"]),
        ):
            uprof_profiler_mcp.main()
        run.assert_called_once_with(
            transport="streamable-http", host="127.0.0.1", port=8000, path="/uprof_mcp"
        )

    def test_http_transport_honours_overrides(self) -> None:
        """Host, port and path overrides reach mcp.run()."""
        argv = [
            "uprof-profiler-mcp",
            "--transport",
            "http",
            "--host",
            "0.0.0.0",  # noqa: S104
            "--port",
            "9500",
            "--path",
            "/custom",
        ]
        with patch.object(uprof_profiler_mcp.mcp, "run") as run, patch("sys.argv", argv):
            uprof_profiler_mcp.main()
        run.assert_called_once_with(
            transport="streamable-http",
            host="0.0.0.0",  # noqa: S104
            port=9500,
            path="/custom",
        )

    def test_invalid_transport_rejected(self) -> None:
        """A transport outside the declared choices is rejected by argparse."""
        with (
            patch("sys.argv", ["uprof-profiler-mcp", "--transport", "carrier-pigeon"]),
            pytest.raises(SystemExit),
        ):
            uprof_profiler_mcp.main()


class TestToolRegistration:
    """Assert the tool is actually registered with the real FastMCP server."""

    def test_tool_registered_with_mcp(self) -> None:
        """The tool name shows up in the server's tool listing."""
        assert "profile_for_hotspots" in _registered_tools()

    def test_server_name(self) -> None:
        """The server advertises itself as uprof_profiler."""
        assert uprof_profiler_mcp.mcp.name == "uprof_profiler"


def _registered_tools() -> set[str]:
    """Fetch registered tool names across FastMCP versions."""
    for attr in ("list_tools", "get_tools"):
        getter = getattr(uprof_profiler_mcp.mcp, attr, None)
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
