# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the HIP compiler MCP server tools.

These exercise the MCP tool layer without ROCm or hipcc by substituting a
fake ``HipCompiler`` over the module-level singleton.  The fixtures build
real ``HipCompilerResult`` objects rather than mocks, so a field rename in
``rocm_mcp.compile.hip_compiler`` breaks these tests instead of silently
passing.

Both tools are covered on the success path, the "compiler ran but rejected
the code" path (``result.success`` is False, the analogue of an empty
result), and the failure paths -- the temporary directory could not be
created, or the compiler itself raised.  Every failure has to both return
the error string and report it through ``ctx.error``, and that is asserted.

All file paths come from ``tmp_path`` so the tests do not depend on the
working directory pytest happens to be invoked from.
"""

import asyncio
import inspect
from pathlib import Path
from unittest.mock import patch

import pytest

from rocm_mcp import HipCompilerResult
from rocm_mcp.compile import hip_compiler_mcp

SOURCE = """
#include <hip/hip_runtime.h>
int main() { return 0; }
"""


def _tool(obj):  # noqa: ANN001, ANN202
    """Return the underlying function whether or not FastMCP wrapped it."""
    return getattr(obj, "fn", obj)


compile_hip_source_file = _tool(hip_compiler_mcp.compile_hip_source_file)
compile_hip_source_string = _tool(hip_compiler_mcp.compile_hip_source_string)


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


class FakeHipCompiler:
    """Stand-in for HipCompiler that records compile() calls."""

    def __init__(
        self, result: HipCompilerResult | None = None, error: Exception | None = None
    ) -> None:
        """Serve ``result`` from compile(), or raise ``error`` if given."""
        self.result = result if result is not None else _ok()
        self.error = error
        self.calls: list[dict] = []

    def compile(
        self,
        source_file: Path,
        output_file: Path,
        extra_flags: list[str] | None = None,
    ) -> HipCompilerResult:
        """Record the arguments and serve the canned result or error."""
        self.calls.append(
            {"source_file": source_file, "output_file": output_file, "extra_flags": extra_flags}
        )
        if self.error is not None:
            raise self.error
        return self.result


def _ok() -> HipCompilerResult:
    """Build a real successful HipCompilerResult."""
    return HipCompilerResult(success=True, errors=None, raw_output="\n")


def _failed(errors: str = "error: expected ';'") -> HipCompilerResult:
    """Build a real failed HipCompilerResult."""
    return HipCompilerResult(success=False, errors=errors, raw_output=f"\n{errors}")


def _call(tool, fake: FakeHipCompiler, **kwargs) -> tuple[str, FakeContext]:  # noqa: ANN001
    """Run an async tool against a patched compiler singleton."""
    ctx = FakeContext()
    with patch.object(hip_compiler_mcp, "compiler", fake):
        return asyncio.run(tool(ctx, **kwargs)), ctx


class TestCompileHipSourceFile:
    """Cover the compile_hip_source_file tool."""

    def test_success_message_names_source_and_output(self, tmp_path: Path) -> None:
        """A successful compile reports both paths and logs via ctx.info."""
        src = tmp_path / "vectoradd.hip"
        src.write_text(SOURCE)
        out_file = tmp_path / "vectoradd.out"
        out, ctx = _call(
            compile_hip_source_file,
            FakeHipCompiler(),
            source_file=src,
            output_file=out_file,
        )
        assert out == f"Compilation of HIP code in {src} succeeded. Executable at {out_file}"
        assert ctx.infos == [out]
        assert ctx.errors == []

    def test_source_file_is_coerced_to_path(self, tmp_path: Path) -> None:
        """A string source path reaches the compiler as a Path."""
        src = tmp_path / "vectoradd.hip"
        src.write_text(SOURCE)
        fake = FakeHipCompiler()
        _call(
            compile_hip_source_file,
            fake,
            source_file=str(src),
            output_file=tmp_path / "a.out",
        )
        assert fake.calls[0]["source_file"] == Path(src)
        assert isinstance(fake.calls[0]["source_file"], Path)

    def test_temporary_output_named_after_source_stem(self, tmp_path: Path) -> None:
        """With no output_file a temporary file named after the source is used."""
        src = tmp_path / "vectoradd.hip"
        src.write_text(SOURCE)
        fake = FakeHipCompiler()
        out, _ = _call(compile_hip_source_file, fake, source_file=src)
        produced = fake.calls[0]["output_file"]
        assert produced.name == "vectoradd"
        # The temporary directory is somewhere other than the source directory.
        assert produced.parent != src.parent
        assert str(produced) in out

    def test_extra_flags_are_forwarded(self, tmp_path: Path) -> None:
        """Extra compiler flags reach the compiler unchanged."""
        src = tmp_path / "vectoradd.hip"
        src.write_text(SOURCE)
        fake = FakeHipCompiler()
        _call(
            compile_hip_source_file,
            fake,
            source_file=src,
            output_file=tmp_path / "a.out",
            extra_flags=["-O3", "-DNDEBUG"],
        )
        assert fake.calls[0]["extra_flags"] == ["-O3", "-DNDEBUG"]

    def test_extra_flags_default_to_none(self, tmp_path: Path) -> None:
        """Omitting extra_flags forwards None rather than an empty list."""
        src = tmp_path / "vectoradd.hip"
        src.write_text(SOURCE)
        fake = FakeHipCompiler()
        _call(compile_hip_source_file, fake, source_file=src, output_file=tmp_path / "a.out")
        assert fake.calls[0]["extra_flags"] is None

    def test_temp_dir_failure_is_returned_and_reported(self, tmp_path: Path) -> None:
        """A failure to allocate the temporary output is reported to the client."""
        src = tmp_path / "vectoradd.hip"
        src.write_text(SOURCE)
        fake = FakeHipCompiler()
        ctx = FakeContext()
        with (
            patch.object(hip_compiler_mcp, "compiler", fake),
            patch("tempfile.mkdtemp", side_effect=OSError("no space left on device")),
        ):
            out = asyncio.run(compile_hip_source_file(ctx, source_file=src))
        assert out == "Failed to create output file: no space left on device"
        assert ctx.errors == [out]
        # The compiler is never reached once the output path could not be made.
        assert fake.calls == []

    def test_compiler_exception_is_returned_and_reported(self, tmp_path: Path) -> None:
        """A raising compiler yields the error string and calls ctx.error."""
        src = tmp_path / "vectoradd.hip"
        src.write_text(SOURCE)
        out, ctx = _call(
            compile_hip_source_file,
            FakeHipCompiler(error=FileNotFoundError("hipcc executable not found at 'hipcc'")),
            source_file=src,
            output_file=tmp_path / "a.out",
        )
        assert out == f"Compilation of {src} failed: hipcc executable not found at 'hipcc'"
        assert ctx.errors == [out]

    def test_unsuccessful_compilation_reports_compiler_errors(self, tmp_path: Path) -> None:
        """A clean run that rejected the code surfaces the compiler diagnostics."""
        src = tmp_path / "broken.hip"
        src.write_text(SOURCE)
        out, ctx = _call(
            compile_hip_source_file,
            FakeHipCompiler(result=_failed("error: expected ';'")),
            source_file=src,
            output_file=tmp_path / "a.out",
        )
        assert out == f"Compilation of HIP code in {src} failed: error: expected ';'"
        assert ctx.errors == [out]
        assert ctx.infos == []


class TestCompileHipSourceString:
    """Cover the compile_hip_source_string tool."""

    def test_source_string_is_written_to_a_temporary_file(self) -> None:
        """The string lands in hip_source.cpp inside a temporary directory."""
        fake = FakeHipCompiler()
        out, ctx = _call(compile_hip_source_string, fake, source=SOURCE)
        written = fake.calls[0]["source_file"]
        assert written.name == "hip_source.cpp"
        assert written.read_text() == SOURCE
        assert out.startswith(f"Compilation of HIP code in {written} succeeded.")
        assert ctx.infos == [out]
        assert ctx.errors == []

    def test_temporary_executable_is_named_hip_exe(self) -> None:
        """With no output_file the executable defaults to hip_exe next to the source."""
        fake = FakeHipCompiler()
        _call(compile_hip_source_string, fake, source=SOURCE)
        call = fake.calls[0]
        assert call["output_file"].name == "hip_exe"
        assert call["output_file"].parent == call["source_file"].parent

    def test_explicit_output_file_is_honoured(self, tmp_path: Path) -> None:
        """An explicit output path is used instead of the temporary one."""
        fake = FakeHipCompiler()
        out_file = tmp_path / "custom.out"
        out, _ = _call(compile_hip_source_string, fake, source=SOURCE, output_file=out_file)
        assert fake.calls[0]["output_file"] == out_file
        assert out.endswith(f"Executable at {out_file}")

    def test_extra_flags_are_forwarded(self) -> None:
        """Extra compiler flags reach the compiler unchanged."""
        fake = FakeHipCompiler()
        _call(compile_hip_source_string, fake, source=SOURCE, extra_flags=["--offload-arch=gfx942"])
        assert fake.calls[0]["extra_flags"] == ["--offload-arch=gfx942"]

    def test_temp_dir_failure_is_returned_and_reported(self) -> None:
        """A failure to write the temporary source is reported to the client."""
        fake = FakeHipCompiler()
        ctx = FakeContext()
        with (
            patch.object(hip_compiler_mcp, "compiler", fake),
            patch("tempfile.mkdtemp", side_effect=OSError("read-only file system")),
        ):
            out = asyncio.run(compile_hip_source_string(ctx, source=SOURCE))
        assert out == "Failed to create temporary source file: read-only file system"
        assert ctx.errors == [out]
        assert fake.calls == []

    def test_compiler_exception_is_returned_and_reported(self) -> None:
        """A raising compiler yields the error string and calls ctx.error."""
        fake = FakeHipCompiler(error=RuntimeError("hipcc execution failed"))
        out, ctx = _call(compile_hip_source_string, fake, source=SOURCE)
        written = fake.calls[0]["source_file"]
        assert out == f"Compilation of {written} failed: hipcc execution failed"
        assert ctx.errors == [out]

    def test_unsuccessful_compilation_reports_compiler_errors(self) -> None:
        """A clean run that rejected the code surfaces the compiler diagnostics."""
        fake = FakeHipCompiler(result=_failed("error: use of undeclared identifier"))
        out, ctx = _call(compile_hip_source_string, fake, source="int main() { nope; }")
        written = fake.calls[0]["source_file"]
        assert out == (
            f"Compilation of HIP code in {written} failed: error: use of undeclared identifier"
        )
        assert ctx.errors == [out]
        assert ctx.infos == []


class TestMain:
    """Cover the transport dispatch in main()."""

    def test_default_transport_is_stdio(self) -> None:
        """With no arguments the server runs over stdio."""
        with (
            patch.object(hip_compiler_mcp.mcp, "run") as run,
            patch("sys.argv", ["hip-compiler-mcp"]),
        ):
            hip_compiler_mcp.main()
        run.assert_called_once_with(transport="stdio")

    def test_explicit_stdio(self) -> None:
        """An explicit --transport stdio behaves like the default."""
        with (
            patch.object(hip_compiler_mcp.mcp, "run") as run,
            patch("sys.argv", ["hip-compiler-mcp", "--transport", "stdio"]),
        ):
            hip_compiler_mcp.main()
        run.assert_called_once_with(transport="stdio")

    def test_http_transport_uses_defaults(self) -> None:
        """HTTP falls back to loopback, port 8000 and the hip_compiler path."""
        with (
            patch.object(hip_compiler_mcp.mcp, "run") as run,
            patch("sys.argv", ["hip-compiler-mcp", "--transport", "http"]),
        ):
            hip_compiler_mcp.main()
        run.assert_called_once_with(
            transport="streamable-http",
            host="127.0.0.1",
            port=8000,
            path="/rocm_mcp/hip_compiler",
        )

    def test_http_transport_honours_overrides(self) -> None:
        """Host, port and path overrides reach mcp.run()."""
        argv = [
            "hip-compiler-mcp",
            "--transport",
            "http",
            "--host",
            "0.0.0.0",  # noqa: S104
            "--port",
            "9300",
            "--path",
            "/custom",
        ]
        with patch.object(hip_compiler_mcp.mcp, "run") as run, patch("sys.argv", argv):
            hip_compiler_mcp.main()
        run.assert_called_once_with(
            transport="streamable-http",
            host="0.0.0.0",  # noqa: S104
            port=9300,
            path="/custom",
        )

    def test_invalid_transport_rejected(self) -> None:
        """A transport outside the declared choices is rejected by argparse."""
        with (
            patch("sys.argv", ["hip-compiler-mcp", "--transport", "carrier-pigeon"]),
            pytest.raises(SystemExit),
        ):
            hip_compiler_mcp.main()


class TestToolRegistration:
    """Assert the tools are actually registered with the real FastMCP server."""

    def test_both_tools_registered_with_mcp(self) -> None:
        """Both tool names show up in the server's tool listing."""
        assert {"compile_hip_source_file", "compile_hip_source_string"} <= _registered_tools()

    def test_server_name(self) -> None:
        """The server advertises itself as hip_compiler."""
        assert hip_compiler_mcp.mcp.name == "hip_compiler"


def _registered_tools() -> set[str]:
    """Fetch registered tool names across FastMCP versions."""
    for attr in ("list_tools", "get_tools"):
        getter = getattr(hip_compiler_mcp.mcp, attr, None)
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
