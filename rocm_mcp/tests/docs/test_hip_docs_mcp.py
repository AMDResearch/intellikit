# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the HIP documentation MCP server tools.

These exercise the MCP tool layer without any network access by substituting
a fake ``HipDocs`` for the class the tools instantiate.  The fixtures build
real ``HipApiResult`` objects rather than mocks, so a field rename in
``rocm_mcp.doc.hip_docs`` breaks these tests instead of silently passing.

Unlike the other rocm_mcp servers, these tools construct their collaborator
per call rather than using a module-level singleton, so the fake records the
``version`` it was constructed with -- that is the only way to prove the
version argument is honoured rather than ignored.

Both tools are covered on the success path, the no-results path, and the
failure path, where the collaborator raises and the tool has to both return
the error string and report it through ``ctx.error``.
"""

import asyncio
import inspect
from typing import ClassVar
from unittest.mock import patch

import pytest

from rocm_mcp.doc import HipApiResult, hip_docs_mcp


def _tool(obj):  # noqa: ANN001, ANN202
    """Return the underlying function whether or not FastMCP wrapped it."""
    return getattr(obj, "fn", obj)


search_hip_api = _tool(hip_docs_mcp.search_hip_api)
get_hip_api_reference = _tool(hip_docs_mcp.get_hip_api_reference)


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


class FakeHipDocs:
    """Stand-in for HipDocs that records construction and query arguments."""

    search_results: ClassVar[list[HipApiResult]] = []
    reference: ClassVar[HipApiResult | None] = None
    error: ClassVar[Exception | None] = None
    calls: ClassVar[list[dict]] = []

    def __init__(self, version: str = "latest") -> None:
        """Record the documentation version the tool asked for."""
        FakeHipDocs.calls.append({"version": version})

    def search_api(self, query: str, limit: int = 5) -> list[HipApiResult]:
        """Record the query and serve the canned results, or raise."""
        FakeHipDocs.calls[-1].update({"query": query, "limit": limit})
        if FakeHipDocs.error is not None:
            raise FakeHipDocs.error
        return FakeHipDocs.search_results

    def get_api_reference(self, api_name: str) -> HipApiResult | None:
        """Record the API name and serve the canned reference, or raise."""
        FakeHipDocs.calls[-1].update({"api_name": api_name})
        if FakeHipDocs.error is not None:
            raise FakeHipDocs.error
        return FakeHipDocs.reference


@pytest.fixture(autouse=True)
def _fake_hip_docs():  # noqa: ANN202
    """Patch HipDocs for every test and reset the recorded state."""
    FakeHipDocs.search_results = []
    FakeHipDocs.reference = None
    FakeHipDocs.error = None
    FakeHipDocs.calls = []
    with patch.object(hip_docs_mcp, "HipDocs", FakeHipDocs):
        yield


def _result(title: str, content: str | None = None) -> HipApiResult:
    """Build a real HipApiResult so a field rename fails the test."""
    return HipApiResult(
        title=title,
        url=f"https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group__Memory.html#{title}",
        description=f"{title} description",
        content=content,
    )


def _call(tool, **kwargs) -> tuple[str, FakeContext]:  # noqa: ANN001
    """Run an async tool with a recording context."""
    ctx = FakeContext()
    return asyncio.run(tool(ctx, **kwargs)), ctx


class TestSearchHipApi:
    """Cover the search_hip_api tool."""

    def test_formats_every_result(self) -> None:
        """The header, the numbering, and each URL and description are rendered."""
        FakeHipDocs.search_results = [_result("hipMalloc"), _result("hipMallocAsync")]
        out, ctx = _call(search_hip_api, query="hipMalloc")
        assert out.startswith("Found 2 results for 'hipMalloc' in HIP latest documentation:")
        assert "1. hipMalloc" in out
        assert "2. hipMallocAsync" in out
        assert f"   URL: {FakeHipDocs.search_results[0].url}" in out
        assert "   Description: hipMalloc description" in out
        assert ctx.errors == []

    def test_query_and_limit_are_forwarded(self) -> None:
        """The query string and result limit reach HipDocs.search_api()."""
        FakeHipDocs.search_results = [_result("hipMemcpy")]
        _call(search_hip_api, query="hipMemcpy", limit=3)
        assert FakeHipDocs.calls[0]["query"] == "hipMemcpy"
        assert FakeHipDocs.calls[0]["limit"] == 3  # noqa: PLR2004

    def test_limit_defaults_to_five(self) -> None:
        """Omitting the limit uses the documented default of five."""
        FakeHipDocs.search_results = [_result("hipMemcpy")]
        _call(search_hip_api, query="hipMemcpy")
        assert FakeHipDocs.calls[0]["limit"] == 5  # noqa: PLR2004

    def test_version_is_forwarded_and_reported(self) -> None:
        """A pinned version reaches the constructor and shows up in the header."""
        FakeHipDocs.search_results = [_result("hipMemcpy")]
        out, _ = _call(search_hip_api, query="hipMemcpy", version="6.2.0")
        assert FakeHipDocs.calls[0]["version"] == "6.2.0"
        assert "in HIP 6.2.0 documentation:" in out

    def test_version_defaults_to_latest(self) -> None:
        """Omitting the version asks for the latest documentation."""
        FakeHipDocs.search_results = [_result("hipMemcpy")]
        _call(search_hip_api, query="hipMemcpy")
        assert FakeHipDocs.calls[0]["version"] == "latest"

    def test_no_results(self) -> None:
        """An empty result set reports the query, and is not treated as an error."""
        out, ctx = _call(search_hip_api, query="hipNotAThing")
        assert out == "No HIP API documentation found for query: hipNotAThing"
        assert ctx.errors == []

    def test_failure_is_returned_and_reported(self) -> None:
        """A raising collaborator yields the error string and calls ctx.error."""
        FakeHipDocs.error = RuntimeError("connection reset")
        out, ctx = _call(search_hip_api, query="hipMalloc")
        assert out == "Error searching HIP API documentation: connection reset"
        assert ctx.errors == [out]


class TestGetHipApiReference:
    """Cover the get_hip_api_reference tool."""

    def test_renders_title_url_and_description(self) -> None:
        """The reference block leads with the title, URL and description."""
        FakeHipDocs.reference = _result("hipMalloc")
        out, ctx = _call(get_hip_api_reference, api_name="hipMalloc")
        assert out.startswith("HIP API Reference: hipMalloc")
        assert f"URL: {FakeHipDocs.reference.url}" in out
        assert "Description:\nhipMalloc description" in out
        assert ctx.errors == []

    def test_full_documentation_included_when_content_present(self) -> None:
        """A reference carrying content gets a Full Documentation section."""
        FakeHipDocs.reference = _result("hipMalloc", content="hipError_t hipMalloc(void** ptr)")
        out, _ = _call(get_hip_api_reference, api_name="hipMalloc")
        assert "Full Documentation:\nhipError_t hipMalloc(void** ptr)" in out

    def test_full_documentation_omitted_when_content_absent(self) -> None:
        """A reference without content omits the Full Documentation section."""
        FakeHipDocs.reference = _result("hipMalloc")
        out, _ = _call(get_hip_api_reference, api_name="hipMalloc")
        assert "Full Documentation:" not in out

    def test_api_name_and_version_are_forwarded(self) -> None:
        """The API name and version reach the collaborator."""
        FakeHipDocs.reference = _result("hipFree")
        _call(get_hip_api_reference, api_name="hipFree", version="6.2.0")
        assert FakeHipDocs.calls[0] == {"version": "6.2.0", "api_name": "hipFree"}

    def test_no_reference_found(self) -> None:
        """A missing reference reports the name, and is not treated as an error."""
        out, ctx = _call(get_hip_api_reference, api_name="hipNotAThing")
        assert out == "No HIP API reference found for: hipNotAThing"
        assert ctx.errors == []

    def test_failure_is_returned_and_reported(self) -> None:
        """A raising collaborator yields the error string and calls ctx.error."""
        FakeHipDocs.error = RuntimeError("read timeout")
        out, ctx = _call(get_hip_api_reference, api_name="hipMalloc")
        assert out == "Failed to get HIP API reference: read timeout"
        assert ctx.errors == [out]


class TestMain:
    """Cover the transport dispatch in main()."""

    def test_default_transport_is_stdio(self) -> None:
        """With no arguments the server runs over stdio."""
        with patch.object(hip_docs_mcp.mcp, "run") as run, patch("sys.argv", ["hip-docs-mcp"]):
            hip_docs_mcp.main()
        run.assert_called_once_with(transport="stdio")

    def test_explicit_stdio(self) -> None:
        """An explicit --transport stdio behaves like the default."""
        with (
            patch.object(hip_docs_mcp.mcp, "run") as run,
            patch("sys.argv", ["hip-docs-mcp", "--transport", "stdio"]),
        ):
            hip_docs_mcp.main()
        run.assert_called_once_with(transport="stdio")

    def test_http_transport_uses_defaults(self) -> None:
        """HTTP falls back to loopback, port 8000 and the hip_docs path."""
        with (
            patch.object(hip_docs_mcp.mcp, "run") as run,
            patch("sys.argv", ["hip-docs-mcp", "--transport", "http"]),
        ):
            hip_docs_mcp.main()
        run.assert_called_once_with(
            transport="streamable-http", host="127.0.0.1", port=8000, path="/rocm_mcp/hip_docs"
        )

    def test_http_transport_honours_overrides(self) -> None:
        """Host, port and path overrides reach mcp.run()."""
        argv = [
            "hip-docs-mcp",
            "--transport",
            "http",
            "--host",
            "0.0.0.0",  # noqa: S104
            "--port",
            "9400",
            "--path",
            "/custom",
        ]
        with patch.object(hip_docs_mcp.mcp, "run") as run, patch("sys.argv", argv):
            hip_docs_mcp.main()
        run.assert_called_once_with(
            transport="streamable-http",
            host="0.0.0.0",  # noqa: S104
            port=9400,
            path="/custom",
        )

    def test_invalid_transport_rejected(self) -> None:
        """A transport outside the declared choices is rejected by argparse."""
        with (
            patch("sys.argv", ["hip-docs-mcp", "--transport", "carrier-pigeon"]),
            pytest.raises(SystemExit),
        ):
            hip_docs_mcp.main()


class TestToolRegistration:
    """Assert the tools are actually registered with the real FastMCP server."""

    def test_both_tools_registered_with_mcp(self) -> None:
        """Both tool names show up in the server's tool listing."""
        assert {"search_hip_api", "get_hip_api_reference"} <= _registered_tools()

    def test_server_name(self) -> None:
        """The server advertises itself as hip_docs."""
        assert hip_docs_mcp.mcp.name == "hip_docs"


def _registered_tools() -> set[str]:
    """Fetch registered tool names across FastMCP versions."""
    for attr in ("list_tools", "get_tools"):
        getter = getattr(hip_docs_mcp.mcp, attr, None)
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
