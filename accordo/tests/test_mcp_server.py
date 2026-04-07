"""Tests for the Accordo MCP server entry point."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


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
    server_path = Path(__file__).resolve().parents[1] / "accordo" / "mcp" / "server.py"
    spec = importlib.util.spec_from_file_location("test_accordo_mcp_server", server_path)
    module = importlib.util.module_from_spec(spec)

    fake_fastmcp = types.ModuleType("fastmcp")
    fake_fastmcp.FastMCP = FakeFastMCP

    fake_accordo = types.ModuleType("accordo")
    fake_accordo.Accordo = object

    monkeypatch.setitem(sys.modules, "fastmcp", fake_fastmcp)
    monkeypatch.setitem(sys.modules, "accordo", fake_accordo)

    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module


@pytest.mark.parametrize(
    ("argv", "expected_run_kwargs"),
    [
        (["accordo-mcp"], {"transport": "stdio"}),
        (
            [
                "accordo-mcp",
                "--transport",
                "http",
                "--host",
                "0.0.0.0",
                "--port",
                "9001",
                "--path",
                "/custom-accordo",
            ],
            {
                "transport": "streamable-http",
                "host": "0.0.0.0",
                "port": 9001,
                "path": "/custom-accordo",
            },
        ),
    ],
)
def test_main_dispatches_transport_options(server_module, monkeypatch, argv, expected_run_kwargs):
    """Accordo MCP should map CLI transport options to the FastMCP runtime."""
    monkeypatch.setattr(sys, "argv", argv)

    server_module.main()

    server_module.mcp.run.assert_called_once_with(**expected_run_kwargs)
