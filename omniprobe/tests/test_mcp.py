# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the Omniprobe MCP server tool registration."""

import asyncio

from omniprobe_ik.mcp.server import mcp


class TestMCPToolRegistration:
    """Verify MCP tools are registered correctly."""

    def test_server_name(self):
        assert mcp.name == "IntelliKit Omniprobe"

    def test_tools_registered(self):
        tools = asyncio.run(mcp.list_tools())
        tool_names = {t.name for t in tools}
        assert "analyze_memory" in tool_names
        assert "analyze_basic_blocks" in tool_names
        assert "list_analyzers" in tool_names
        assert "get_instrumentation_flags" in tool_names
        assert "run_analyzer" in tool_names
        assert len(tool_names) == 5
