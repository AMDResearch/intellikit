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


class TestDiscoveryGuidance:
    """What an agent that never opens a skill sees: server instructions and tool descriptions."""

    def test_server_instructions_name_the_trigger_and_workflow(self):
        text = mcp.instructions
        assert "global_load_efficiency" in text
        assert "get_instrumentation_flags" in text and "analyze_memory" in text

    def test_analyze_memory_description_leads_with_when_to_use_it(self):
        tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
        description = tools["analyze_memory"].description
        assert "global_load_efficiency" in description
        assert "index array" in description
