#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Quick test script to verify MCP servers are working."""

import json
import subprocess
import sys
from pathlib import Path


def test_mcp_server(tool_name: str):
    """Test an MCP server by listing its tools."""
    server_path = Path(__file__).parent / tool_name / "server.py"

    if not server_path.exists():
        print(f"❌ Server not found: {server_path}")
        return False

    # MCP list tools request
    request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    })

    try:
        result = subprocess.run(
            ["python3", str(server_path)],
            check=False, input=request,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            print(f"❌ {tool_name}: Server error")
            print(f"   stderr: {result.stderr}")
            return False

        # Parse response
        response = json.loads(result.stdout)
        if "result" in response and "tools" in response["result"]:
            tools = response["result"]["tools"]
            print(f"✅ {tool_name}: {len(tools)} tools available")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description'][:60]}...")
            return True
        print(f"❌ {tool_name}: Unexpected response format")
        return False

    except subprocess.TimeoutExpired:
        print(f"❌ {tool_name}: Server timeout")
        return False
    except json.JSONDecodeError:
        print(f"❌ {tool_name}: Invalid JSON response")
        print(f"   stdout: {result.stdout[:200]}")
        return False
    except Exception as e:
        print(f"❌ {tool_name}: {e}")
        return False


def main():
    """Test all MCP servers."""
    print("Testing IntelliKit MCP Servers")
    print("=" * 60)

    tools = ["linex", "nexus", "metrix", "accordo"]
    results = {}

    for tool in tools:
        print(f"\nTesting {tool}...")
        results[tool] = test_mcp_server(tool)

    print("\n" + "=" * 60)
    print("Summary:")
    passed = sum(results.values())
    total = len(results)
    print(f"{passed}/{total} servers working correctly")

    if passed == total:
        print("\n✅ All MCP servers are ready!")
        print("\nNext steps:")
        print("1. Install MCP SDK: pip install mcp")
        print("2. Register with Cursor: python mcp/setup_cursor.py")
        print("3. Restart Cursor")
        print("4. Ask questions like 'What are the top instructions taking latency?'")
        return 0
    print("\n❌ Some servers failed. Check the errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
