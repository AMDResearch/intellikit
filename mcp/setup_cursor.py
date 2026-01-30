#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Setup script to register IntelliKit MCP servers with Cursor.

This script updates Cursor's MCP configuration to enable all IntelliKit tools:
- Nexus: Extract GPU kernel source code and assembly
- Accordo: Validate kernel correctness
- Metrix: Profile with human-readable metrics
- Linex: Source-level performance profiling

Usage:
    python setup_cursor.py              # Register all tools
    python setup_cursor.py --tool linex # Register specific tool only
"""

import argparse
import json
import sys
from pathlib import Path


def get_cursor_config_path() -> Path:
    """Get path to Cursor's MCP configuration file."""
    home = Path.home()

    # Try common Cursor config locations
    possible_paths = [
        home / ".cursor" / "mcp.json",
        home / ".config" / "cursor" / "mcp.json",
        home / "Library" / "Application Support" / "Cursor" / "mcp.json",
    ]

    for path in possible_paths:
        if path.parent.exists():
            return path

    # Default to first option
    return possible_paths[0]


def get_mcp_config(tool_name: str, repo_path: Path) -> dict:
    """Get MCP configuration for a specific tool."""
    server_path = repo_path / "mcp" / tool_name / "server.py"

    configs = {
        "nexus": {
            "command": "python3",
            "args": [str(server_path)],
            "description": "Extract GPU kernel source code and assembly from HSA packets",
        },
        "accordo": {
            "command": "python3",
            "args": [str(server_path)],
            "description": "Validate GPU kernel correctness with side-by-side comparison",
        },
        "metrix": {
            "command": "python3",
            "args": [str(server_path)],
            "description": "Profile GPU kernels with human-readable metrics",
        },
        "linex": {
            "command": "python3",
            "args": [str(server_path)],
            "description": "Source-level GPU performance profiling with cycle-accurate metrics",
        },
    }

    if tool_name not in configs:
        raise ValueError(f"Unknown tool: {tool_name}")

    return configs[tool_name]


def update_cursor_config(tools: list[str], repo_path: Path, dry_run: bool = False):
    """Update Cursor's MCP configuration."""
    config_path = get_cursor_config_path()

    print(f"Cursor MCP config: {config_path}")

    # Load existing config
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"✓ Loaded existing config with {len(config.get('mcpServers', {}))} servers")
    else:
        config = {"mcpServers": {}}
        print("Creating new config file")

    # Ensure mcpServers key exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Add/update each tool
    for tool in tools:
        tool_config = get_mcp_config(tool, repo_path)
        config["mcpServers"][f"intellikit-{tool}"] = tool_config
        print(f"✓ Configured {tool}: {tool_config['description']}")

    if dry_run:
        print("\nDry run - would write:")
        print(json.dumps(config, indent=2))
        return

    # Write config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Successfully updated Cursor MCP config at {config_path}")
    print("\nRestart Cursor to load the new MCP servers.")
    print("\nYou can now ask questions like:")
    print("  - 'What are the top instructions taking latency?' (uses Linex)")
    print("  - 'Extract the kernel source from my app' (uses Nexus)")
    print("  - 'Profile memory bandwidth of my kernel' (uses Metrix)")
    print("  - 'Validate my optimized kernel matches the reference' (uses Accordo)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Register IntelliKit MCP servers with Cursor",
    )
    parser.add_argument(
        "--tool",
        choices=["nexus", "accordo", "metrix", "linex"],
        help="Register specific tool only (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without actually writing",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Override Cursor config path",
    )

    args = parser.parse_args()

    # Get repository path
    repo_path = Path(__file__).parent.parent
    print(f"IntelliKit repository: {repo_path}")

    # Determine which tools to register
    tools = [args.tool] if args.tool else ["nexus", "accordo", "metrix", "linex"]

    # Update config
    try:
        update_cursor_config(tools, repo_path, dry_run=args.dry_run)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
