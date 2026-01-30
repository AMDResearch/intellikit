# IntelliKit MCP Servers

**Model Context Protocol (MCP) servers for all IntelliKit tools.**

This enables LLMs (like Claude in Cursor) to automatically discover and use IntelliKit tools to answer GPU profiling questions.

## Quick Start

### 1. Install MCP Python SDK

```bash
pip install mcp
```

### 2. Register with Cursor

```bash
cd /work1/amd/muhaawad/git/amd/audacious/intellikit
python mcp/setup_cursor.py
```

This will update your Cursor configuration (`~/.cursor/mcp.json`) to register all four tools.

### 3. Restart Cursor

Restart Cursor to load the MCP servers.

### 4. Ask Questions!

Now you can ask Cursor questions like:

- **"What are the top instructions taking latency in my kernel?"**
  → Cursor uses **Linex** to profile and analyze

- **"Extract the HIP source code from my application"**
  → Cursor uses **Nexus** to intercept and extract

- **"What's the memory bandwidth utilization of my kernel?"**
  → Cursor uses **Metrix** to profile metrics

- **"Validate my optimized kernel matches the reference"**
  → Cursor uses **Accordo** to compare outputs

## Available MCP Servers

### Linex (`intellikit-linex`)
**Source-level GPU performance profiling**

Tools:
- `profile_application` - Get cycle counts mapped to source lines
- `analyze_instruction_hotspots` - Drill down into ISA-level metrics

### Nexus (`intellikit-nexus`)
**GPU kernel source extraction**

Tools:
- `extract_kernel_code` - Extract HIP/Triton source and assembly
- `list_kernels` - List all kernels in an application

### Metrix (`intellikit-metrix`)
**Human-readable GPU metrics**

Tools:
- `profile_metrics` - Collect hardware performance counters
- `list_available_metrics` - Show all available metrics

### Accordo (`intellikit-accordo`)
**Kernel validation**

Tools:
- `validate_kernel_correctness` - Compare reference vs optimized outputs

## Manual Configuration

If the setup script doesn't work for your Cursor installation, manually add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "intellikit-linex": {
      "command": "python3",
      "args": ["/path/to/intellikit/mcp/linex/server.py"],
      "description": "Source-level GPU performance profiling"
    },
    "intellikit-nexus": {
      "command": "python3",
      "args": ["/path/to/intellikit/mcp/nexus/server.py"],
      "description": "Extract GPU kernel source code"
    },
    "intellikit-metrix": {
      "command": "python3",
      "args": ["/path/to/intellikit/mcp/metrix/server.py"],
      "description": "Human-readable GPU metrics"
    },
    "intellikit-accordo": {
      "command": "python3",
      "args": ["/path/to/intellikit/mcp/accordo/server.py"],
      "description": "Validate kernel correctness"
    }
  }
}
```

## Testing MCP Servers

Test a server directly:

```bash
# Test Linex MCP
echo '{"method":"tools/list"}' | python mcp/linex/server.py

# Test with actual profiling
python mcp/linex/server.py
# Then send MCP messages via stdin
```

## Requirements

- Python >= 3.10
- `mcp` Python package
- IntelliKit tools installed (`pip install -e .`)
- ROCm environment configured

## Troubleshooting

**MCP servers not showing up in Cursor:**
1. Check config path: `~/.cursor/mcp.json` or `~/.config/cursor/mcp.json`
2. Verify paths in config are absolute
3. Restart Cursor completely
4. Check Cursor's developer console for errors

**Import errors in MCP servers:**
1. Ensure IntelliKit is installed: `pip install -e .`
2. Check Python path in MCP config matches your environment
3. Test server manually: `python mcp/linex/server.py`

**Permission errors:**
1. Make servers executable: `chmod +x mcp/*/server.py`
2. Ensure Python has access to ROCm libraries

## Architecture

```
mcp/
├── linex/server.py       # Linex MCP server
├── nexus/server.py       # Nexus MCP server  
├── metrix/server.py      # Metrix MCP server
├── accordo/server.py     # Accordo MCP server
├── setup_cursor.py       # Automatic configuration helper
└── README.md            # This file
```

Each server:
1. Exposes tool functionality via MCP protocol
2. Handles JSON-based requests/responses
3. Returns structured data for LLM consumption
4. Runs as a subprocess managed by Cursor

## Example Interaction

User asks: *"What line in my kernel is taking the most cycles?"*

1. Cursor identifies this needs profiling data
2. Discovers `intellikit-linex` MCP server
3. Calls `profile_application` tool
4. Receives JSON with source-level cycle counts
5. Formats and presents the answer to user

All automatic! 🎯
