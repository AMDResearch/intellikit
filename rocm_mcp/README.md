# ROCm MCP Servers

A collection of Model Context Protocol (MCP) servers for interacting with the AMD ROCm™ ecosystem. This package provides tools for LLMs to compile HIP code, access documentation, and query system information.

## Components

### 1. HIP Compiler (`hip-compiler-mcp`)

Tool for compiling HIP C/C++ code into binary executables using `hipcc` compiler.

### 2. HIP Documentation (`hip-docs-mcp`)

Provides access to the official HIP language and runtime developer reference documentation.

### 3. ROCm System Info (`rocminfo-mcp`)

Exposes system topology and device information via the `rocminfo` utility.

## Installation

You can install the package directly using `uv` or `pip`.

```bash
# Using uv (recommended)
uv pip install .

# Using pip
pip install .
```

## Configuration

To use these servers, add the following to your configuration file:

```json
{
  "mcpServers": {
    "hip-compiler-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/rocm_mcp", "hip-compiler-mcp"]
    },
    "hip-docs-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/rocm_mcp", "hip-docs-mcp"]
    },
    "rocminfo-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/rocm_mcp", "rocminfo-mcp"]
    }
  }
}
```

*Note: Adjust `/path/to/rocm_mcp` to the actual path where you have cloned or installed the package.*

## Development

This project uses `uv` for dependency management.

1. **Sync dependencies:**

   ```bash
   uv sync --dev
   ```

2. **Run a server locally (for testing):**

   ```bash
   uv run ./examples/hip_compiler.py
   ```

3. **Run tests:**

   ```bash
   pytest
   ```

## Use as a Claude Code plugin

`rocm-mcp` ships as a plugin in the [IntelliKit marketplace](../README.md#quick-start). The plugin registers all four MCP servers (`hip-compiler`, `hip-docs`, `amd-smi`, `rocminfo`) in a single install — no skill bundled.

```bash
# In Claude Code
/plugin marketplace add AMDResearch/intellikit
/plugin install rocm-mcp@intellikit
```

Host requirements when installed as a plugin:

- [`uv`](https://docs.astral.sh/uv/) on `PATH` (each MCP launches via `uv --directory ${CLAUDE_PLUGIN_ROOT} run <name>-mcp`)
- ROCm with the `amdsmi` Python bindings installable from the ROCm tree (for `amd-smi`), `rocminfo` on `PATH` (for `rocminfo`), and `hipcc` on `PATH` (for `hip-compiler`)
- Network access (for `hip-docs`, which scrapes the official HIP reference)

If you don't need all four, disable individual servers by editing the installed `.mcp.json`, or use the `claude plugin` CLI to remove the plugin and install only what you need from upstream.
