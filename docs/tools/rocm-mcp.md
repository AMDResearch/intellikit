# ROCm MCP

ROCm MCP is a collection of Model Context Protocol (MCP) servers for interacting with the AMD ROCm ecosystem. It provides tools for LLMs to compile HIP code, access documentation, and query system information.

## Components

### HIP Compiler (`hip-compiler-mcp`)

A tool for compiling HIP C/C++ code into binary executables using the `hipcc` compiler.

### HIP Documentation (`hip-docs-mcp`)

Provides access to the official HIP language and runtime developer reference documentation.

### ROCm System Info (`rocminfo-mcp`)

Exposes system topology and device information via the `rocminfo` utility.

## Installation

```bash
# Using uv (recommended)
uv pip install .

# Using pip
pip install .
```

## Configuration

Add the following to your MCP client configuration:

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

Replace `/path/to/rocm_mcp` with the actual path where you have cloned or installed the package.

If you installed with `pip` or `install.sh`, use the console script names (`hip-compiler-mcp`, `hip-docs-mcp`, `rocminfo-mcp`) from your `PATH`, or the full path in your virtual environment.

## Development

This project uses `uv` for dependency management.

```bash
# Sync dependencies
uv sync --dev

# Run a server locally (for testing)
uv run ./examples/hip_compiler.py

# Run tests
pytest
```

See [Set up MCP](../how-to/mcp-setup.md) for information about multi-server configuration.
