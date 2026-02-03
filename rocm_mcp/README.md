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
