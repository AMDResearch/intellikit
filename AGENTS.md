# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

IntelliKit is a monorepo of LLM-ready GPU profiling and analysis tools for AMD ROCm. It provides clean Python abstractions over complex GPU internals with MCP (Model Context Protocol) server support for LLM integration.

**Requirements:** Python >= 3.10, ROCm >= 6.0 (7.0+ for linex), MI300+ GPUs. Note: Individual tools may support older Python versions (check each tool's `pyproject.toml`).

## Build Commands

```bash
# Install all tools from Git (supported path for users and CI-style setups)
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash
# Subset only: ... | bash -s -- --tools metrix,linex
# From a clone:
#   ./install/tools/install.sh [--tools ...] [--pip-cmd ...] [--repo-url ...] [--ref ...] [--dry-run]

# Editable installs for development (any subset; from repo root)
pip install -e accordo/ -e kerncap/ -e linex/ -e metrix/ -e nexus/ -e rocm_mcp/ -e uprof_mcp/

# Install individual tools
pip install -e metrix/
pip install -e linex/

# Build nexus C++ component (scikit-build-core handles CMake automatically)
pip install -e nexus/
# Or build manually if needed:
# cd nexus && mkdir -p build && cd build && cmake .. && make

# Build kerncap (scikit-build-core builds libkerncap.so + kerncap-replay)
pip install -e kerncap/
```

## Testing

Test structure varies by tool:

- **metrix**: `tests/unit/` and `tests/integration/` with pytest markers (unit, integration, e2e, slow)
- **kerncap**: `tests/unit/` and `tests/integration/` with pytest markers (docker, gpu)
- **rocm_mcp**: `tests/` directory
- Other tools have `examples/` directories for usage demonstrations

```bash
# Run metrix tests (has most comprehensive test suite)
cd metrix && pytest

# Run specific test file
pytest metrix/tests/unit/test_api.py

# Run by marker (defined in metrix/pytest.ini)
pytest -m unit      # Fast unit tests
pytest -m integration  # Requires GPU/rocprof
pytest -m e2e      # End-to-end tests (require GPU and benchmarks)
pytest -m slow     # Slow tests (> 5s)

# Run kerncap unit tests (no GPU required)
cd kerncap && pytest tests/unit/

# Run rocm_mcp tests
cd rocm_mcp && pytest tests/
```

## Linting

```bash
# Lint entire repo (shared config: ruff.toml; packages may extend and override)
ruff check .
ruff format .

# Lint specific tool
ruff check metrix/
```

## Architecture

### Monorepo Structure

Each tool is a standalone Python package with its own `pyproject.toml`:

| Tool | Build System | Description |
| ------ | -------------- | ------------- |
| **accordo** | scikit-build-core (CMake) | GPU kernel validation, C++ compiled at runtime |
| **kerncap** | scikit-build-core (CMake) | Kernel extraction and isolation, C++ HSA interception |
| **linex** | setuptools | Source-level SQTT profiling (`src/` layout) |
| **metrix** | setuptools | Hardware counter profiling (`src/` layout) |
| **nexus** | scikit-build-core (CMake) | HSA packet interception, C++ shared library |
| **rocm_mcp** | setuptools | MCP servers for ROCm tools (`src/` layout) |
| **uprof_mcp** | setuptools | MCP server for uProf (`src/` layout) |

### Metrix Backend System

Metrix uses a decorator-based architecture for GPU hardware counter metrics:

- `backends/base.py`: Abstract `CounterBackend` with profiling orchestration
- `backends/decorator.py`: `@metric` decorator auto-discovers counter requirements from function parameter names
- `backends/gfx942.py`, `gfx90a.py`, etc.: Architecture-specific implementations

Counter names appear exactly once as function parameters - no mapping tables:

```python
@metric("memory.l2_hit_rate")
def _l2_hit_rate(self, TCC_HIT_sum, TCC_MISS_sum):
    """
    L2 cache hit rate as percentage

    Formula: (hits / (hits + misses)) * 100
    """
    total = TCC_HIT_sum + TCC_MISS_sum
    return (TCC_HIT_sum / total) * 100 if total > 0 else 0.0
```

### MCP Server Pattern

All tools expose MCP servers via the MCP SDK's FastMCP module:

- Entry points defined in `pyproject.toml` `[project.scripts]`
- Server implementations in `<tool>/mcp/server.py` or `<tool>_mcp.py`
- MCP servers: `accordo-mcp`, `kerncap-mcp`, `linex-mcp`, `metrix-mcp`, `nexus-mcp`, `hip-compiler-mcp`, `hip-docs-mcp`, `rocminfo-mcp`, `uprof-profiler-mcp`

### Nexus C++ Integration

- C++ source in `nexus/csrc/src/` (`.cpp` files)
- Headers in `nexus/csrc/include/nexus/` (`.hpp` files: `nexus.hpp`, `log.hpp`)
- Python bindings via shared library built with CMake
- Requires LLVM from ROCm (`LLVM_INSTALL_DIR=/opt/rocm/llvm`)

### Kerncap C++ Integration

- C++ source in `kerncap/src/` (`.hip`, `.cpp` files)
- Headers in `kerncap/src/` (`.hpp` files: `kerncap.hpp`, `kerncap_log.hpp`)
- `libkerncap.so`: HSA tool library loaded via `HSA_TOOLS_LIB` for kernel capture
- `kerncap-replay`: VA-faithful HSA kernel replay binary
- Vendored nlohmann/json in `kerncap/vendor/`
- Built with scikit-build-core (CMake + HIP language support)

### Accordo Runtime Compilation

- C++ validation code in `accordo/src/` compiled at runtime (`.hip` and `.hpp` files)
- Uses HSA for GPU memory interception
- Python package in `accordo/accordo/` with validator implementation
- Internal utilities in `accordo/_internal/` (outside main package)
- Dependencies include external `kerneldb` library for kernel extraction

## Package Layout Variations

Not all tools follow the same directory structure:

| Tool | Layout | Package Location |
|------|--------|------------------|
| **metrix** | `src/` layout | `metrix/src/metrix/` |
| **linex** | `src/` layout | `linex/src/linex/` |
| **rocm_mcp** | `src/` layout | `rocm_mcp/src/rocm_mcp/` |
| **uprof_mcp** | `src/` layout | `uprof_mcp/src/uprof_mcp/` |
| **accordo** | flat layout | `accordo/accordo/` |
| **kerncap** | flat layout | `kerncap/kerncap/` |
| **nexus** | flat layout | `nexus/nexus/` |

This affects import paths and where to find source code.

## Dependency Management

- **install.sh**: Installs packages from Git via `pip` (`install/tools/install.sh`); default is all tools, optional `--tools` for a subset
- **pip**: Editable installs per tool from a clone (`pip install -e <tool>/`)
- **External dependencies**: Some tools depend on external repos (e.g., `accordo` requires `kerneldb` from GitHub)
- **C++ dependencies**: `nexus` requires LLVM from ROCm (`LLVM_INSTALL_DIR=/opt/rocm/llvm`); `kerncap` requires `hipcc`, `cmake`, and HSA headers (standard ROCm)

## CI/CD and Development Environment

### GitHub Actions CI

- **Runners**: Self-hosted with MI300+ GPUs
- **Container**: Uses Apptainer for containerized testing
- **Python versions**: Tests run on 3.10, 3.11, and 3.12
- **Installation testing**: Each tool is installed individually to verify isolated installation works

### Linting Enforcement

- **Auto-fix enabled**: CI runs `ruff check --fix` and `ruff format`
- **Strict enforcement**: PRs fail if formatting changes are needed
- **Pre-commit**: Run `ruff check --fix && ruff format` before committing

### Container Scripts

- `.github/scripts/container_build.sh`: Builds Apptainer container
- `.github/scripts/container_exec.sh`: Executes commands inside container

## MCP Server Development

### Server Entry Points

All MCP servers are defined in each tool's `pyproject.toml` under `[project.scripts]`:

```toml
[project.scripts]
accordo-mcp = "accordo.mcp.server:main"
kerncap-mcp = "kerncap.mcp.server:main"
linex-mcp = "linex.mcp.server:main"
metrix-mcp = "metrix.mcp.server:main"
nexus-mcp = "nexus.mcp.server:main"
hip-compiler-mcp = "rocm_mcp.compile.hip_compiler_mcp:main"
hip-docs-mcp = "rocm_mcp.doc.hip_docs_mcp:main"
rocminfo-mcp = "rocm_mcp.sysinfo.rocminfo_mcp:main"
uprof-profiler-mcp = "uprof_mcp.uprof_profiler_mcp:main"
```

### Testing MCP Servers Locally

```bash
# Install the tool first
pip install -e metrix/

# Run MCP server directly (will use stdio transport)
metrix-mcp

# Or from the package directory with uv
cd metrix && uv run metrix-mcp
```

### MCP Configuration Example

Add to your Claude Desktop or other MCP client config:

```json
{
  "mcpServers": {
    "metrix-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/intellikit/metrix", "metrix-mcp"]
    },
    "kerncap-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/intellikit/kerncap", "kerncap-mcp"]
    },
    "hip-compiler-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/intellikit/rocm_mcp", "hip-compiler-mcp"]
    }
  }
}
```

### MCP Server Implementation Pattern

All servers use the MCP SDK (`mcp[cli]` package) with the FastMCP module:

- Dependency: `mcp[cli]` in each tool's `pyproject.toml`
- Import: `from mcp.server.fastmcp import FastMCP`
- Server code in `<tool>/mcp/server.py` or `<tool>/<tool>_mcp.py`
- Use `@mcp.tool()` decorator for tool definitions
- Follow async patterns for I/O operations

## Common Development Workflows

### Adding a New Metric to Metrix

1. Identify the hardware counter(s) needed
2. Add a method to the appropriate backend (e.g., `metrix/src/metrix/backends/gfx942.py`)
3. Use `@metric("category.metric_name")` decorator
4. Counter names as function parameters (auto-discovery)
5. Add tests in `metrix/tests/unit/`

Example:

```python
@metric("memory.l2_hit_rate")
def _l2_hit_rate(self, TCC_HIT_sum, TCC_MISS_sum):
    """
    L2 cache hit rate as percentage

    Formula: (hits / (hits + misses)) * 100
    """
    total = TCC_HIT_sum + TCC_MISS_sum
    return (TCC_HIT_sum / total) * 100 if total > 0 else 0.0
```

### Working with C++ Components

**Nexus:**

```bash
cd nexus
mkdir -p build && cd build
cmake ..
make
cd ../..
pip install -e nexus/
```

**Kerncap:**

```bash
# scikit-build-core builds libkerncap.so + kerncap-replay automatically
pip install -e kerncap/
```

**Accordo:**

```bash
# Accordo compiles at runtime, just install
pip install -e accordo/
```

### Running Integration Tests

Integration tests require GPU and ROCm:

```bash
cd metrix
pytest -m integration  # Requires GPU/rocprof
pytest -m unit         # No GPU required
```

## Key Design Principles for AI Agents

### LLM-First Design

This project is built for LLM consumption:

- Clean, human-readable APIs (not raw hardware counters)
- MCP servers for all tools
- Examples in every tool's `examples/` directory
- Comprehensive docstrings and type hints

### No Mapping Tables

The decorator pattern in Metrix eliminates mapping tables:

- Counter names appear exactly once (as function parameters)
- The `@metric` decorator auto-discovers requirements
- No separate configuration files for metrics

### Modular Monorepo

Each tool can be installed independently:

- Separate `pyproject.toml` for each tool
- Individual testing and development
- Shared root-level `ruff.toml` (packages extend via `pyproject.toml`)
