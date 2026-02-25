# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

IntelliKit is a monorepo of LLM-ready GPU profiling and analysis tools for AMD ROCm. It provides clean Python abstractions over complex GPU internals with MCP (Model Context Protocol) server support for LLM integration.

**Requirements:** Python >= 3.10, ROCm >= 6.0 (7.0+ for linex), MI300+ GPUs

## Build Commands

```bash
# Using uv (preferred for development)
uv sync

# Or using pip - install all dependencies (from repo root)
pip install -e accordo/ -e linex/ -e metrix/ -e nexus/ -e rocm_mcp/ -e uprof_mcp/

# Install individual tools
pip install -e metrix/
pip install -e linex/

# Build nexus C++ component (requires CMake)
cd nexus && mkdir -p build && cd build && cmake .. && make
```

## Testing

Test structure varies by tool:

- **metrix**: `tests/unit/` and `tests/integration/` with pytest markers (unit, integration, e2e, slow)
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

# Run rocm_mcp tests
cd rocm_mcp && pytest tests/
```

## Linting

```bash
# Lint entire repo (ruff configured in root pyproject.toml)
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
    total = TCC_HIT_sum + TCC_MISS_sum
    return (TCC_HIT_sum / total) * 100 if total else 0.0
```

### MCP Server Pattern

All tools expose MCP servers via FastMCP:

- Entry points defined in `pyproject.toml` `[project.scripts]`
- Server implementations in `<tool>/mcp/server.py` or `<tool>_mcp.py`
- Tools: `metrix-mcp`, `linex-mcp`, `nexus-mcp`, `accordo-mcp`, `hip-compiler-mcp`, `hip-docs-mcp`, `rocminfo-mcp`, `uprof-profiler-mcp`

### Nexus C++ Integration

- C++ source in `nexus/csrc/` (headers in `include/nexus/`)
- Python bindings via shared library built with CMake
- Requires LLVM from ROCm (`LLVM_INSTALL_DIR=/opt/rocm/llvm`)

### Accordo Runtime Compilation

- C++ validation code in `accordo/src/` compiled at runtime
- Uses HSA for GPU memory interception
- Python package in `accordo/accordo/` with validator implementation
- Dependencies include external `kerneldb` library for kernel extraction
