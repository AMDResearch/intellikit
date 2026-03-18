# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

IntelliKit is a monorepo of LLM-ready GPU profiling and analysis tools for AMD ROCm. It provides clean Python abstractions over complex GPU internals with MCP (Model Context Protocol) server support for LLM integration.

**Requirements:** Python >= 3.10 (root), ROCm >= 6.0 (7.0+ for linex), MI300+ GPUs. Note: Individual tools may support older Python versions (check each tool's `pyproject.toml`).

## Tool Descriptions

| Tool | Purpose | Key Use Case |
| ------ | --------- | -------------- |
| **accordo** | Kernel validation | Verify optimized GPU kernels match reference implementations |
| **linex** | Source-line profiling | Map cycle-level timing and stall analysis to source code lines |
| **metrix** | Hardware counter metrics | Profile GPU kernels with human-readable performance insights |
| **nexus** | HSA packet interception | Capture GPU kernel launches and memory operations |
| **rocm_mcp** | ROCm MCP servers | LLM-accessible HIP compilation, docs, and system info |
| **uprof_mcp** | uProf MCP server | LLM-accessible AMD uProf profiling |

## Build Commands

```bash
# Using uv (preferred for development) - installs all tools
uv sync

# Or using pip - install individual tools (from repo root)
# Note: root pyproject.toml only includes accordo, linex, metrix, nexus as dependencies
pip install -e accordo/ -e linex/ -e metrix/ -e nexus/ -e rocm_mcp/ -e uprof_mcp/

# Install individual tools
pip install -e metrix/
pip install -e linex/

# Build nexus C++ component (scikit-build-core handles CMake automatically)
pip install -e nexus/
# Or build manually if needed:
# cd nexus && mkdir -p build && cd build && cmake .. && make
```

## Testing

All main tools now have `tests/` directories with pytest-based test suites:

- **metrix**: `tests/unit/` and `tests/integration/` with pytest markers (unit, integration, e2e, slow)
- **accordo**: `tests/` directory
- **nexus**: `tests/` directory
- **linex**: `tests/` directory
- **rocm_mcp**: `tests/` directory

All tools also have `examples/` directories for usage demonstrations.

```bash
# Run tests for individual tools
cd metrix && pytest
cd accordo && pytest
cd nexus && pytest
cd linex && pytest

# Run specific test file
pytest metrix/tests/unit/test_api.py

# Run metrix tests by marker (defined in metrix/pytest.ini)
pytest -m unit      # Fast unit tests
pytest -m integration  # Requires GPU/rocprof
pytest -m e2e      # End-to-end tests (require GPU and benchmarks)
pytest -m slow     # Slow tests (> 5s)
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
- MCP servers: `accordo-mcp`, `linex-mcp`, `metrix-mcp`, `nexus-mcp`, `hip-compiler-mcp`, `hip-docs-mcp`, `rocminfo-mcp`, `uprof-profiler-mcp`

### Nexus C++ Integration

- C++ source in `nexus/csrc/src/` (`.cpp` files)
- Headers in `nexus/csrc/include/nexus/` (`.hpp` files: `nexus.hpp`, `log.hpp`)
- Python bindings via shared library built with CMake
- Requires LLVM from ROCm (`LLVM_INSTALL_DIR=/opt/rocm/llvm`)

### Accordo Runtime Compilation

- C++ validation code in `accordo/src/` compiled at runtime (`.hip` and `.hpp` files)
- Uses HSA for GPU memory interception
- Python package in `accordo/accordo/` with validator implementation
- Internal utilities in `accordo/accordo/_internal/` (inside main package) for IPC and other internals
- Dependencies include external `kerneldb` library for kernel extraction (pinned to specific commit in pyproject.toml)

## Package Layout Variations

Not all tools follow the same directory structure:

| Tool | Layout | Package Location | C++ Source | Tests | Skills |
|------|--------|------------------|------------|-------|--------|
| **metrix** | `src/` layout | `metrix/src/metrix/` | N/A | `metrix/tests/` | `metrix/skill/` |
| **linex** | `src/` layout | `linex/src/linex/` | N/A | `linex/tests/` | `linex/skill/` |
| **rocm_mcp** | `src/` layout | `rocm_mcp/src/rocm_mcp/` | N/A | `rocm_mcp/tests/` | N/A |
| **uprof_mcp** | `src/` layout | `uprof_mcp/src/uprof_mcp/` | N/A | `uprof_mcp/tests/` | N/A |
| **accordo** | flat layout | `accordo/accordo/` | `accordo/src/` (runtime compiled) | `accordo/tests/` | `accordo/skill/` |
| **nexus** | flat layout | `nexus/nexus/` | `nexus/csrc/` (CMake built) | `nexus/tests/` | `nexus/skill/` |

This affects import paths and where to find source code. Each main tool (accordo, linex, metrix, nexus) has a `skill/` directory containing `SKILL.md` files for AI agent integration.

## Dependency Management

- **uv**: Preferred for development (lockfile: `uv.lock`)
- **pip**: Supported for individual tool installation
- **External dependencies**: Some tools depend on external repos (e.g., `accordo` requires `kerneldb` from GitHub)
- **C++ dependencies**: `nexus` requires LLVM from ROCm (`LLVM_INSTALL_DIR=/opt/rocm/llvm`)

## CI/CD and Development Environment

### GitHub Actions CI

- **Runners**: Self-hosted with MI300+ GPUs
- **Container**: Uses Apptainer for containerized testing
- **Installation testing** (`intellikit-ci-test.yml`): Tests three installation methods for each tool:
  - Editable install: `pip install -e <tool>/`
  - Non-editable install: `pip install ./<tool>/`
  - GitHub install: `pip install 'git+https://github.com/AMDResearch/intellikit.git@<sha>#subdirectory=<tool>'`
- **Pytest testing** (`intellikit-pytest.yml`): Runs `pytest` for accordo, metrix, nexus, and linex

### Linting Enforcement

- **Auto-fix enabled**: CI runs `ruff check --fix` and `ruff format`
- **Strict enforcement**: PRs fail if formatting changes are needed
- **Pre-commit**: Run `ruff check --fix && ruff format` before committing

### Container Scripts

- `.github/scripts/container_build.sh`: Builds Apptainer container
- `.github/scripts/container_exec.sh`: Executes commands inside container

### Install Scripts

The repository includes install scripts for both tools and agent skills:

- `install/tools/install.sh`: Installs all IntelliKit tools from GitHub (supports editable/non-editable, custom pip commands, specific branches/tags)
- `install/skills/install.sh`: Installs agent skills (SKILL.md files) for AI agents (supports multiple targets: agents, codex, cursor, claude, github; global or local installation)

## MCP Server Development

### Server Entry Points

All MCP servers are defined in each tool's `pyproject.toml` under `[project.scripts]`:

```toml
[project.scripts]
accordo-mcp = "accordo.mcp.server:main"
linex-mcp = "linex.mcp.server:main"
metrix-mcp = "metrix.mcp.server:main"
nexus-mcp = "nexus.mcp.server:main"
hip-compiler-mcp = "rocm_mcp.compile.hip_compiler_mcp:main"
hip-docs-mcp = "rocm_mcp.doc.hip_docs_mcp:main"
rocminfo-mcp = "rocm_mcp.sysinfo.rocminfo_mcp:main"
uprof-profiler-mcp = "uprof_mcp.uprof_profiler_mcp:main"
```

### MCP Transport Options

All MCP servers support two transport modes via CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--transport` | `stdio` | Transport type: `stdio` or `http` |
| `--host` | `127.0.0.1` | HTTP server host (only for http transport) |
| `--port` | `8000` | HTTP server port (only for http transport) |
| `--path` | (varies) | HTTP endpoint path (only for http transport) |

Default HTTP paths per server:

| Server | Default Path |
|--------|--------------|
| `linex-mcp` | `/linex` |
| `metrix-mcp` | `/metrix` |
| `nexus-mcp` | `/nexus` |
| `uprof-profiler-mcp` | `/uprof_mcp` |
| `hip-compiler-mcp` | `/rocm_mcp/hip_compiler` |
| `hip-docs-mcp` | `/rocm_mcp/hip_docs` |
| `rocminfo-mcp` | `/rocm_mcp/rocminfo` |

### Testing MCP Servers Locally

```bash
# Install the tool first
pip install -e metrix/

# Run MCP server with stdio transport (default)
metrix-mcp

# Run MCP server with streamable-http transport
metrix-mcp --transport http --port 8001

# Or via uv
cd metrix && uv run metrix-mcp
cd metrix && uv run metrix-mcp --transport http --port 8001
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

# Run all tests for a specific tool
cd accordo && pytest -v
cd linex && pytest -v
cd nexus && pytest -v
```

### Working with Agent Skills

Each main tool has a `skill/` directory with `SKILL.md` files for AI agent integration:

```bash
# Skills are located at:
accordo/skill/SKILL.md
linex/skill/SKILL.md
metrix/skill/SKILL.md
nexus/skill/SKILL.md

# Install skills for AI agents using the install script
./install/skills/install.sh                    # local: ./.agents/skills/
./install/skills/install.sh --target cursor     # local: ./.cursor/skills/
./install/skills/install.sh --target claude --global  # global: ~/.claude/skills/
./install/skills/install.sh --target github     # local: ./.github/agents/skills/
```

## Key Design Principles for AI Agents

### LLM-First Design

This project is built for LLM consumption:

- Clean, human-readable APIs (not raw hardware counters)
- MCP servers for all tools
- Examples in every tool's `examples/` directory
- Agent skills in `skill/SKILL.md` files for AI agent discovery
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
- Shared root-level ruff configuration
