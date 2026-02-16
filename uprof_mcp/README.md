# uProf MCP Server

A Model Context Protocol (MCP) server for profiling x86 CPU applications using AMD uProf. This package enables LLMs to analyze CPU performance hotspots through the AMD uProf profiler.

## Features

### CPU Hotspot Profiling (`uprof-profiler-mcp`)

Tool for profiling x86 CPU executables to identify performance hotspots using AMD uProf.

**Capabilities:**

- Profile CPU applications for hotspot analysis
- Identify top functions consuming CPU time
- Generate detailed profiling reports
- Support for custom executable arguments

## Installation

You can install the package directly using `uv` or `pip`.

```bash
# Using uv (recommended)
uv pip install .

# Using pip
pip install .
```

## Configuration

To use this server with an MCP client, add the following to your configuration file:

```json
{
  "mcpServers": {
    "uprof-profiler-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/uprof_mcp", "uprof-profiler-mcp"]
    }
  }
}
```

*Note: Adjust `/path/to/uprof_mcp` to the actual path where you have cloned or installed the package.*

## Usage

### Python API (Non-Agentic Mode)

You can use the profiler directly without MCP:

```python
import tempfile
from uprof_mcp.uprof_profiler import UProfProfiler

profiler = UProfProfiler()

with tempfile.TemporaryDirectory() as tmpdir:
    result = profiler.find_hotspots(
        output_dir=tmpdir,
        executable="./my_app",
        executable_args=["arg1", "arg2"],
    )

    with result.report_path.open() as report:
        print(report.read())
```

### MCP Tool (Agentic Mode)

When configured as an MCP server, the tool can be called by LLMs:

**Tool:** `profile_for_hotspots`

**Parameters:**

- `executable` (str): Path to the executable to profile
- `executable_arguments` (list[str]): Arguments to pass to the executable
- `output_dir` (str, optional): Directory to store profiling results

**Returns:** A summary report of the profiling results with CPU hotspots.

### Example with LangChain

See `examples/uprof_profiler.py` for a complete example using LangChain agents:

```bash
# Agentic mode (with LLM)
python examples/uprof_profiler.py --executable ./my_app --args arg1 arg2

# Non-agentic mode (direct profiling)
python examples/uprof_profiler.py --executable ./my_app --args arg1 arg2 --classic
```

## Requirements

- Python >= 3.10
- AMD uProf installed and available in PATH
- x86 CPU architecture

## Development

This project uses `uv` for dependency management.

1. **Sync dependencies:**

   ```bash
   uv sync --dev
   ```

2. **Run the server locally (for testing):**

   ```bash
   uv run uprof-profiler-mcp
   ```

3. **Run tests:**

   ```bash
   pytest
   ```

## API Reference

### UProfProfiler Class

```python
from uprof_mcp.uprof_profiler import UProfProfiler

profiler = UProfProfiler(logger=None)
```

**Methods:**

- `find_hotspots(output_dir, executable, executable_args)` → `UProfProfilerResult`
  - Profiles the executable and returns hotspot analysis
  - Parameters:
    - `output_dir` (str | Path): Directory to store results
    - `executable` (str | Path): Path to executable
    - `executable_args` (list[str]): Arguments for the executable
  - Returns: `UProfProfilerResult` with `report_path` attribute
