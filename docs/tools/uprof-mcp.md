# uProf MCP

uProf MCP is a Model Context Protocol (MCP) server for using AMD uProf to profile x86 CPU applications. It enables LLMs to analyze CPU performance hotspots through the AMD uProf profiler.

## Features

- Profile CPU applications for hotspot analysis
- Identify top functions consuming CPU time
- Generate detailed profiling reports
- Support for custom executable arguments

## Requirements

- Python >= 3.10
- AMD uProf
- x86 CPU architecture

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
    "uprof-profiler-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/uprof_mcp", "uprof-profiler-mcp"]
    }
  }
}
```

Replace `/path/to/uprof_mcp` with the actual path where you have cloned or installed the package.

## Python API (non-agentic)

Use the profiler directly without MCP:

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

## LangChain example

```bash
# Agentic mode (with LLM)
python examples/uprof_profiler.py --executable ./my_app --args arg1 arg2

# Non-agentic mode (direct profiling)
python examples/uprof_profiler.py --executable ./my_app --args arg1 arg2 --classic
```

## API reference

### UProfProfiler class

```python
from uprof_mcp.uprof_profiler import UProfProfiler

profiler = UProfProfiler(logger=None)
```

**Methods:**

- `find_hotspots(output_dir, executable, executable_args)` → `UProfProfilerResult`

  **Parameters:**

  - `output_dir` (str | Path): directory to store results
  - `executable` (str | Path): path to executable
  - `executable_args` (list[str] | None): arguments for the executable

  **Returns:**
  
  - `UProfProfilerResult` with a `report_path` attribute

## Development

```bash
# Sync dependencies
uv sync --dev

# Run the server locally
uv run uprof-profiler-mcp

# Run tests
pytest
```
