---
myst:
    html_meta:
        "description": "uProf MCP is an MCP server that enables LLMs to profile x86 CPU applications using AMD uProf, identifying hotspot functions consuming CPU time."
        "keywords": "uProf MCP, AMD uProf, CPU profiling, MCP server, LLM, hotspot analysis, x86"
---

# uProf MCP (IntelliKit)

uProf MCP is a Model Context Protocol (MCP) server for using AMD uProf™ to profile x86 CPU applications. It enables LLMs to analyze CPU performance hotspots through the AMD uProf profiler.

## Features

uProf MCP provides the following capabilities.

- Profile CPU applications for hotspot analysis
- Identify top functions consuming CPU time
- Generate detailed profiling reports
- Support for custom executable arguments

## Requirements

uProf MCP requires the following.

- Python 3.10+
- AMD uProf
- x86 CPU architecture

## Installation

Run the following commands from the `uprof_mcp` directory:

```bash
# Using uv (recommended)
uv pip install .

# Using pip
pip install .
```

## Setting the uProf Path

uProf MCP needs to know where AMD uProf is installed. There are three ways to set this, in order of precedence:

1. **Environment variable** `INTELLIKIT_UPROF_CLI`:
   ```bash
   export INTELLIKIT_UPROF_CLI=/opt/AMDuProf_5.3-521/bin/AMDuProfCLI
   ```

2. **Constructor argument** (Python API only):
   ```python
   profiler = UProfProfiler(uprof="/opt/AMDuProf_5.3-521/bin/AMDuProfCLI")
   ```

3. **Default path**: `/opt/AMDuProf_5.1-701/bin/AMDuProfCLI`

For MCP server usage, set the environment variable in the MCP client configuration (see below).

## Configuration

Add the following to your MCP client configuration:

```json
{
  "mcpServers": {
    "uprof-profiler-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/uprof_mcp", "uprof-profiler-mcp"],
      "env": {
        "INTELLIKIT_UPROF_CLI": "/opt/AMDuProf_5.3-521/bin/AMDuProfCLI"
      }
    }
  }
}
```

Replace `/path/to/uprof_mcp` with the actual path where you have cloned or installed the package, and update the uProf path to match your installation.

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

Run the following commands from the `uprof_mcp` directory:

```bash
# Agentic mode (with LLM)
python examples/uprof_profiler.py --executable ./my_app --args arg1 arg2

# Non-agentic mode (direct profiling)
python examples/uprof_profiler.py --executable ./my_app --args arg1 arg2 --classic
```

## API reference

The following describes the uProf MCP Python API for non-agentic use.

### UProfProfiler class

```python
from uprof_mcp.uprof_profiler import UProfProfiler

profiler = UProfProfiler(logger=None, uprof=None)
```

**Constructor parameters:**

- `logger` (logging.Logger | None): Logger instance. If None, a default logger is created.
- `uprof` (str | PathLike | None): Path to the uProf CLI executable. If None, uses the `INTELLIKIT_UPROF_CLI` environment variable, or falls back to the default path.

**Methods:**

- `find_hotspots(output_dir, executable, executable_args)` → `UProfProfilerResult`

  **Parameters:**

  - `output_dir` (str | Path): directory to store results
  - `executable` (str | Path): path to executable
  - `executable_args` (list[str] | None): arguments for the executable

  **Returns:**
  
  - `UProfProfilerResult` with a `report_path` attribute

## Development

Use the following commands for local development and testing.

```bash
# Sync dependencies
uv sync --dev

# Run the server locally
uv run uprof-profiler-mcp

# Run tests
pytest
```
