# Nexus Scripts

This directory contains scripts for building and running Nexus in different modes.

## Scripts Overview

### `rebuild.sh`
Builds the native C++ library (`libnexus.so`) from source.

**Usage:**
```bash
./scripts/rebuild.sh
```

**Output:** `build/lib/libnexus.so`

---

### `run.sh`
Runs test binaries with Nexus tracing using direct C++ library mode (environment variables).

**Usage:**
```bash
./scripts/run.sh
```

Edit the script to select which test kernel to run (default: `vector_add`).

**Environment Variables Set:**
- `HSA_TOOLS_LIB` → points to `build/lib/libnexus.so`
- `NEXUS_OUTPUT_FILE` → output JSON file path
- `NEXUS_LOG_LEVEL` → logging verbosity (0-4)

---

### `nexus`
Wrapper script for running any application with Nexus tracing.

**Usage:**
```bash
./scripts/nexus [options] <command...>
```

**Options:**
- `-v[v[v[v]]]` : Set log level (more v's = more verbose)
- `-o, --output <file>` : Output JSON file path
- `-s, --search-prefix <prefix>` : Additional search directories for HIP source files
- `-f, --full-trace-dump-file <file>` : Full trace dump file
- `-g, --gdb` : Run with GDB
- `-h, --help` : Show help

**Examples:**
```bash
# Basic usage
./scripts/nexus python test/my_script.py

# With verbosity and output file
./scripts/nexus -vv -o trace.json ./test/vector_add

# With source search path
./scripts/nexus -s './test' python app.py
```

---

## Usage Modes

Nexus supports **two usage modes**:

### Mode 1: Python API (Recommended)
Use the Python package for programmatic tracing.

**Installation:**
```bash
pip install -e .
```

**Usage:**
```python
from nexus import Nexus

nexus = Nexus(log_level=1)
trace = nexus.run(["python", "my_app.py"])

# Iterate over kernels
for kernel in trace:
    # Iterate assembly with line numbers
    for i, asm_line in enumerate(kernel.assembly, 1):
        print(f"  {i:3d}. {asm_line}")

    # Iterate HIP source with actual source line numbers
    if kernel.lines and len(kernel.lines) == len(kernel.hip):
        for line_no, hip_line in zip(kernel.lines, kernel.hip):
            print(f"  {line_no:3d}. {hip_line}")

# Save trace
trace.save("output.json")
```

### Mode 2: Direct C++ Library (Environment Variables)
Use environment variables to inject Nexus into any application.

**Method A: Using the `nexus` wrapper script:**
```bash
./scripts/nexus -vv -o output.json ./my_app
```

**Method B: Setting environment variables manually:**
```bash
export HSA_TOOLS_LIB=$PWD/build/lib/libnexus.so
export NEXUS_OUTPUT_FILE=output.json
export NEXUS_LOG_LEVEL=1
./my_app
```

---

## Building and Testing

1. **Build the library:**
   ```bash
   ./scripts/rebuild.sh
   ```

2. **Run tests (C++ library mode):**
   ```bash
   ./scripts/run.sh
   ```

3. **Run examples (Python API mode):**
   ```bash
   pip install -e .
   python examples/simple_hip.py
   python examples/simple_triton.py
   ```

---

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `HSA_TOOLS_LIB` | Path to libnexus.so | (required) |
| `NEXUS_OUTPUT_FILE` | Output JSON file path | `result.json` |
| `NEXUS_LOG_LEVEL` | Log level (0=none, 1=info, 2=warn, 3=error, 4=detail) | `0` |
| `NEXUS_EXTRA_SEARCH_PREFIX` | Colon-separated search paths for HIP source files | (empty) |
| `NEXUS_KERNELS_DUMP_FILE` | Full trace dump file | (empty) |
| `TRITON_ALWAYS_COMPILE` | Force Triton recompilation (automatically set) | `1` |
| `TRITON_DISABLE_LINE_INFO` | Enable line info in Triton (0=enabled, automatically set) | `0` |

---

## Output JSON Structure

```json
{
  "kernels": {
    "kernel_name(signature)": {
      "assembly": ["instruction1", "instruction2", ...],
      "hip": ["source_line1", "source_line2", ...],
      "files": ["path/to/file.cpp", ...],
      "lines": [line_number1, line_number2, ...],
      "signature": "full_function_signature"
    }
  }
}
```

