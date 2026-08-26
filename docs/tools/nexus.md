---
myst:
    html_meta:
        "description": "Nexus intercepts HSA packets to extract GPU kernel assembly and HIP source from AMD GPU applications, enabling detailed GPU execution inspection with ROCm."
        "keywords": "Nexus, AMD GPU, ROCm, HSA, AQL, GPU assembly, HIP source, kernel inspection"
---

# Nexus

Nexus intercepts GPU AQL (Architected Queuing Language) packets within the Heterogeneous System Architecture (HSA), extracts source code, and outputs both assembly and HIP code in a structured format.

## Installation

Nexus compiles a native C++ library during installation. Ensure `cmake`, `libdwarf-dev`, and `libzstd-dev` are installed beforehand:

```bash
# System prerequisites (Debian/Ubuntu)
sudo apt-get update && sudo apt-get install -y cmake libdwarf-dev libzstd-dev

# From Git
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=nexus"

# Editable install for development
git clone https://github.com/AMDResearch/intellikit.git
cd intellikit
pip install -e ./nexus
```

### Manual C++ build (optional)

If you need to build the C++ library manually, use the following commands.

```bash
cmake -B build \
    -DCMAKE_PREFIX_PATH=${ROCM_PATH} \
    -DLLVM_INSTALL_DIR=/opt/rocm/llvm \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel 16
```

## Usage

Nexus can be used through the Python API or by setting environment variables directly.

### Python API

```python
from nexus import Nexus

# Create tracer
nexus = Nexus(log_level=1)

# Run and get trace
trace = nexus.run(["python", "my_gpu_script.py"])

# Iterate over kernels
for kernel in trace:
    print(f"{kernel.name}: {len(kernel.assembly)} instructions")

    # Assembly instructions
    for i, asm_line in enumerate(kernel.assembly, 1):
        print(f"  {i:3d}. {asm_line}")

    # HIP source with line numbers
    if kernel.lines and len(kernel.lines) == len(kernel.hip):
        for line_no, hip_line in zip(kernel.lines, kernel.hip):
            print(f"  {line_no:3d}. {hip_line}")

# Access specific kernel by name
kernel = trace["vector_add(float const*, float const*, float*, int)"]
print(kernel.assembly)   # Array of assembly strings
print(kernel.hip)        # Array of HIP source lines
print(kernel.signature)  # Kernel signature
print(kernel.files)      # Source files
print(kernel.lines)      # Line numbers

# Save and load traces
trace.save("my_trace.json")
old_trace = Nexus.load("my_trace.json")
```

### Environment variables

Use environment variables to run Nexus directly using `HSA_TOOLS_LIB`:

```bash
export HSA_TOOLS_LIB=/path/to/libnexus.so  # installed into site-packages/nexus/
export NEXUS_LOG_LEVEL=3
export NEXUS_OUTPUT_FILE=result.json

hipcc vector_add.hip -g -o vector_add   # -g needed for source line mapping
./vector_add
```

| Variable | Description |
|----------|-------------|
| `NEXUS_LOG_LEVEL` | Controls log verbosity (`0` = none, `1` = info, `2` = warning, `3` = error, `4` = detail) |
| `NEXUS_OUTPUT_FILE` | File path for Nexus JSON output |
| `NEXUS_EXTRA_SEARCH_PREFIX` | Additional search directories for HIP files. Supports wildcards, colon-separated paths |
| `TRITON_DISABLE_LINE_INFO` | Set to `0` to enable line information in Triton kernels. Automatically managed by the Python API |

Without the `-g` flag, you still get assembly and HIP source in the trace. However, the mapping to original source line numbers will be unavailable.

## Output format

Nexus produces a JSON file with kernel data:

```json
{
  "kernels": {
    "vector_add(float const*, float const*, float*, int)": {
      "assembly": [
        "global_load_dword v6, v[4:5], off",
        "global_load_dword v7, v[2:3], off",
        "global_store_dword v[0:1], v2, off"
      ],
      "hip": [
        "    c[idx] = a[idx] + b[idx];",
        "    c[idx] = a[idx] + b[idx];",
        "    c[idx] = a[idx] + b[idx];"
      ],
      "files": ["vector_add.hip", "..."],
      "lines": [11, 11, 11],
      "signature": "vector_add(float const*, float const*, float*, int)"
    }
  }
}
```

Each kernel entry contains:
- **assembly:** ISA instructions extracted from the code object
- **hip:** corresponding HIP source lines (when available)
- **files:** source file paths for each instruction
- **lines:** source line numbers for each instruction
- **signature:** the full kernel signature
