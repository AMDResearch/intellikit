<!--
MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# Nexus: HSA Packet Source Code Extractor

> [!IMPORTANT]
> This project is intended for research purposes only and is provided by AMD Research and Advanced Development team.
This is not a product. Use it at your own risk and discretion.
>
Nexus is a custom tool that intercepts Heterogeneous System Architecture (HSA) packets, extracts the source code from them, and outputs it to a JSON file containing the assembly and the HIP code.

## Installation

### System prerequisites

Nexus compiles a native C++ library during installation.
You need `cmake`, `libdwarf-dev`, and `libzstd-dev` installed first:

```bash
# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y cmake libdwarf-dev libzstd-dev

# Fedora / RHEL
sudo dnf install -y cmake libdwarf-devel libzstd-devel
```

### From Git Repository (Recommended)

```bash
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=nexus"
```

This will automatically build the native C++ library during installation.

For development/editable install:

```bash
git clone https://github.com/AMDResearch/intellikit.git
cd intellikit
pip install -e ./nexus
```

### Manual Build (Optional)

```bash
cmake -B build\
    -DCMAKE_PREFIX_PATH=${ROCM_PATH}\
    -DLLVM_INSTALL_DIR=/opt/rocm/llvm\
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel 16
```

## Usage

### Python API

After installation, you can use Nexus from Python:

```python
from nexus import Nexus

# Create tracer
nexus = Nexus(log_level=1)

# Run and get trace
trace = nexus.run(["python", "my_gpu_script.py"])

# Iterate over kernels
for kernel in trace:
    print(f"{kernel.name}: {len(kernel.assembly)} instructions")

    # Iterate through assembly instructions with line numbers
    for i, asm_line in enumerate(kernel.assembly, 1):
        print(f"  {i:3d}. {asm_line}")

    # Iterate through HIP source with actual source line numbers
    if kernel.lines and len(kernel.lines) == len(kernel.hip):
        for line_no, hip_line in zip(kernel.lines, kernel.hip):
            print(f"  {line_no:3d}. {hip_line}")
    else:
        # Fallback to sequential numbering if line numbers not available
        for i, hip_line in enumerate(kernel.hip, 1):
            print(f"  {i:3d}. {hip_line}")

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

For more details, see the [examples directory](examples/README.md) (HIP, Triton, and **PyTorch** in `06_pytorch_tensor_add` / `07_pytorch_matmul`).

### Command Line Usage

#### Environment Variables

* `NEXUS_LOG_LEVEL`: Verbosity level (0 = none, 1 = info, 2 = warning, 3 = error, 4 = detail)
* `NEXUS_OUTPUT_FILE`: Path to the JSON output file
* `NEXUS_EXTRA_SEARCH_PREFIX`: Additional search directories for HIP files with relative paths. Supports wildcards and is a colon-separated list.
* `TRITON_DISABLE_LINE_INFO`: Set to `0` to enable line info in Triton kernels (automatically set by Python API)

#### Example

To use Nexus, simply export the following environment variables and run your application:

```terminal
export HSA_TOOLS_LIB=/path/to/libnexus.so  # installed into site-packages/nexus/
export NEXUS_LOG_LEVEL=3
export NEXUS_OUTPUT_FILE=result.json

cd test/
hipcc vector_add.hip -g -o vector_add   # -g needed for source line mapping in trace
cd ..
./test/vector_add
```

Without `-g` you still get assembly and HIP source in the trace; only the mapping to original source line numbers may be missing.

<details><summary>JSON Output</summary>
<p>


```console
cat result.json
{
    "kernels": {
        "vector_add(float const*, float const*, float*, int)": {
            "assembly": [
                "global_load_dword v6, v[4:5], off   // 000000001E68: DC508000 ...",
                "global_load_dword v7, v[2:3], off   // 000000001E70: DC508000 ...",
                "global_store_dword v[0:1], v2, off  // 000000001E8C: DC708000 ..."
            ],
            "files": [
                "test/vector_add.hip",
                "test/vector_add.hip",
                "test/vector_add.hip"
            ],
            "hip": [
                "    c[idx] = a[idx] + b[idx];",
                "    c[idx] = a[idx] + b[idx];",
                "    c[idx] = a[idx] + b[idx];"
            ],
            "lines": [11, 11, 11],
            "signature": "vector_add(float const*, float const*, float*, int)"
        }
    }
}
```

</p>
</details>

## License

MIT
