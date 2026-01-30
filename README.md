<div align="center">

![Intellikit Logo](./docs/intellikit.svg)

# IntelliKit

</div>

**LLM-Ready Profiling and Analysis Toolkit for AMD CPU and GPUs**

IntelliKit is a collection of intelligent tools designed to make CPU and GPU code development, profiling, and validation accessible to LLMs and human developers alike. Built for AMD ROCm, these tools provide clean abstractions over complex GPU internals.

## Philosophy

Traditional CPU and GPU profiling and analysis tools expose raw hardware counters and assembly. IntelliKit tools are designed to:

- **Decode complexity**: Turn hardware metrics into human-readable insights
- **Enable LLM integration**: Provide clean APIs suitable for LLM-driven workflows (MCP-ready)

## Tools

### [Accordo](accordo/) - Automated Kernel Validation

Side-by-side correctness validation for GPU kernel optimizations.

**Use cases:**

- Verify optimized kernels match reference implementation
- Compare performance while ensuring correctness
- Snapshot-based testing for multiple optimization candidates

**Quick example:**

```python
from accordo import Accordo

config = Accordo.Config(
    kernel_name="my_kernel",
    kernel_args=[Accordo.KernelArg(name="result", type="double*")],
    tolerance=1e-6
)
validator = Accordo(config)

# Capture snapshots from both versions
ref_snapshot = validator.capture_snapshot(binary=["./ref"], working_directory=".")
opt_snapshot = validator.capture_snapshot(binary=["./opt"], working_directory=".")

# Compare for correctness
result = validator.compare_snapshots(ref_snapshot, opt_snapshot)
print(result.summary())  # Shows validation results
```

### [Linex](linex/) - Source-Level GPU Performance Profiling

Maps GPU performance metrics to your source code lines.

**Use cases:**

- Identify performance hotspots at source code granularity
- Understand cycle-level timing for each line of code
- Analyze stall patterns and execution bottlenecks

**Quick example:**

```python
from linex import Linex

profiler = Linex()
profiler.profile("./my_app", kernel_filter="my_kernel")

# Show hotspots
for line in profiler.source_lines[:5]:
    print(f"{line.file}:{line.line_number}")
    print(f"  {line.total_cycles:,} cycles ({line.stall_percent:.1f}% stalled)")
```

### [Metrix](metrix/) - Human-Readable GPU Metrics

Decodes hardware counters into actionable performance insights.

**Use cases:**

- Profile GPU kernels with clean, understandable metrics
- Identify memory bandwidth bottlenecks
- Analyze compute utilization patterns

**Quick example:**

```python
from metrix import Metrix

profiler = Metrix()
results = profiler.profile("./my_app", metrics=["memory.hbm_bandwidth_utilization"])

for kernel in results.kernels:
    print(f"{kernel.name}: {kernel.duration_us.avg:.2f} μs")
    print(f"Memory BW: {kernel.metrics['memory.hbm_bandwidth_utilization'].avg:.1f}%")
```

### [Nexus](nexus/) - HSA Packet Source Code Extractor

Intercepts GPU kernel launches and extracts source code + assembly from HSA packets.

**Use cases:**

- Understand what code actually runs on the GPU
- Debug kernel compilation and optimization
- Trace HIP, Triton, and other GPU frameworks

**Quick example:**

```python
from nexus import Nexus

nexus = Nexus(log_level=1)
trace = nexus.run(["python", "gpu_app.py"])

for kernel in trace:
    print(f"{kernel.name}: {len(kernel.assembly)} instructions")
    print(kernel.hip)  # Source code
```

### [ROCm-MCP](rocm_mcp/) - Model Context Protocol Servers of ROCm Tools

Enables LLMs to interact with ROCm tools via MCP.

**Use cases:**

- Compile HIP code.
- Access HIP reference guide.
- Query device capabilities.

**Quick example:**

Add to your JSON MCP config:

```json
{
  "mcpServers": {
    "hip-compiler": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/rocm_mcp",
        "run",
        "hip-compiler-mcp"
      ]
    }
  }
}
```

### [uprof-MCP](uprof_mcp/) - Model Context Protocol Server for uProf

Enables LLMs to interact with AMD uProf via MCP.

**Use cases:**

- Profile applications using uProf.

**Quick example:**
Add to your JSON MCP config:

```json
{
  "mcpServers": {
    "uprof": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/uprof_mcp",
        "run",
        "uprof-profiler-mcp"
      ]
    }
  }
}
```

## Installation

### Install All Tools

```bash
pip install "git+https://github.com/AMDResearch/intellikit.git#egg=intellikit[all]"
```

This installs: `accordo`, `linex`, `metrix`, `nexus`, `rocm_mcp`, and `uprof_mcp`.

### Install Individual Tools

Install only what you need using extras:

```bash
# Accordo only
pip install "git+https://github.com/AMDResearch/intellikit.git#egg=intellikit[accordo]"

# Linex only
pip install "git+https://github.com/AMDResearch/intellikit.git#egg=intellikit[linex]"

# Metrix only
pip install "git+https://github.com/AMDResearch/intellikit.git#egg=intellikit[metrix]"

# Nexus only
pip install "git+https://github.com/AMDResearch/intellikit.git#egg=intellikit[nexus]"

# ROCm-MCP only
pip install "git+https://github.com/AMDResearch/intellikit.git#egg=intellikit[rocm_mcp]"

# uprof-MCP only
pip install "git+https://github.com/AMDResearch/intellikit.git#egg=intellikit[uprof_mcp]"

# Multiple tools
pip install "git+https://github.com/AMDResearch/intellikit.git#egg=intellikit[nexus,metrix]"
```

### Development Installation

```bash
git clone https://github.com/AMDResearch/intellikit.git
cd intellikit

# Install all tools in editable mode
pip install -e ".[all]"

# Or install specific tools only
pip install -e ".[accordo]"
pip install -e ".[linex]"
pip install -e ".[metrix]"
pip install -e ".[nexus]"
pip install -e ".[rocm_mcp]"
pip install -e ".[uprof_mcp]"
```

## Requirements

- **Python**: >= 3.10
- **ROCm**: >= 6.0 (7.0+ for linex)
- **Hardware**: MI300+ GPUs

## Documentation

Each tool has its own detailed documentation:

- [Accordo Documentation](accordo/README.md) + [Examples](accordo/examples/)
- [Linex Documentation](linex/README.md) + [Examples](linex/examples/)
- [Metrix Documentation](metrix/README.md) + [Examples](metrix/examples/)
- [Nexus Documentation](nexus/README.md) + [Examples](nexus/examples/)
- [ROCm-MCP Documentation](rocm_mcp/README.md) + [Examples](rocm_mcp/examples/)
- [uprof-MCP Documentation](uprof_mcp/README.md) + [Examples](uprof_mcp/examples/)

## Example Workflow

```python
# 1. Profile baseline kernel with Metrix
from metrix import Metrix
profiler = Metrix()
baseline_results = profiler.profile("./app_baseline")
baseline_bw = baseline_results.kernels[0].metrics['memory.hbm_bandwidth_utilization'].avg

# 2. Extract kernel source with Nexus
from nexus import Nexus
nexus = Nexus()
trace = nexus.run(["./app_baseline"])
for kernel in trace:
    print(kernel.hip)  # Source code

# 3. Apply optimization (external step)
# ... modify kernel ...

# 4. Validate with Accordo
from accordo import Accordo
config = Accordo.Config(kernel_name="my_kernel", ...)
validator = Accordo(config)

ref_snap = validator.capture_snapshot(binary=["./app_baseline"], working_directory=".")
opt_snap = validator.capture_snapshot(binary=["./app_opt"], working_directory=".")
result = validator.compare_snapshots(ref_snap, opt_snap)

if result.is_valid:
    opt_results = profiler.profile("./app_opt")
    opt_bw = opt_results.kernels[0].metrics['memory.hbm_bandwidth_utilization'].avg
    print(f"VALIDATION PASSED: {result.num_arrays_validated} arrays matched")
    print(f"BW Improvement: {opt_bw - baseline_bw:.1f}%")
```

## Contributing

We welcome contributions and feedback! Open an issue or create a PR.

## License

MIT License - Copyright (c) 2025-2026 Advanced Micro Devices, Inc.

See [LICENSE](LICENSE) for full details.

## Support

Need help? Here's how to reach us:

- **Issues**: Found a bug or have a feature request? [Open an issue on GitHub](https://github.com/AMDResearch/intellikit/issues)

---

**Made with 🧠 for the future of LLM-assisted GPU development**
