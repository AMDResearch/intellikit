<div align="center">

![Intellikit Logo](./docs/intellikit.svg)

# IntelliKit

</div>

**LLM-Ready Profiling and Analysis Toolkit for AMD CPU and GPUs**

IntelliKit is a collection of profiling and analysis tools for AMD ROCm that expose clean, human-readable APIs for both developers and LLM-driven workflows.

## Installation

**Install all tools and agent skills** (one command each):

```bash
# Tools
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash

# Agent skills (so AI agents can discover and use IntelliKit tools)
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash
```

See [install/README.md](install/README.md) for all options (custom pip command, branch/tag, agent targets, dry-run, etc.).

**Install individual tools from Git:**

```bash
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=accordo"
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=linex"
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=metrix"
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=nexus"
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=rocm_mcp"
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=uprof_mcp"
```

**From a clone (editable installs):**

```bash
git clone https://github.com/AMDResearch/intellikit.git
cd intellikit
pip install -e ./accordo
pip install -e ./linex
# ... or any subset of the tools
```

## Requirements

- **Python**: >= 3.10
- **ROCm**: >= 6.0 (7.0+ for linex)
- **Hardware**: MI300+ GPUs

## Tools

### [Accordo](accordo/) - Automated Kernel Validation

Automated correctness validation for GPU kernel optimizations.

**Use cases:**

- Verify optimized kernels match reference implementation
- Compare performance while ensuring correctness
- Test multiple optimization candidates efficiently

**Quick example:**

```python
from accordo import Accordo

# Create validator (auto-extracts kernel signature)
validator = Accordo(binary="./ref", kernel_name="reduce_sum")

# Capture snapshots from reference and optimized binaries
ref = validator.capture_snapshot(binary="./ref")
opt = validator.capture_snapshot(binary="./opt")

# Compare for correctness
result = validator.compare_snapshots(ref, opt, tolerance=1e-6)

if result.is_valid:
    print(f"✓ PASS: {result.num_arrays_validated} arrays matched")
else:
    print(result.summary())
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
    "hip-compiler-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/rocm_mcp", "hip-compiler-mcp"]
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
    "uprof-profiler-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/uprof_mcp", "uprof-profiler-mcp"]
    }
  }
}
```

## Documentation

Each tool has its own detailed documentation:

- [Accordo Documentation](accordo/README.md) + [Examples](accordo/examples/)
- [Linex Documentation](linex/README.md) + [Examples](linex/examples/)
- [Metrix Documentation](metrix/README.md) + [Examples](metrix/examples/)
- [Nexus Documentation](nexus/README.md) + [Examples](nexus/examples/)
- [ROCm-MCP Documentation](rocm_mcp/README.md) + [Examples](rocm_mcp/examples/)
- [uprof-MCP Documentation](uprof_mcp/README.md) + [Examples](uprof_mcp/examples/)

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
