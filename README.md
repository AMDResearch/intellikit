# IntelliKit

> [!IMPORTANT]
> This project is intended for research purposes only and is provided by AMD Research and Advanced Development team.
This is not a product. Use it at your own risk and discretion.

> **LLM-Ready GPU Profiling and Analysis Toolkit for AMD ROCm**

IntelliKit is a collection of intelligent tools designed to make GPU kernel development, profiling, and validation accessible to LLMs and human developers alike. Built for AMD ROCm, these tools provide clean abstractions over complex GPU internals.

## Philosophy

Traditional GPU profiling and analysis tools expose raw hardware counters and assembly. IntelliKit tools are designed to:
- **Decode complexity**: Turn hardware metrics into human-readable insights
- **Enable LLM integration**: Provide clean APIs suitable for LLM-driven workflows (MCP-ready)
- **Focus on research**: Experimental tools for pushing GPU development forward

## Tools

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
result = validator.validate(reference=["./ref"], optimized=["./opt"])
print(result.summary())  # ✓ Validation passed! or ✗ Validation failed
```

### [Metrix](metrix/) - Human-Readable GPU Metrics
Decodes hardware counters into actionable performance insights.

**Use cases:**
- Profile GPU kernels with clean, understandable metrics
- Identify memory bandwidth bottlenecks
- Analyze compute utilization patterns

**Quick example:**
```python
from metrix import profile

metrics = profile("my_app")
print(f"Memory BW Utilization: {metrics.memory_bandwidth_util:.1f}%")
print(f"Compute Efficiency: {metrics.compute_efficiency:.1f}%")
```

## Installation

### Install All Tools

```bash
pip install "git+https://github.com/AMDResearch/nexus.git#egg=intellikit[all]"
```

This installs: `nexus`, `accordo`, and `metrix`

### Install Individual Tools

Install only what you need using extras:

```bash
# Nexus only
pip install "git+https://github.com/AMDResearch/nexus.git#egg=intellikit[nexus]"

# Accordo only
pip install "git+https://github.com/AMDResearch/nexus.git#egg=intellikit[accordo]"

# Metrix only
pip install "git+https://github.com/AMDResearch/nexus.git#egg=intellikit[metrix]"

# Multiple tools
pip install "git+https://github.com/AMDResearch/nexus.git#egg=intellikit[nexus,metrix]"
```

### Development Installation

```bash
git clone https://github.com/AMDResearch/nexus.git
cd nexus

# Install all tools in editable mode
pip install -e ".[all]"

# Or install specific tools only
pip install -e ".[nexus]"
pip install -e ".[accordo]"
pip install -e ".[metrix]"
```

## 🔧 Requirements

- **Python**: >= 3.8 (3.9+ for metrix)
- **ROCm**: >= 6.0 (tested on 6.3.2)
- **OS**: Linux (tested on RHEL/CentOS)
- **Hardware**: AMD GPUs (MI200/MI300 series recommended)

## Documentation

Each tool has its own detailed documentation:
- [Nexus Documentation](nexus/README.md)
- [Accordo Documentation](accordo/README.md)
- [Metrix Documentation](metrix/README.md)

## 🎓 Use Cases

### For LLM-Driven GPU Development (IntelliPerf)
IntelliKit tools provide clean APIs that LLMs can call to:
- Profile kernels and get human-readable feedback (Metrix)
- Extract actual GPU code for analysis (Nexus)
- Validate optimizations automatically (Accordo)

### For Research & Development
- Experiment with kernel optimizations
- Debug GPU compilation pipelines
- Analyze performance characteristics
- Validate correctness of transformations

### For Performance Engineering
- Identify bottlenecks in GPU applications
- Compare optimization strategies
- Track performance regressions

## 🧪 Example Workflow

```python
# 1. Profile baseline kernel with Metrix
from metrix import profile
baseline_metrics = profile("./app_baseline", gpu_arch="gfx942")

# 2. Extract kernel source with Nexus
from nexus import Nexus
nexus = Nexus()
trace = nexus.run(["./app_baseline"])
print(trace["my_kernel"].hip)

# 3. Apply optimization (external step)
# ... modify kernel ...

# 4. Validate with Accordo
from accordo import Accordo
config = Accordo.Config(kernel_name="my_kernel", ...)
validator = Accordo(config)
result = validator.validate(reference=["./app_baseline"], optimized=["./app_opt"])

if result.is_valid:
    opt_metrics = profile("./app_opt")
    print(f"✓ Validation passed! {result.num_arrays_validated} arrays matched")
    print(f"BW Improvement: {opt_metrics.memory_bandwidth_util - baseline_metrics.memory_bandwidth_util:.1f}%")
```

## Contributing

We welcome contributions and feedback! Open an issue or create a PR.

## License

MIT License - Copyright (c) 2025 Advanced Micro Devices, Inc.

See [LICENSE](LICENSE) for full details.


## Support

Need help? Here's how to reach us:

- **Issues**: Found a bug or have a feature request? [Open an issue on GitHub](https://github.com/AMDResearch/nexus/issues)
- **Contact**: Reach out to the development team at [muhaawad@amd.com](mailto:muhaawad@amd.com)

---

**Made with 🧠 for the future of LLM-assisted GPU development**

