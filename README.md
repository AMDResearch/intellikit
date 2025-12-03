# IntelliKit

> [!IMPORTANT]
> This project is intended for research purposes only and is provided by AMD Research and Advanced Development team.
This is not a product. Use it at your own risk and discretion.

**LLM-Ready GPU Profiling and Analysis Toolkit for AMD ROCm**

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

# Capture snapshots from both versions
ref_snapshot = validator.capture_snapshot(binary=["./ref"], working_directory=".")
opt_snapshot = validator.capture_snapshot(binary=["./opt"], working_directory=".")

# Compare for correctness
result = validator.compare_snapshots(ref_snapshot, opt_snapshot)
print(result.summary())  # Shows validation results
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

## Requirements

- **Python**: >= 3.8 (3.9+ for metrix)
- **ROCm**: >= 6.0 (tested on 6.3.2)
- **Hardware**: MI300+ GPUs

## Documentation

Each tool has its own detailed documentation:
- [Nexus Documentation](nexus/README.md) + [Examples](nexus/examples/)
- [Accordo Documentation](accordo/README.md) + [Examples](accordo/examples/)
- [Metrix Documentation](metrix/README.md) + [Examples](metrix/examples/)

## Use Cases

### For LLM-Driven GPU Development
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

MIT License - Copyright (c) 2025 Advanced Micro Devices, Inc.

See [LICENSE](LICENSE) for full details.


## Support

Need help? Here's how to reach us:

- **Issues**: Found a bug or have a feature request? [Open an issue on GitHub](https://github.com/AMDResearch/nexus/issues)
- **Contact**: Reach out to the development team at [muhaawad@amd.com](mailto:muhaawad@amd.com)

---

**Made with 🧠 for the future of LLM-assisted GPU development**

