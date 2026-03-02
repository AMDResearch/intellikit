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

## Installation

Install each tool from its subdirectory. No top-level metapackage.

**Install all tools from Git (one command):**

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash
```

**Options:**

- **Custom pip command** (for multiple Python versions):
  ```bash
  curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash -s -- --pip-cmd pip3.12
  # or
  curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash -s -- --pip-cmd "python3.12 -m pip"
  ```

- **Install from a specific branch/tag:**
  ```bash
  curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash -s -- --ref my-branch
  ```

- **Dry-run (preview commands):**
  ```bash
  curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash -s -- --dry-run
  ```

- **From a clone:**
  ```bash
  ./install/tools/install.sh --pip-cmd pip3.12 --ref main --dry-run
  ```

**Environment variables** (CLI options take precedence): `PIP_CMD`, `INTELLIKIT_REPO_URL`, `INTELLIKIT_REF`

**Install individual tools from Git:**

```bash
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=accordo"
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=linex"
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=metrix"
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=nexus"
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=rocm_mcp"
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=uprof_mcp"
```

**From a clone (editable installs from local paths):**

```bash
git clone https://github.com/AMDResearch/intellikit.git
cd intellikit
pip install -e ./accordo
pip install -e ./linex
# ... or any subset of the tools
```

### Agent Skills (AI agents)

Install IntelliKit skills so AI agents can discover and use Metrix, Accordo, and Nexus. Skills are installed as `SKILL.md` files under a single directory; agents that read that location get the instructions automatically.

**Default: local (current workspace) - agents target**

```bash
# One-liner: installs into ./.agents/skills/ (current directory)
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash
```

**Different agent targets** (cursor, claude, codex, agents):

```bash
# Cursor
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash -s -- --target cursor

# Claude
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash -s -- --target claude

# Codex
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash -s -- --target codex
```

**Global (all projects)**

```bash
# Install into ~/.cursor/skills/ (or ~/.claude/skills/, etc.)
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash -s -- --target cursor --global
```

**From a clone**

```bash
git clone https://github.com/AMDResearch/intellikit.git
cd intellikit
./install/skills/install.sh                    # local: ./.agents/skills/
./install/skills/install.sh --target cursor     # local: ./.cursor/skills/
./install/skills/install.sh --target claude --global  # global: ~/.claude/skills/
./install/skills/install.sh --dry-run          # show what would be installed
```

**Resulting layout:**

- **Local (agents):** `./.agents/skills/metrix/SKILL.md`, `./.agents/skills/accordo/SKILL.md`, `./.agents/skills/nexus/SKILL.md`
- **Local (cursor):** `./.cursor/skills/metrix/SKILL.md`, `./.cursor/skills/accordo/SKILL.md`, `./.cursor/skills/nexus/SKILL.md`
- **Global (claude):** `~/.claude/skills/metrix/SKILL.md`, `~/.claude/skills/accordo/SKILL.md`, `~/.claude/skills/nexus/SKILL.md`

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
validator = Accordo(binary="./app_baseline", kernel_name="my_kernel")

ref_snap = validator.capture_snapshot(binary="./app_baseline")
opt_snap = validator.capture_snapshot(binary="./app_opt")
result = validator.compare_snapshots(ref_snap, opt_snap, tolerance=1e-6)

if result.is_valid:
    opt_results = profiler.profile("./app_opt")
    opt_bw = opt_results.kernels[0].metrics['memory.hbm_bandwidth_utilization'].avg
    print(f"✓ PASS: {result.num_arrays_validated} arrays matched")
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
