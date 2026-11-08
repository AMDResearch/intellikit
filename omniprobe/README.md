# Omniprobe (IntelliKit)

GPU memory coalescing analysis and basic-block ISA timing for IntelliKit.

This package wraps [Omniprobe](https://github.com/AMDResearch/omniprobe)'s Python API to provide memory access pattern analysis and basic-block timing through IntelliKit's standard Python API, MCP server, CLI, and agent skill interfaces.

## Prerequisites

Omniprobe must be **pre-installed** on the system. This package does not bundle or build Omniprobe — it drives the existing installation via its Python API.

1. **Omniprobe executable** — `omniprobe` must be on `PATH`
2. **Omniprobe Python API** — the Omniprobe repository must be on `PYTHONPATH` or installed so that `from omniprobe.api import Omniprobe` works
3. **ROCm 7.0+** and a supported AMD GPU
4. **Python 3.10+**
5. **An instrumented build of the application.** Omniprobe only analyzes kernels compiled with its LLVM pass plugin (`-fpass-plugin=<omniprobe>/lib/plugins/<plugin>.so -fgpu-rdc --offload-arch=<arch>`), and memory and basic-block analysis use different plugins. `instrumentation_flags()` and `omniprobe-ik flags` return the exact flags; a binary built without them raises `NoInstrumentedKernelsError`.

## Installation

```bash
# From the IntelliKit monorepo root (editable install)
pip install -e omniprobe/

# Or from Git
pip install "git+https://github.com/AMDResearch/intellikit.git@main#subdirectory=omniprobe"
```

## Quick Start

### Python API

```python
from omniprobe_ik import Omniprobe, instrumentation_flags

# Build ./my_gpu_app with these (plus -g) first; basic blocks need
# instrumentation_flags("basic_blocks") and a separate build
print(" ".join(instrumentation_flags("memory")))

probe = Omniprobe()

# Memory coalescing analysis
result = probe.analyze_memory("./my_gpu_app")
for kernel in result.kernels:
    for access in kernel.accesses:
        print(f"{access.source_file}:{access.line} "
              f"bank_conflicts={access.bank_conflicts}")

# Basic-block timing
result = probe.analyze_basic_blocks("./my_gpu_app")
for kernel in result.kernels:
    for bb in kernel.basic_blocks:
        print(f"BB {bb.bb_id}: {bb.p50:.1f} cycles (p50)")
```

### CLI

```bash
# Build with the memory plugin, then run memory analysis
hipcc -g -O2 $(omniprobe-ik flags memory) my_gpu_app.hip -o my_gpu_app
omniprobe-ik memory ./my_gpu_app

# Basic-block analysis needs its own build
hipcc -g -O2 $(omniprobe-ik flags basic-blocks) my_gpu_app.hip -o my_gpu_app_bb
omniprobe-ik basic-blocks ./my_gpu_app_bb

# Filter kernels
omniprobe-ik memory --kernel "matmul.*" ./my_gpu_app
```

### MCP Server

```bash
# Start stdio server
omniprobe-mcp --transport stdio

# Start HTTP server
omniprobe-mcp --transport http --port 8000
```

The MCP server exposes four tools: `get_instrumentation_flags`, `analyze_memory`, `analyze_basic_blocks`, and `list_analyzers`.

## Analysis Types

| Analysis | Detects | Key Fields |
|----------|---------|------------|
| **Memory** | Uncoalesced global accesses, LDS bank conflicts, excess cache lines | `bank_conflicts`, `excess_cache_lines`, `memory_space` |
| **Basic Blocks** | Per-block cycle timing, latency distribution | `min_cycles`, `max_cycles`, `p50`, `p95`, `p99` |

## License

MIT
