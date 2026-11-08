---
name: omniprobe-analysis
description: Analyze GPU memory coalescing and basic-block timing using Omniprobe. Use when investigating uncoalesced memory accesses, LDS bank conflicts, or ISA-level timing per basic block.
---

# Omniprobe: GPU Memory and Basic-Block Analysis

Detect GPU memory coalescing issues (uncoalesced global accesses, LDS bank conflicts) and measure ISA basic-block timing with cycle-level distributions.

## Prerequisites

Omniprobe must be pre-installed on the system:
- The `omniprobe` executable must be on `PATH`
- The Omniprobe Python API (`omniprobe.api`) must be importable (Omniprobe repo on `PYTHONPATH` or installed)
- ROCm 7.0+ and a supported AMD GPU
- **The application must be rebuilt with Omniprobe's instrumentation plugin.** Omniprobe only sees kernels compiled with `-fpass-plugin=<omniprobe>/lib/plugins/<plugin>.so -fgpu-rdc --offload-arch=<gfx arch>`. An ordinary build is reported as "no instrumented kernels" (`NoInstrumentedKernelsError`). Get the exact flags with `omniprobe-ik flags memory|basic-blocks`, `instrumentation_flags(...)`, or the MCP tool `get_instrumentation_flags`. Memory and basic-block analysis use **different plugins**, so they need separate builds.

## When to Use

- User asks about memory coalescing or uncoalesced accesses
- Investigating LDS bank conflicts
- Need cycle-level timing per ISA basic block
- Performance analysis at the memory-access or instruction-block granularity

## Instructions

1. **Ensure the target runs on AMD ROCm** (e.g. `hipcc`-built binary or Python script that launches HIP/ROCm kernels).
2. **Rebuild the target with the instrumentation flags** for the analysis you want, plus `-g` for source lines (see Prerequisites). Without this step every analysis fails with "no instrumented kernels".
3. **Choose execution path:**
   - If an Omniprobe MCP server is available: use `list_analyzers` to see analysis types, `get_instrumentation_flags` for the build flags, then `analyze_memory` or `analyze_basic_blocks`.
   - Otherwise: run the CLI or Python API.

### CLI

```bash
# Memory coalescing analysis: build with the memory plugin, then analyze
hipcc -g -O2 $(omniprobe-ik flags memory) my_gpu_app.hip -o my_gpu_app
omniprobe-ik memory ./my_gpu_app

# Basic-block timing: a separate build with the basic-block plugin
hipcc -g -O2 $(omniprobe-ik flags basic-blocks) my_gpu_app.hip -o my_gpu_app_bb
omniprobe-ik basic-blocks ./my_gpu_app_bb

# Filter kernels by name
omniprobe-ik memory --kernel "matmul.*" ./my_gpu_app

# Save results
omniprobe-ik memory --output results.json ./my_gpu_app
```

### Python API

```python
from omniprobe_ik import Omniprobe, instrumentation_flags

# hipcc flags for each analysis; build the application with them first
mem_flags = instrumentation_flags("memory")
bb_flags = instrumentation_flags("basic_blocks")

probe = Omniprobe()

# Memory analysis
mem = probe.analyze_memory("./my_gpu_app")
for kernel in mem.kernels:
    for access in kernel.accesses:
        print(f"{access.source_file}:{access.line} "
              f"bank_conflicts={access.bank_conflicts} "
              f"excess_cache_lines={access.excess_cache_lines}")

# Basic-block analysis
bb = probe.analyze_basic_blocks("./my_gpu_app")
for kernel in bb.kernels:
    for block in kernel.basic_blocks:
        print(f"BB {block.bb_id}: p50={block.p50:.1f} cycles")
```

## Workflow

1. Identify the GPU executable or script to analyze.
2. Choose analysis type: `memory` for coalescing/bank conflicts, `basic-blocks` for ISA timing.
3. Rebuild with that analysis's instrumentation flags and `-g`.
4. Optionally filter kernels with `--kernel` regex.
5. Interpret results:
   - High `excess_cache_lines` → uncoalesced global memory accesses
   - Non-zero `bank_conflicts` → LDS bank conflicts
   - High `max_cycles` relative to `min_cycles` → variable-latency basic blocks

## Analysis Types

- **Memory**: Detects uncoalesced global accesses, LDS bank conflicts, excess cache lines per memory access site
- **Basic Blocks**: Measures min/max cycles, percentile distributions (p25–p99), wave counts per ISA basic block
