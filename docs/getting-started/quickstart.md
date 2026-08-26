---
myst:
    html_meta:
        "description": "Profile a GPU application with IntelliKit Metrix using the CLI or Python API. Collect hardware counters and human-readable performance metrics on AMD GPUs."
        "keywords": "IntelliKit, Metrix, GPU profiling, AMD GPU, ROCm, HIP, hardware counters, bandwidth, cache"
---

# Profile a GPU application with Metrix in IntelliKit

This topic explains how to use Metrix to profile a GPU application and generate human-readable performance metrics.

## CLI

Use the CLI to quickly profile your application and gather performance metrics.

```bash
# Profile with all metrics (GPU architecture auto-detected)
metrix ./my_app

# Time only (fast)
metrix --time-only -n 10 ./my_app

# Filter kernels by name
metrix --kernel matmul ./my_app

# Specific metrics
metrix --metrics memory.hbm_bandwidth_utilization,memory.l2_hit_rate ./my_app
```

## Python API

Access more advanced profiling features through the Python API.

```python
from metrix import Metrix

profiler = Metrix()
results = profiler.profile("./my_app", num_replays=5)

for kernel in results.kernels:
    print(f"{kernel.name}: {kernel.duration_us.avg:.2f} μs")
    for metric, stats in kernel.metrics.items():
        print(f"  {metric}: {stats.avg:.2f}")
```

## Example output

The following output shows Metrix profiling a vector add kernel with all metrics enabled.

```
================================================================================
Metrix: all metrics (12 total)
Target: ./examples/01_vector_add/vector_add
================================================================================

────────────────────────────────────────────────────────────────────────────────
Dispatch #1: vector_add(float*, float const*, float const*, int)
────────────────────────────────────────────────────────────────────────────────
Duration: 7.29 - 7.29 μs (avg=7.29)

MEMORY BANDWIDTH:
  Total HBM Bytes Transferred                   8400896.00 bytes
  HBM Bandwidth Utilization                           1.34 percent
  HBM Read Bandwidth                                 35.47 GB/s
  HBM Write Bandwidth                                35.36 GB/s

CACHE PERFORMANCE:
  L1 Cache Hit Rate                                  66.67 percent
  L2 Cache Hit Rate                                  26.72 percent
```

## Next steps

After profiling your first application, explore the rest of the IntelliKit toolkit.

- **Dive deeper into profiling:** see [Metrix](../tools/metrix.md) for a full list of available metrics
- **Map performance to source lines:** see [Linex](../tools/linex.md) for source-level profiling
- **Extract and isolate a kernel:** see [Kerncap](../tools/kerncap.md) for standalone reproducers
- **Inspect GPU execution:** see [Nexus](../tools/nexus.md) for HSA packet tracing
- **Validate optimizations:** see [Accordo](../tools/accordo.md) for correctness checking
- **Set up MCP servers:** see [MCP Setup](../how-to/mcp-setup.md) for LLM integration
