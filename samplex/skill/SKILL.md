---
name: samplex-pc-sampling
description: PC sampling profiling for GPU kernels - find instruction-level hotspots and stall reasons
---

# Samplex: GPU PC Sampling

Statistical instruction-level profiling via rocprofv3 PC sampling. Answers: "Where is my kernel stuck?"

## When to Use

- You need to find which **instructions** are bottlenecks in a GPU kernel
- You want to understand **stall reasons** (memory waits, ALU dependencies, barriers)
- You want a lightweight statistical profile (no replay overhead like counter collection)
- You need to see the **exec mask** to check for divergent control flow

## When NOT to Use

- You need **hardware counter values** (bandwidth, cache hits) - use **Metrix** instead
- You need **source-line mapping** - use **Linex** instead
- You need **kernel dispatch timing** only - use Metrix with `--time-only`

## Sampling Methods

| Method | Unit | Hardware | Precision | Extra Info |
|--------|------|----------|-----------|------------|
| `host_trap` | microseconds | MI200+ | ~2 instruction skid | Reliable, more samples |
| `stochastic` | cycles (power of 2) | MI300+ | Zero skid | Stall reasons, instruction types |

## CLI

```bash
# Basic PC sampling (host_trap, 1us interval)
samplex ./my_app

# With kernel filter
samplex --kernel "gemm.*" ./my_app

# Stochastic sampling (precise, with stall reasons)
samplex --method stochastic --interval 256 ./my_app

# JSON output
samplex -o results.json ./my_app

# List available configs
samplex list-configs
```

## Python API

```python
from samplex import Samplex

sampler = Samplex()
results = sampler.sample("./my_app")

for kernel in results.kernels:
    print(f"{kernel.name}: {kernel.total_samples} samples")
    for hotspot in kernel.top_instructions[:5]:
        print(f"  {hotspot.percentage:.1f}% {hotspot.opcode}")
    if kernel.top_stall_reasons:
        print(f"  Stall reasons: {kernel.top_stall_reasons}")
```

## Understanding the Output

### Key Opcodes

- **`s_waitcnt`**: GPU is waiting for memory operations to complete. High percentage = memory-bound.
- **`s_endpgm`**: End of program. High percentage = kernels are very short or GPU is underutilized.
- **`s_barrier`**: Workgroup barrier synchronization. High percentage = load imbalance.
- **`v_mfma_*`**: Matrix multiply-accumulate. Seeing these means compute is happening.
- **`global_load_*`**: Global memory loads. Many samples here = memory-bound.
- **`(empty)`**: No instruction captured ("holes"). Indicates idle GPU or between-dispatch gaps.

### Stall Reasons (stochastic only)

- **WAITCNT**: Waiting for memory operation (`s_waitcnt`)
- **ALU_DEPENDENCY**: Data dependency on ALU result
- **OTHER_WAIT**: Other wait (barrier, LDS, etc.)

### Exec Mask

- **Full mask (100%)**: All 64 SIMD lanes active. No divergence.
- **Partial mask**: Some lanes masked off due to control flow divergence.
- **Zero mask**: No lanes active (between-wave gaps).

## Install

```bash
pip install git+https://github.com/AMDResearch/intellikit.git#subdirectory=samplex
```
