---
name: samplex-pc-sampling
description: PC sampling for GPU kernels - find instruction-level hotspots, stall reasons, and wave utilization
---

# Samplex: GPU PC Sampling

PC sampling via rocprofv3. Answers: "Where is my kernel stuck and why?"

Two methods:
- **stochastic** (default): Hardware-based, cycle-accurate, zero skid. Provides stall reasons, instruction types, wave counts. Requires MI300+ (gfx942+).
- **host_trap**: Software-based, time-based. Broader GPU support (MI200+). No stall reasons or instruction types.

## When to Use

- You need to find which **instructions** are bottlenecks in a GPU kernel
- You want to understand **stall reasons** (memory waits, ALU dependencies, barriers)
- You want a lightweight statistical profile (no replay overhead like counter collection)
- You need to see **wave occupancy** and **exec mask** divergence

## When NOT to Use

- You need **hardware counter values** (bandwidth, cache hits) - use **Metrix** instead
- You need **source-line mapping** - use **Linex** instead
- You need **kernel dispatch timing** only - use Metrix with `--time-only`

## CLI

```bash
# Stochastic PC sampling (default, MI300+)
samplex ./my_app

# Host-trap PC sampling (MI200+)
samplex --method host_trap ./my_app

# Coarser sampling (less overhead)
samplex --interval 4096 ./my_app

# Filter specific kernels
samplex --kernel "gemm.*" ./my_app

# JSON output
samplex -o results.json ./my_app

# List available configs
samplex list-configs
```

## Python API

```python
from samplex import Samplex

sampler = Samplex()
results = sampler.sample("./my_app")  # stochastic by default
# or: results = sampler.sample("./my_app", method="host_trap")

for kernel in results.kernels:
    print(f"{kernel.name}: {kernel.total_samples} samples")
    print(f"  Issued: {kernel.issued_pct:.1f}%")
    for hotspot in kernel.top_instructions[:5]:
        print(f"  {hotspot.percentage:.1f}% {hotspot.opcode} "
              f"[issued={hotspot.issued_count}, stalled={hotspot.stalled_count}]")
    if kernel.top_stall_reasons:
        for reason, pct in kernel.top_stall_reasons.items():
            print(f"  stall: {pct:.1f}% {reason}")
```

## Understanding the Output

### Key Opcodes

- **`s_waitcnt`**: GPU waiting for memory ops. High % = memory-bound.
- **`s_endpgm`**: End of program. High % = short kernels or GPU underutilized.
- **`s_barrier`**: Workgroup barrier. High % = load imbalance.
- **`v_mfma_*`**: Matrix multiply-accumulate. Compute is happening.
- **`global_load_*`**: Global memory loads.
- **`(empty)`**: No instruction captured ("holes") = idle GPU.

### Stall Reasons (stochastic only)

- **WAITCNT**: Waiting for memory operation (most common)
- **ALU_DEPENDENCY**: Data dependency on ALU result
- **OTHER_WAIT**: Barrier, LDS, or other synchronization
- **INTERNAL**: Internal hardware stall

### Issued vs Stalled (stochastic only)

Each sample tells you whether the wave **issued** the instruction or was **stalled**:
- High issued % = GPU is actively computing
- High stalled % = GPU is waiting (check stall reasons)

### Exec Mask

- **Full mask (100%)**: All 64 SIMD lanes active. No divergence.
- **Partial mask**: Control flow divergence.

### Wave Count

Number of waves active on the compute unit when sampled.
Low wave count with stalls = occupancy-limited.

## Install

```bash
pip install git+https://github.com/AMDResearch/intellikit.git#subdirectory=samplex
```
