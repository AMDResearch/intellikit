# Samplex

**GPU PC Sampling. Where is my kernel stuck?**

Instruction-level hotspots, stall reasons, and wave utilization for AMD GPUs via rocprofv3.

## Why Samplex?

When your GPU kernel is slow, you need to know **which instructions** are the bottleneck
and **why** the GPU is stalling. Samplex wraps rocprofv3's PC sampling to give you:

- **Instruction hotspots** - which opcodes the GPU spends the most time on
- **Stall reasons** - WAITCNT (memory), BARRIER_WAIT, ALU_DEPENDENCY, etc.
- **Issued vs stalled** - is the GPU computing or waiting?
- **Exec mask divergence** - are all SIMD lanes active?
- **Two sampling methods** - stochastic (MI300+) and host_trap (MI200+)

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Stochastic PC sampling (default, MI300+)
samplex ./my_app

# Host-trap PC sampling (MI200+)
samplex --method host_trap ./my_app

# Filter specific kernels
samplex --kernel "vector_add" ./my_app

# JSON output
samplex -o results.json ./my_app

# List available PC sampling configs
samplex list-configs
```

## Python API

```python
from samplex import Samplex

sampler = Samplex()
results = sampler.sample("./my_app")

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

## Sampling Methods

| | **stochastic** (default) | **host_trap** |
|---|---|---|
| Mechanism | Hardware-based | Software trap |
| Precision | Cycle-accurate, zero skid | Non-zero skid |
| GPU support | MI300+ (gfx942+) | MI200+ |
| Stall reasons | Yes | No |
| Issued/stalled | Yes | No |
| Instruction types | Yes | No |
| Wave count | Yes | No |
| Interval unit | Cycles (power of 2) | Nanoseconds |
| Default interval | 65536 | 65536 |

## CLI Options

```
samplex [--version] <command> ...

samplex profile [options] <target>

  --method, -m     Sampling method: stochastic | host_trap (default: stochastic)
  --interval, -i   Sampling interval (default: 65536). Stochastic: cycles. Host_trap: ns.
  --kernel, -k     Filter kernels by name (regex, passed to rocprofv3)
  --top N          Show top N instructions per kernel (default: 10)
  --output, -o     Output file (.json or .txt)
  --timeout        Profiling timeout in seconds (default: no timeout)
  --log, -l        Log level: debug | info | warning | error (default: warning)

samplex list-configs
```

## Understanding the Output

### Key Opcodes

- **`s_waitcnt`** - GPU waiting for memory ops. High % = memory-bound.
- **`s_barrier`** - Workgroup barrier. High % = load imbalance.
- **`v_mfma_*`** - Matrix multiply-accumulate. Compute is happening.
- **`global_load_*`** / **`buffer_load_*`** - Global memory loads.
- **`(empty)`** - No instruction captured ("holes") = idle GPU.

### Stall Reasons (stochastic only)

- **WAITCNT** - Waiting for memory operation (most common)
- **BARRIER_WAIT** - Workgroup barrier synchronization
- **ALU_DEPENDENCY** - Data dependency on ALU result
- **ARBITER_WIN_EX_STALL** - CU execution scheduling stall
- **ARBITER_NOT_WIN** - CU arbitration loss
- **NO_INSTRUCTION_AVAILABLE** - No instruction ready to issue

### Issued vs Stalled (stochastic only)

Each sample tells you whether the wave **issued** the instruction or was **stalled**:
- High issued % = GPU is actively computing
- High stalled % = GPU is waiting (check stall reasons)

### Exec Mask

- **Full mask (100%)** - All 64 SIMD lanes active. No divergence.
- **Partial mask** - Control flow divergence.

## Examples

See [examples/01_basic_sampling](examples/01_basic_sampling/) for a complete working example with expected output.

## Testing

```bash
python3 -m pytest tests/ -v
```

## Requirements

- Python 3.9+
- ROCm 6.x with rocprofv3
- MI300+ for stochastic sampling, MI200+ for host_trap

## License

MIT
