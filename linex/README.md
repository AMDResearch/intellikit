# Linex

**Source-Level GPU Performance Profiling for AMD ROCm**

Map GPU performance metrics to your source code lines.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from linex import Linex

profiler = Linex()
profiler.profile("./my_app", kernel_filter="my_kernel")

# Show hotspots
for line in profiler.source_lines[:5]:
    print(f"{line.file}:{line.line_number}")
    print(f"  {line.total_cycles:,} cycles ({line.stall_percent:.1f}% stalled)")
```

## Distributed Launchers

Linex can wrap distributed launcher commands (`torchrun`, `mpirun/mpiexec`, `srun`,
`horovodrun`) and automatically records rank metadata from common environment variables.

```python
profiler = Linex()
profiler.profile(
    "torchrun --nproc_per_node=8 train.py",
    output_dir="linex_sqtt",
)

print(profiler.distributed_context.global_rank)
for rank_key, rank_profile in profiler.rank_profiles.items():
    print(rank_key, len(rank_profile.source_lines))
```

In distributed mode, Linex writes traces into rank-specific subdirectories
(`.../rank0000`, `.../rank0001`, ...) to avoid collisions.

## What You Get

**Instruction-level metrics mapped to source lines:**
- `latency_cycles` - Total GPU cycles
- `stall_cycles` - Cycles waiting (memory, dependencies)
- `idle_cycles` - Unused execution slots
- `execution_count` - How many times it ran
- `instruction_address` - Where in GPU memory

## Requirements

- Python >= 3.8
- ROCm 7.0+ with `rocprofv3`

### Compiling with and without `-g`

| Build        | `instructions`   | `source_lines`   | `InstructionData.file` / `.line` |
|-------------|------------------|------------------|-----------------------------------|
| **With `-g`**  | Populated (ISA + cycles) | Populated (aggregated by file:line) | Real file path and line number    |
| **Without `-g`** | Populated (ISA + cycles) | Empty            | `""` and `0`                      |

- **Use `-g`** when you want **source-line mapping**: ISA instructions tied to `file:line`, and `source_lines` aggregated by source line.
- **Omit `-g`** when you only need **assembly-level metrics**: you still get every instruction with `isa`, `latency_cycles`, `stall_cycles`, etc.; only file/line and `source_lines` are empty or zero.

## API

### Linex Class

```python
profiler = Linex(
    target_cu=0,                      # Target compute unit
    shader_engine_mask="0xFFFFFFFF",  # All shader engines
    activity=10,                      # Activity counter polling
)
```

**Methods:**
- `profile(command, kernel_filter=None)` - Run profiling

**Properties:**
- `source_lines` - List[SourceLine] sorted by total_cycles
- `instructions` - List[InstructionData]
- `rank_profiles` - Per-rank profiling data for distributed runs
- `distributed_context` - Detected launcher/rank metadata

### SourceLine

Aggregated metrics for one source code line.

```python
line.file                  # Source file path
line.line_number           # Line number
line.total_cycles          # Sum of all instruction cycles
line.stall_cycles          # Cycles spent waiting
line.idle_cycles           # Cycles slot was idle
line.execution_count       # Total executions
line.instructions          # List of ISA instructions
line.stall_percent         # Convenience: stall_cycles / total_cycles * 100
```

### InstructionData

Per-ISA-instruction metrics.

```python
inst.isa                   # ISA instruction text
inst.latency_cycles        # Total cycles for this instruction
inst.stall_cycles          # Cycles spent waiting
inst.idle_cycles           # Cycles slot was idle
inst.execution_count       # How many times it ran
inst.instruction_address   # Virtual address in GPU memory
inst.file                  # Parsed from source_location (empty without -g)
inst.line                  # Parsed from source_location (0 without -g)
inst.stall_percent         # Convenience: stall_cycles / latency_cycles * 100
```

## Example

```python
# Find memory-bound lines
memory_bound = [
    l for l in profiler.source_lines 
    if l.stall_percent > 50
]

# Find hotspots with high execution count
hotspots = [
    l for l in profiler.source_lines
    if l.execution_count > 10000
]

# Instruction-level analysis
for line in profiler.source_lines[:1]:
    for inst in line.instructions:
        print(f"{inst.isa}: {inst.latency_cycles} cycles")
```

See `examples/01_simple_sqtt/example.py` for a complete working example.

## License

MIT License - Copyright (c) 2026 Advanced Micro Devices, Inc.
