---
myst:
    html_meta:
        "description": "Linex maps GPU performance metrics to your HIP source code lines, identifying hotspots, stall cycles, and idle execution on AMD GPUs with ROCm."
        "keywords": "Linex, AMD GPU, ROCm, source-line profiling, GPU hotspot, stall cycles, HIP, instruction analysis"
---

# Linex

Linex maps GPU performance metrics directly to the corresponding lines of your source code.

## Requirements

Linex requires the following.

- Python >= 3.10
- ROCm >= 7.0 with `rocprofv3`

## Installation

```bash
pip install -e .
```

## Quick start

The following example profiles an application and prints the five hottest source lines.

```python
from linex import Linex

profiler = Linex()
profiler.profile("./my_app", kernel_filter="my_kernel")

# Show hotspots
for line in profiler.source_lines[:5]:
    print(f"{line.file}:{line.line_number}")
    print(f"  {line.total_cycles:,} cycles ({line.stall_percent:.1f}% stalled)")
```

## What you get

Linex provides instruction-level GPU performance metrics mapped to corresponding source lines:

| Metric | Description |
|--------|-------------|
| `latency_cycles` | Total GPU cycles |
| `stall_cycles` | Cycles spent waiting (memory, dependencies) |
| `idle_cycles` | Unused execution slots |
| `execution_count` | Number of times the instruction was executed |
| `instruction_address` | Memory address of the GPU instruction |

## Compiling with and without `-g`

The availability of source-line data depends on whether your application is compiled with debug symbols.

| Build | `instructions` | `source_lines` | `file` / `line` |
|-------|----------------|-----------------|------------------|
| **With `-g`** | Populated (ISA + cycles) | Populated (aggregated by file:line) | Real file path and line number |
| **Without `-g`** | Populated (ISA + cycles) | Empty | `""` and `0` |

- Use `-g` when you need source-line mapping. This generates ISA instructions tied to `file:line`, and `source_lines` aggregated by source line.
- Omit `-g` when you only need assembly-level metrics. You'll still receive data for each instruction, including `isa`, `latency_cycles`, `stall_cycles`, and so on.

## API

The Linex API exposes the profiler class, source line objects, and instruction data objects.

### Linex class

```python
profiler = Linex(
    target_cu=0,                      # Target compute unit
    shader_engine_mask="0xFFFFFFFF",  # All shader engines
    activity=10,                      # Activity counter polling
)
```

**Methods:**
- `profile(command, kernel_filter=None)`: run profiling

**Properties:**
- `source_lines`: `List[SourceLine]` sorted by `total_cycles`
- `instructions`: `List[InstructionData]`

### `SourceLine`

Aggregated performance metrics for a single source code line.

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

### `InstructionData`

Performance metrics for a single ISA instruction.

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

## Examples

The following examples demonstrate common Linex analysis patterns.

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
