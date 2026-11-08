# Example 01: Memory Coalescing Analysis

Demonstrates using Omniprobe to find uncoalesced global memory accesses.

## What it does

The Python script:

1. Writes a HIP file containing two kernels to a temporary directory
2. Compiles it with `hipcc -O2 -g` plus the Omniprobe instrumentation flags from `instrumentation_flags()` (`-fpass-plugin=...`, `--offload-arch=...`)
3. Runs `analyze_memory` and reports the worst accesses per kernel

The two kernels do identical arithmetic and differ only in indexing:

| Kernel | Indexing | Expected finding |
|--------|----------|------------------|
| `coalesced_copy` | `in[idx]` | Adjacent threads share cache lines — little or no excess |
| `strided_copy` | `in[idx * 16]` | Each thread pulls its own cache line — high `excess_cache_lines` |

That contrast is the point: `excess_cache_lines` counts bytes the hardware fetched
and threw away, which is invisible in a wall-clock measurement of either kernel alone.

## Run it

```bash
python3 analyze_memory.py
```

If Omniprobe is not available, the script says why and runs the compiled kernel
directly, so a failure in the toolchain is distinguishable from a failure in Omniprobe.

## Why `-g`

Omniprobe reports each finding against `source_file`, `line` and `column`, and memory
analysis matches each access to its instruction through the source line. Compiled without
debug information, no access can be matched: they are dropped and counted in
`unattributed_accesses`, and `summarize_memory` warns.

## Requirements

- ROCm and `hipcc` installed
- `omniprobe` on `PATH`, with its Python API importable
- This package installed: `pip install -e omniprobe/`

See the [package README](../../README.md) for full prerequisites.
