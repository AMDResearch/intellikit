# Example 02: Basic-Block Execution

Demonstrates using Omniprobe to see which basic blocks of a kernel execute and how
many waves reach each.

## What it does

The Python script:

1. Writes a HIP kernel with a data-dependent branch to a temporary directory
2. Compiles it with `hipcc -O2 -g` plus `instrumentation_flags("basic_blocks")`
   (`-fpass-plugin=...BBStart...`, `--offload-arch=...`)
3. Runs `analyze_basic_blocks` and prints each block's `wave_count`, most-reached first

## Reading the output

`wave_count` is how many waves reached a block, which shows which paths run.
Omniprobe's per-block cycle fields (`min_cycles` … `p99`) are **not** per-block cost
today: they are intervals between instrumentation submits, dominated by
instrumentation overhead, and block ids and lines are not yet reliably attributed.
Omniprobe is reworking this. For where time goes, use Metrix kernel timing or Linex
per-line cycles.

## Run it

```bash
python3 analyze_basic_blocks.py
```

If Omniprobe is not available, the script says why and runs the compiled kernel
directly, so a toolchain failure is distinguishable from an Omniprobe failure.

## Requirements

- ROCm and `hipcc` installed
- `omniprobe` on `PATH`, with its Python API importable
- This package installed: `pip install -e omniprobe/`

See the [package README](../../README.md) for full prerequisites.
