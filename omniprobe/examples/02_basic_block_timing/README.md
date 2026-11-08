# Example 02: Basic-Block Timing Analysis

Demonstrates using Omniprobe to measure how long each basic block of a kernel takes.

## What it does

The Python script:

1. Writes a HIP kernel with a data-dependent branch to a temporary directory
2. Compiles it with `hipcc -O2 -g` plus the Omniprobe instrumentation flags from `instrumentation_flags()` (`-fpass-plugin=...`, `-fgpu-rdc`, `--offload-arch=...`)
3. Runs `analyze_basic_blocks` and prints the costliest blocks by median cycles

The kernel branches on thread index. One side does a couple of multiplies; the other
runs a short loop of transcendentals. They become separate basic blocks with clearly
different cycle counts.

## Reading the output

The report gives `p50`, `p99` and `max_cycles` per block rather than one average.
That is deliberate — a block whose p99 sits far above its p50 is being reached by
waves under very different conditions, and averaging the two together hides exactly
the case worth investigating.

## Run it

```bash
python3 analyze_basic_blocks.py
```

If Omniprobe is not available, the script says why and runs the compiled kernel
directly, so a toolchain failure is distinguishable from an Omniprobe failure.

## Why `-g`

Without debug information, `bb.source_file` and `bb.line` come back empty and you get
block ids with no way to map them to code. The timings are still correct, just much
harder to act on.

## Requirements

- ROCm and `hipcc` installed
- `omniprobe` on `PATH`, with its Python API importable
- This package installed: `pip install -e omniprobe/`

See the [package README](../../README.md) for full prerequisites.
