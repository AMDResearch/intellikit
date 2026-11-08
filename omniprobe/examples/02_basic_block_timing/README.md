# Example 02: Basic-Block Execution

Demonstrates Omniprobe's exact per-block execution counts and what they say about divergence.

## What it does

The Python script:

1. Writes a HIP kernel with a data-dependent branch to a temporary directory
2. Compiles it with `hipcc -O2 -g` plus `instrumentation_flags("basic_blocks")`
   (`-fpass-plugin=...BBStart...`, `--offload-arch=...`)
3. Runs `analyze_basic_blocks` and prints each block's `wave_visits` and `active_lanes`, most-executed first

## Reading the output

`wave_visits` is how many times a wave entered a block; `active_lanes` sums the active lanes over
those entries. Both sides of the branch are entered by every wave with 32 of its 64 lanes: divergence.
The counts are exact. Basic-block analysis does not time blocks; for where time goes use Metrix kernel
timing or Linex per-line cycles.

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
