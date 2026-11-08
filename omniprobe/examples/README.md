# Omniprobe Examples

Runnable examples for the two analyses this package exposes.

| Example | Analysis | Demonstrates |
|---------|----------|--------------|
| [`01_memory_coalescing/`](01_memory_coalescing/) | `analyze_memory` | Detecting uncoalesced global accesses and LDS bank conflicts |
| [`02_basic_block_timing/`](02_basic_block_timing/) | `analyze_basic_blocks` | Per-block cycle counts and latency percentiles |
| [`03_lds_bank_conflicts/`](03_lds_bank_conflicts/) | `analyze_memory` + `summarize_memory` | Locating LDS bank conflicts to a source line, and the padding fix |

Each example is self-contained: it writes a HIP kernel to a temporary directory,
compiles it with `hipcc` and the Omniprobe instrumentation plugin for its analysis
(the flags come from `instrumentation_flags()`), and analyses the resulting binary.
Nothing is left behind. Omniprobe only sees kernels built with the plugin, which is
why the examples never use a plain `hipcc` build.

## Requirements

These examples need more than a `pip install` — Omniprobe itself is a separate,
pre-installed tool (see [Prerequisites](../README.md#prerequisites)):

- ROCm 7.0+, `hipcc`, and a supported AMD GPU
- The `omniprobe` executable on `PATH`
- Omniprobe's Python API importable (`from omniprobe.api import Omniprobe`)
- This package installed: `pip install -e omniprobe/`

If Omniprobe is unavailable, each script reports why and falls back to running the
compiled kernel directly, so you can still confirm the HIP toolchain works.

Both examples compile with `-g`. Omniprobe reports findings against `source_file`,
`line` and `column`; without debug information those fields come back empty and the
output is far less useful.
