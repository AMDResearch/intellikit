---
myst:
    html_meta:
        "description": "Omniprobe detects uncoalesced global memory accesses, LDS bank conflicts, and basic-block ISA timing in AMD GPU kernels with ROCm."
        "keywords": "Omniprobe, AMD GPU, ROCm, memory coalescing, bank conflicts, basic block, ISA timing, cache lines, HIP"
---

# Omniprobe

Omniprobe finds memory access patterns that waste bandwidth, and measures how long each basic block of a kernel takes.

It answers a question raw counters cannot: *which source line* caused the wasted traffic. Where a bandwidth metric tells you a kernel moved more bytes than it needed to, Omniprobe attributes the excess to a `file:line:column` and tells you how many extra cache lines that access cost.

## Requirements

Omniprobe is a **separate tool** that this package drives — it is not a pip dependency, and installing the package does not install it.

- Python >= 3.10
- ROCm 7.0+ and a supported AMD GPU
- The `omniprobe` executable on `PATH`
- Omniprobe's Python API importable, so that `from omniprobe.api import Omniprobe` works — the [Omniprobe repository](https://github.com/AMDResearch/omniprobe) must be on `PYTHONPATH` or installed

The package wraps the existing installation through a subprocess. It does not bundle or build Omniprobe's C++ components.

## Installation

```bash
# From the IntelliKit monorepo root (editable install)
pip install -e omniprobe/

# Or from Git
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=omniprobe"
```

The Python package is named `omniprobe-ik` and imports as `omniprobe_ik`, to avoid colliding with upstream Omniprobe's own `omniprobe` namespace. The CLI is `omniprobe-ik` for the same reason — `omniprobe` remains the upstream command.

## Quick start

The following example analyses an application and prints the accesses that cost the most excess cache lines.

```python
from omniprobe_ik import Omniprobe

probe = Omniprobe()
result = probe.analyze_memory("./my_gpu_app")

for kernel in result.kernels:
    worst = sorted(kernel.accesses, key=lambda a: a.excess_cache_lines, reverse=True)
    for access in worst[:5]:
        print(f"{access.source_file}:{access.line}:{access.column}")
        print(f"  {access.excess_cache_lines} excess cache lines, "
              f"{access.bank_conflicts} bank conflicts")
```

## What you get

Omniprobe exposes two analyses.

| Analysis | Method | Detects |
|----------|--------|---------|
| **Memory** | `analyze_memory` | Uncoalesced global accesses, LDS bank conflicts, excess cache lines |
| **Basic blocks** | `analyze_basic_blocks` | Per-block cycle counts and latency percentiles |

Memory analysis reports these fields per access site:

| Field | Description |
|-------|-------------|
| `excess_cache_lines` | Cache lines fetched beyond what the access needed |
| `bank_conflicts` | LDS bank conflicts observed at this access |
| `memory_space` | Which memory space the access targets |
| `access_type` | Read or write |
| `execution_count` | Number of times the access executed |

Basic-block analysis reports cycle counts as a distribution — `min_cycles`, `max_cycles`, and the `p25`/`p50`/`p75`/`p95`/`p99` percentiles — rather than a single average. A block whose p99 sits far above its p50 is being reached by waves under very different conditions, which an average hides.

## Compiling with `-g`

Both analyses attribute their findings to source locations, so debug information changes what you can act on.

| Build | Counts and timings | `source_file` / `line` / `column` |
|-------|--------------------|-----------------------------------|
| **With `-g`** | Correct | Real file path, line, and column |
| **Without `-g`** | Correct | Empty |

Compile with `-g` unless you only need aggregate counts. The numbers are right either way; without it you cannot map them back to code.

## API

The Omniprobe API exposes the analysis class and the result dataclasses for each analysis.

### `Omniprobe` class

```python
probe = Omniprobe(
    omniprobe_path=None,   # Explicit path to the omniprobe executable;
                           # None locates it via PATH
)
```

**Methods:**

- `analyze_memory(command, kernel_filter=None, dispatches="all")`: run memory analysis, returns `MemoryAnalysisResult`
- `analyze_basic_blocks(command, kernel_filter=None, dispatches="all")`: run basic-block analysis, returns `BasicBlockResult`

**Arguments shared by both methods:**

- `command`: application command line to instrument
- `kernel_filter`: optional ECMAScript regex limiting which kernels are analysed
- `dispatches`: which dispatches to capture — `"all"`, `"random"`, or `"1"`

Constructing `Omniprobe` raises `ImportError` if Omniprobe's Python API is not importable, and `FileNotFoundError` if the executable cannot be found.

### `MemoryAccess`

A single observed memory access.

```python
access.source_file          # Source file path (empty without -g)
access.line                 # Line number (0 without -g)
access.column               # Column number (0 without -g)
access.excess_cache_lines   # Cache lines fetched beyond what was needed
access.bank_conflicts       # LDS bank conflicts at this access
access.memory_space         # Memory space targeted
access.access_type          # Read or write
access.execution_count      # How many times it executed
```

`MemoryAnalysisResult.kernels` is a list of `KernelMemoryResult`, each with `kernel_name`, `dispatch_id`, and `accesses`.

### `BasicBlock`

Timing data for a single basic block.

```python
bb.bb_id                    # Basic block identifier
bb.source_file              # Source file path (empty without -g)
bb.line                     # Line number (0 without -g)
bb.min_cycles               # Fastest observed execution
bb.max_cycles               # Slowest observed execution
bb.p25, bb.p50, bb.p75      # Cycle count percentiles
bb.p95, bb.p99
bb.wave_count               # Waves that executed this block
```

`BasicBlockResult.kernels` is a list of `KernelBBResult`, each with `kernel_name`, `dispatch_id`, and `basic_blocks`.

## CLI

```bash
# Memory analysis
omniprobe-ik memory ./my_gpu_app

# Basic-block analysis
omniprobe-ik basic-blocks ./my_gpu_app

# Filter to specific kernels (ECMAScript regex)
omniprobe-ik memory --kernel "matmul.*" ./my_gpu_app

# Capture a single dispatch instead of all of them
omniprobe-ik memory --dispatches 1 ./my_gpu_app

# Write results to a file
omniprobe-ik memory --output results.json ./my_gpu_app
```

## MCP server

```bash
# stdio transport (default)
omniprobe-mcp --transport stdio

# HTTP transport
omniprobe-mcp --transport http --port 8000
```

The server exposes three tools:

| Tool | Purpose |
|------|---------|
| `analyze_memory` | Runs memory analysis, returns per-kernel access findings |
| `analyze_basic_blocks` | Runs basic-block analysis, returns per-kernel timing data |
| `list_analyzers` | Reports the available analysis types and which tool calls each |

## Examples

The following patterns build on the result objects described above.

```python
# Accesses that waste the most bandwidth, across all kernels
wasteful = [
    a
    for kernel in result.kernels
    for a in kernel.accesses
    if a.excess_cache_lines > 0
]
wasteful.sort(key=lambda a: a.excess_cache_lines, reverse=True)

# LDS bank conflicts only
conflicts = [
    a
    for kernel in result.kernels
    for a in kernel.accesses
    if a.bank_conflicts > 0
]

# Basic blocks with a long tail — p99 far above the median
bb_result = probe.analyze_basic_blocks("./my_gpu_app")
divergent = [
    bb
    for kernel in bb_result.kernels
    for bb in kernel.basic_blocks
    if bb.p50 > 0 and bb.p99 > bb.p50 * 4
]
```

Runnable versions of both analyses live in [`omniprobe/examples/`](https://github.com/AMDResearch/intellikit/tree/main/omniprobe/examples). Each writes a HIP kernel to a temporary directory, compiles it, and analyses it, so they run without a prepared workload.
