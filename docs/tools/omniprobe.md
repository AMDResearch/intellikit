---
myst:
    html_meta:
        "description": "Omniprobe detects uncoalesced global memory accesses and LDS bank conflicts at source level in AMD GPU kernels, and reports which basic blocks execute with ROCm."
        "keywords": "Omniprobe, AMD GPU, ROCm, memory coalescing, bank conflicts, basic block, ISA timing, cache lines, HIP"
---

# Omniprobe

Omniprobe finds the memory accesses that waste bandwidth, at source level, and reports which basic blocks of a kernel execute and how often.

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

Omniprobe analyzes only kernels that were compiled with its instrumentation plugin, so the first step is always a build with the right flags (see [Compiling for Omniprobe](#compiling-for-omniprobe)):

```bash
hipcc -g -O2 $(omniprobe-ik flags memory) my_gpu_app.hip -o my_gpu_app
```

The following example then analyses the application and prints the accesses that cost the most excess cache lines.

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
| **Basic blocks** | `analyze_basic_blocks` | Which blocks execute and how many waves reach each (`wave_count`); see the caveat below |

Memory analysis reports these fields per access site:

| Field | Description |
|-------|-------------|
| `excess_cache_lines` | Cache lines fetched beyond what the access needed |
| `bank_conflicts` | LDS bank conflicts observed at this access |
| `memory_space` | Which memory space the access targets |
| `access_type` | Read or write |
| `execution_count` | Number of times the access executed |

**Basic-block analysis tells you which blocks run and how often; its cycle fields are not per-block cost.** Use `wave_count`. Do not read its cycle fields (`min_cycles`, `max_cycles`, `p25`…`p99`) as per-block cost: Omniprobe's basic-block "cycles" are currently intervals between instrumentation submits, dominated by instrumentation overhead (millions of cycles at thousands of waves, thousands at a few waves, for the same kernel), and block ids and source lines are not yet reliably attributed. Omniprobe is reworking this (2026-11-08). For time hotspots use Metrix kernel timing or Linex per-line cycles.

## Reading the results

These definitions are pinned by `omniprobe/tests/integration/test_calibration.py`, which checks
them against closed-form expectations on real hardware. Every number is a **total over the
dispatch**, keyed by source location, memory space and access type (a load and a store on the same
line are separate entries; Omniprobe may also split one location into several entries, so sum
them).

| Field | Meaning |
|-------|---------|
| `execution_count` | Wave executions of the instrumented instruction (one per wave each time it runs) |
| `excess_cache_lines` | Global only. Cache lines touched beyond the minimum the active lanes needed: per wave execution, `ceil(active_lanes × access_bytes / line)` lines are needed and the distinct lines actually touched are counted |
| `bank_conflicts` | LDS only. Extra serialized cycles from bank conflicts: 32 banks of 4 bytes, a 64-lane wave checked as two halves of 32 lanes (8-byte accesses: four groups of 16; 16-byte: eight of 8); each group costs `ways − 1`, where `ways` is the largest number of distinct dwords mapped to one bank. Lanes reading the same dword (broadcast) cost nothing |

Worked values on gfx90a (128-byte lines), 4-byte accesses, 64 active lanes, **per wave execution**:

| Lane stride (floats) | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| global `excess_cache_lines` | 0 | 2 | 6 | 14 | 30 | 62 | 62 |
| LDS `bank_conflicts` | 0 | 2 | 6 | 14 | 30 | 62 | 62 |

An LDS stride of 33 (a padded tile) gives 0 conflicts.

To rank findings, divide by `execution_count`. `excess_cache_lines / execution_count` is the
wasted lines per wave execution: 0 is perfectly coalesced and 62 is the worst case (every lane on
its own line). The raw total ranks sites by how much bandwidth they waste overall, so use it to
choose what to fix first.

### Kernels that ran without instrumentation

If some dispatched kernels have no instrumented clone (their file was built without the plugin, or
one of the cases below), Omniprobe still runs them, uninstrumented, and they are simply absent from
the results. The wrapper reads Omniprobe's log and lists them in `result.uninstrumented_kernels`;
`summarize_memory`, the MCP tools and the CLI turn that into a warning. `None` means the log could
not be read, not that nothing was missed. The HIP runtime's own helper kernels (`__amd_rocclr_*`)
are never instrumented and are not listed.

### What Omniprobe cannot see

- **Accesses through a pointer that was itself loaded from device memory** (for example
  `float* p = table[k]; p[i]`). These are generic (flat) address-space accesses, and
  MemoryAnalysis ignores them silently. Pointers passed as kernel arguments, held in a by-value
  struct argument, or stored in `__constant__` variables are seen.
- **Code built at `-O0`.** Pointers are kept on the stack and every access becomes generic (flat),
  so instrumented kernels report no accesses at all. This is CMake's default (no build type) and its
  `Debug` type. Build with `-O1` or higher (CMake: `-DCMAKE_BUILD_TYPE=RelWithDebInfo`);
  `summarize_memory` warns when an instrumented kernel reports no global or LDS access.
- **Kernels without parameters.** They are cloned at compile time but not matched at run time
  ("No instrumented alternative found"), so they report nothing.
- **Kernels with internal linkage** (`static` or in an anonymous namespace) are often not matched
  at run time either, for the same symptom. Check Omniprobe's stderr for
  `No instrumented alternative found for <kernel>`.
- **Architectures without a cache-line size in Omniprobe's table** (gfx906/908: 64 bytes;
  gfx90a/940/941/942: 128 bytes). On any other architecture, global cache-line analysis is
  skipped and only LDS analysis remains.

## Compiling for Omniprobe

Omniprobe instruments kernels at compile time. An LLVM pass plugin clones every kernel and injects data-collection calls into the clone; at run time Omniprobe swaps the clone in. A kernel compiled without the plugin has no clone, so Omniprobe has nothing to observe: it exits successfully and reports no kernels. `analyze_memory` and `analyze_basic_blocks` turn that case into a `NoInstrumentedKernelsError` that explains the fix, rather than returning an empty result.

Each analysis needs its own plugin, so memory and basic-block analysis need separate builds:

| Analysis | Plugin passed with `-fpass-plugin=` |
|----------|-------------------------------------|
| Memory | `libAMDGCNSubmitAddressMessages-rocm.so` |
| Basic blocks | `libAMDGCNSubmitBBStart-rocm.so` |

The plugins live in `lib/plugins/` of the Omniprobe build or install tree. `instrumentation_flags` locates them next to the `omniprobe` executable on `PATH` (or in `OMNIPROBE_PLUGIN_DIR`) and returns the flag set: `-fpass-plugin=<plugin>` and `--offload-arch=<gfx arch>`, detected with `rocminfo` unless given.

```python
from omniprobe_ik import instrumentation_flags

instrumentation_flags("memory")
# ['-fpass-plugin=/opt/omniprobe/lib/plugins/libAMDGCNSubmitAddressMessages-rocm.so',
#  '--offload-arch=gfx90a']
```

From a shell, `omniprobe-ik flags memory` or `omniprobe-ik flags basic-blocks` prints the same flags.

### Relocatable device code (`-fgpu-rdc`)

Instrumentation does **not** need `-fgpu-rdc`, although Omniprobe's own documentation says so.
Add the flags to the application's existing device-compile flags and keep its build mode. Measured
on gfx90a with Omniprobe `9b8a3c4`:

| Build mode | Result |
|------------|--------|
| Without `-fgpu-rdc` (one-step or separate compile and link) | Every kernel is matched, including `static` and anonymous-namespace kernels and kernels in other translation units |
| With `-fgpu-rdc`, one translation unit | Works; internal-linkage kernels are renamed `<name>.intern.<hash>` and may fail to match at run time (Omniprobe defect `bf_intern-clone-name`) |
| With `-fgpu-rdc`, two or more translation units | Linking fails: `lld: error: duplicate symbol: __hip_cuid_...` |

If an application must build with RDC, use `instrumentation_flags(..., rdc=True)` or
`omniprobe-ik flags --rdc` and pass `-fgpu-rdc` to the link step as well.

### Adding `-g`

Both analyses attribute their findings to source locations, so debug information changes what you can act on.

| Build | Counts and timings | `source_file` / `line` / `column` |
|-------|--------------------|-----------------------------------|
| **With `-g`** | Correct | Real file path, line, and column |
| **Without `-g`** | Correct | Empty |

Compile with `-g` unless you only need aggregate counts. The numbers are right either way; without it you cannot map them back to code.

## Other analyzers and bespoke handlers

`run_analyzer(analyzer, command, ...)` (CLI `omniprobe-ik run`, MCP `run_analyzer`) runs any Omniprobe
analyzer — `Heatmap`, `AddressLogger`, `BasicBlockLogger`, `MemoryAnalysis`, `BasicBlockAnalysis` —
or a message-handler plugin you built, and returns everything it produced (`output` for JSON
formats; `log` for what was printed — the `omniprobe` launcher forwards the application's and the
handlers' output on its stderr). `analyze_heatmap`, `log_addresses` and `log_basic_blocks` parse
the first three into typed results.

A **bespoke handler** is a C++ class derived from `dh_comms::message_handler_base` with
`handle(message)` (return `true` if it consumed the message) and `report()` (called once per
dispatch), exported through `extern "C" getMessageHandlers(kernel, dispatch_id, handlers)`. Address
messages (`user_type == address`) carry one 64-bit address per active lane; `user_data` packs the
access kind (bits 0–1), address space (bits 2–5) and size in bytes (bits 6 and up). Omniprobe does
not install its headers, so build against its source tree: `handler_build_flags()` /
`omniprobe-ik handler-flags` give the `hipcc` command. `omniprobe/examples/04_custom_handler/` is a
complete handler with a test against hand-computed counts; test any handler that way before using
it on a real application. `omniprobe/examples/05_cmake_app/` shows instrumenting a CMake project.

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
- `kernel_filter`: optional ECMAScript regex limiting which kernels are analysed, searched within the **demangled** kernel name (for example `'gstride<16>'`; a mangled name such as `_Z7gstride...` matches nothing)
- `dispatches`: which dispatches to capture — `"all"`, `"random"`, or `"1"`

Constructing `Omniprobe` raises `ImportError` if Omniprobe's Python API is not importable, and `FileNotFoundError` if the executable cannot be found. Both methods raise `NoInstrumentedKernelsError` (a `RuntimeError`) when Omniprobe reports no kernel — almost always a binary built without the plugin; the message names the flags to rebuild with.

### `instrumentation_flags`

```python
instrumentation_flags(
    analysis="memory",     # "memory" or "basic_blocks"
    offload_arch=None,     # e.g. "gfx90a"; detected with rocminfo if None
    omniprobe_path=None,   # used to locate lib/plugins; None uses PATH
)
```

Returns the list of `hipcc` flags described in [Compiling for Omniprobe](#compiling-for-omniprobe). Raises `FileNotFoundError` if the plugins cannot be located and `RuntimeError` if no architecture is given and none can be detected.

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

Data for a single basic block. Rely on `wave_count`; the cycle fields are not per-block cost (see [What you get](#what-you-get)).

```python
bb.bb_id                    # Basic block identifier
bb.source_file              # Source file path (empty without -g)
bb.line                     # Line number (0 without -g)
bb.min_cycles               # Not per-block cost today (see above)
bb.max_cycles               # Not per-block cost today
bb.p25, bb.p50, bb.p75      # Not per-block cost today
bb.p95, bb.p99
bb.wave_count               # Waves that executed this block
```

`BasicBlockResult.kernels` is a list of `KernelBBResult`, each with `kernel_name`, `dispatch_id`, and `basic_blocks`.

## CLI

```bash
# Print the hipcc flags for an analysis (build with these first)
omniprobe-ik flags memory
omniprobe-ik flags basic-blocks --offload-arch gfx942

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

The server exposes four tools:

| Tool | Purpose |
|------|---------|
| `get_instrumentation_flags` | Returns the `hipcc` flags that instrument kernels for an analysis |
| `analyze_memory` | Runs memory analysis; returns a ranked `summary` (worst sites first, per-wave-execution values, `how_to_read`) and the raw per-kernel findings |
| `analyze_basic_blocks` | Runs basic-block analysis; returns which blocks ran and their wave counts |
| `list_analyzers` | Reports the available analysis types, which tool and plugin each needs, and Omniprobe's known blind spots |

Omniprobe's Python API lives only in its **source** tree (the install tree does not include it), so
the server needs that tree on `PYTHONPATH`. MCP clients usually start servers with a minimal
environment, so set it, and the GPU selection, in the server's configuration:

```json
"omniprobe": {
  "command": "omniprobe-mcp",
  "env": {
    "PATH": "/opt/omniprobe/bin:/opt/rocm/bin:/usr/bin:/bin",
    "PYTHONPATH": "/path/to/omniprobe-source",
    "ROCR_VISIBLE_DEVICES": "0"
  }
}
```

If `PYTHONPATH` is missing, the wrapper also tries the source tree that the `omniprobe` executable
resolves to, which works when `omniprobe` on `PATH` is a build tree's symlink.

`summarize_memory(result, top=10)` produces the same ranked summary from the Python API.

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

# Basic blocks that executed, most-reached first (wave_count, not cost)
bb_result = probe.analyze_basic_blocks("./my_gpu_app_bb")
reached = sorted(
    (bb for kernel in bb_result.kernels for bb in kernel.basic_blocks if bb.wave_count > 0),
    key=lambda bb: bb.wave_count,
    reverse=True,
)
```

Runnable versions of both analyses live in [`omniprobe/examples/`](https://github.com/AMDResearch/intellikit/tree/main/omniprobe/examples). Each writes a HIP kernel to a temporary directory, compiles it, and analyses it, so they run without a prepared workload.
