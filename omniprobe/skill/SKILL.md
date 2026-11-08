---
name: omniprobe-analysis
description: Pinpoint which source lines (file:line:column) of a HIP kernel make uncoalesced global memory accesses or cause LDS bank conflicts, measured by recompiling with Omniprobe instrumentation. Use when you must name the inefficient access, after a kernel-level profiler (metrix) shows poor coalescing or LDS bank conflicts, or to verify a memory-access fix. Basic-block analysis shows which blocks run and how often, not their cost. Not for kernel duration or bandwidth (metrix) or per-line stall cycles (linex).
---

# Omniprobe: which line wastes memory bandwidth

Omniprobe instruments every memory instruction of a kernel at compile time and reports, per source
location, how many cache lines were fetched beyond the minimum (uncoalesced global accesses) and
how many extra cycles LDS bank conflicts cost. It is the only IntelliKit tool that attributes
these to an individual access; it costs a rebuild with a compiler plugin.

## When to use, and when not to

| You need | Use |
|---|---|
| Kernel time, bandwidth, cache hit rates, overall coalescing efficiency or LDS bank-conflict counts | **metrix** (no rebuild) |
| Which source lines stall the most cycles | **linex** |
| **Which access (file:line) is uncoalesced, or causes LDS bank conflicts, and how badly** | **Omniprobe memory analysis** |
| Checking that a coalescing/padding fix worked, per access | **Omniprobe** (before/after) |
| Which basic blocks run, and how many waves reach each | Omniprobe basic-block analysis (see caveat below) |
| Where kernel **time** goes | **metrix** / rocprof kernel timing, **linex** per-line cycles — not Omniprobe |

When not to use: questions about duration, throughput or occupancy; kernels you cannot rebuild
(prebuilt libraries such as rocBLAS/Tensile assembly); Triton kernels unless Omniprobe's Triton setup
exists. A typical sequence is metrix first (is memory efficiency poor?), then Omniprobe (which line?).
Reading the source can suggest a culprit; Omniprobe measures it — report measured numbers.

## Workflow

1. **Build instrumented.** Omniprobe sees only kernels compiled with its plugin. Get the flags:
   `omniprobe-ik flags memory` (or MCP `get_instrumentation_flags`, or Python
   `instrumentation_flags("memory")`). They are `-fpass-plugin=<omniprobe>/lib/plugins/
   libAMDGCNSubmitAddressMessages-rocm.so --offload-arch=<arch>`. Add `-g` for source lines.
   - Single file: `hipcc -O2 -g $(omniprobe-ik flags memory) app.hip -o app`
   - Makefile: rebuild from clean, keeping the project's own flags and adding ours, e.g.
     `make clean && make HIPFLAGS="-O2 -g $(omniprobe-ik flags memory)"` (read the Makefile for the
     variable name and its current value). Every file that contains a kernel needs the plugin.
   - CMake: configure a **fresh** build directory with
     `-DCMAKE_HIP_FLAGS="-g $(omniprobe-ik flags memory)"` (or `CMAKE_CXX_FLAGS` when hipcc is the
     C++ compiler).
   - **Do not add `-fgpu-rdc`.** Instrumentation does not need it, and with the plugin an RDC build
     of two or more files fails to link (`duplicate symbol: __hip_cuid_...`). Only if the project
     already builds with RDC, keep it and use `omniprobe-ik flags memory --rdc`.
2. **Run the analysis.** `omniprobe-ik memory ./app [args]` (MCP `analyze_memory`, Python
   `Omniprobe().analyze_memory("./app")`). The command is split on whitespace (no shell quoting);
   wrap complex command lines in a script. Limit cost with `--kernel '<regex>'` (matched against the
   **demangled** name, e.g. `'advance'`, `'gstride<16>'`) and `--dispatches 1`.
3. **Read the ranked summary** ("Top sites", or `summary` from MCP / `summarize_memory()`): worst
   sites first, with file:line:column, read/write, and a per-wave-execution value (below).
   **Check `warnings` / `uninstrumented_kernels` first**: a kernel listed there ran without
   instrumentation and is absent from the results — rebuild the file that defines it with the
   flags, or report the gap; never conclude "no problem" for a kernel that was not measured.
4. **Explain and fix**, then rebuild and rerun to show the per-execution number dropped.

## Reading the results (calibrated on gfx90a)

All counts are totals over the run; divide by `execution_count` (wave executions of that
instruction) to get the per-wave-execution value the summary prints.

- **Global `excess_cache_lines` per wave execution**: cache lines touched beyond the minimum.
  0 = perfectly coalesced. For 4-byte accesses by 64 lanes on 128-byte lines, a lane stride of S
  elements gives `min(64, 2S) - 2`: stride 2 → 2, 4 → 6, 8 → 14, 16 → 30, ≥32 → **62 (worst: every
  lane its own line)**. Random gathers through an index array also approach 62.
- **LDS `bank_conflicts` per wave execution**: extra cycles. 32 banks × 4 bytes, wave checked as two
  halves of 32 lanes; 2 × (ways − 1). Column reads of a 32-wide float tile → **62**; padding the row
  to 33 → 0. Lanes reading the same word (broadcast) cost nothing.
- Rank by **total** to decide what to fix first; use **per execution** to say how bad the pattern is.
- A load and a store on the same line are separate sites.

## Blind spots — when Omniprobe reports nothing for an access you suspect

- **Pointers loaded from device memory** (e.g. `float* a = table->field[k]; a[i]`) are generic
  (flat) accesses that Omniprobe **silently ignores**: the line will be missing from the results or
  show only the table read. Pointers passed as kernel arguments, held in a by-value struct argument,
  or stored in `__constant__` variables are seen. To measure such an access, temporarily pass the
  pointer as a kernel argument (or a by-value struct), rebuild, rerun — and say in your answer that
  you did so and why.
- **Kernels without parameters** are not matched at run time ("No instrumented alternative found").
  Temporarily add an unused pointer parameter to measure them.
- **Architecture**: cache-line analysis exists for gfx906/908 (64-B lines) and gfx90a/940-942
  (128-B). Elsewhere only LDS analysis is reported.
- If a kernel is absent from the results entirely, check Omniprobe's stderr for
  `No instrumented alternative found for <kernel>` and see Troubleshooting.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `NoInstrumentedKernelsError` / "no instrumented kernels" | Binary built without the plugin, or wrong plugin | Rebuild with `$(omniprobe-ik flags memory)`; `make clean` first |
| `ImportError: omniprobe.api` | Omniprobe source tree not on `PYTHONPATH` (MCP servers start with a minimal env) | Set `PYTHONPATH` (and `ROCR_VISIBLE_DEVICES`) in the shell or MCP `env` |
| Link error `duplicate symbol: __hip_cuid_...` | `-fgpu-rdc` with ≥ 2 files | Drop `-fgpu-rdc` unless the project needs it |
| `WARNING: … ran WITHOUT instrumentation` / `uninstrumented_kernels` non-empty | That kernel's file was built without the plugin, or a known non-match (no parameters; internal linkage under `-fgpu-rdc`) | Rebuild that file with the flags; see Blind spots |
| A kernel or line is missing | Blind spot above, or `--kernel` regex too narrow | See Blind spots; filter on demangled names |
| `RuntimeError: omniprobe exited with code N` | The application itself failed | Run the binary alone first; fix arguments/inputs |
| No GPU / runs on the wrong GPU | `HIP_VISIBLE_DEVICES` set to the same id as `ROCR_VISIBLE_DEVICES` selects nothing | Set only `ROCR_VISIBLE_DEVICES` |

## Analyses and their plugins

| Analysis | Build flags | CLI | MCP tool | Python |
|---|---|---|---|---|
| Memory (coalescing + LDS bank conflicts) | `omniprobe-ik flags memory` (AddressMessages plugin) | `omniprobe-ik memory` | `analyze_memory` | `analyze_memory`, `summarize_memory` |
| Basic blocks (execution counts) | `omniprobe-ik flags basic-blocks` (BBStart plugin) — a **separate build** | `omniprobe-ik basic-blocks` | `analyze_basic_blocks` | `analyze_basic_blocks` |

`list_analyzers` (MCP) reports these and the blind spots.

**Basic-block analysis: which blocks run and how often — its cycles are not per-block cost.** Use `wave_count` (how
many waves reached a block) to see which paths execute. Do **not** use its cycle fields
(`min_cycles`…`p99`) as per-block cost: Omniprobe's basic-block "cycles" are currently intervals between instrumentation submits, dominated by instrumentation overhead (millions of cycles at thousands of waves, thousands at a few waves, for the same kernel), and block ids and source lines are not yet reliably attributed. Omniprobe is reworking this (2026-11-08). For time hotspots use metrix or rocprof kernel
timing, and linex for per-line cycles. Never run the basic-block analysis on a binary built for
memory analysis.

Further detail: `docs/tools/omniprobe.md` in the IntelliKit repository
(https://github.com/AMDResearch/intellikit/blob/main/docs/tools/omniprobe.md).
