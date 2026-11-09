---
name: omniprobe-analysis
description: Pinpoint which source lines (file:line:column) of a HIP kernel make uncoalesced global memory accesses or cause LDS bank conflicts, measured by recompiling with Omniprobe instrumentation. Use when you must name the inefficient access, after a kernel-level profiler (metrix) shows poor coalescing or LDS bank conflicts, or to verify a memory-access fix. Basic-block analysis gives exact per-block execution counts, not timing. Not for kernel duration or bandwidth (metrix) or per-line stall cycles (linex).
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
| Exactly how often each basic block runs, and with how many active lanes (divergence) | Omniprobe basic-block analysis |
| Where kernel **time** goes | **metrix** / rocprof kernel timing, **linex** per-line cycles — not Omniprobe |

When not to use: questions about duration, throughput or occupancy; kernels you cannot rebuild
(prebuilt libraries such as rocBLAS/Tensile assembly); Triton kernels unless Omniprobe's Triton setup
exists. A typical sequence is metrix first (is memory efficiency poor?), then Omniprobe (which line?).
Reading the source can suggest a culprit; Omniprobe measures it — report measured numbers.

## Workflow

1. **Build instrumented.** Omniprobe sees only kernels compiled with its plugin. Get the flags:
   `omniprobe-ik flags memory` (or MCP `get_instrumentation_flags`, or Python
   `instrumentation_flags("memory")`). They are `-fpass-plugin=<omniprobe>/lib/plugins/
   libAMDGCNSubmitAddressMessages-rocm.so --offload-arch=<arch>`. **Add `-g`**: without it memory
   analysis cannot match accesses to source lines and drops them (`unattributed_accesses`, warned).
   - Single file: `hipcc -O2 -g $(omniprobe-ik flags memory) app.hip -o app`
   - Makefile: rebuild from clean, keeping the project's own flags and adding ours, e.g.
     `make clean && make HIPFLAGS="-O2 -g $(omniprobe-ik flags memory)"` (read the Makefile for the
     variable name and its current value). Every file that contains a kernel needs the plugin.
   - CMake: configure a **fresh** build directory with
     `-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_HIP_FLAGS="$(omniprobe-ik flags memory)"` (or
     `CMAKE_CXX_FLAGS` when hipcc is the C++ compiler).
   - **Optimisation level:** `-O0` builds (CMake's default build type, or `Debug`) are analyzed too
     (Omniprobe e807927 and later), but they measure `-O0` code, whose accesses differ from the
     optimized build's.
   - **Do not add `-fgpu-rdc`** unless the project already uses it: instrumentation does not need
     it. If it does, keep it (`omniprobe-ik flags memory --rdc`); static kernels are then named
     `<name> [clone .intern] [clone .<hash>]`. Multi-file RDC builds link and every kernel is
     matched with Omniprobe 3003058 or later (older versions fail with `duplicate symbol: __hip_cuid_...`).
2. **Run the analysis.** `omniprobe-ik memory [options] ./app [app args]` (or
   `omniprobe-ik memory [options] -- ./app --app-option`; MCP `analyze_memory`, Python
   `Omniprobe().analyze_memory("./app arg")`). The command is split on whitespace (no shell
   quoting); wrap command lines with quoted arguments in a script. Limit cost with `--kernel '<regex>'` and `--dispatches 1`.
   The regex is searched anywhere in the **demangled** name (`'advance'`, `'gstride<16>'`). A
   template kernel's name starts with its return type (`void gstride<16>(float const*, float*)`), so
   `'^gstride'` matches nothing; a filter that matches no dispatched kernel is an error listing the
   kernels that were dispatched. `--dispatches` selects dispatches of each kernel, numbered per
   kernel from 1: `all`, `N`, `A-B`, `A-`, comma lists, `:count[:stride]` repeats (`2,5`, `1::100`);
   Omniprobe validates it and a malformed spec is an error before the run. Results carry
   `kernel_dispatch` (that number) next to `dispatch_id` (all kernels together). Omniprobe's own
   warnings are in `omniprobe_messages` and quoted in errors.
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
- **Excess cache lines are L2 traffic, not necessarily DRAM (HBM) traffic.** If the wasted lines
  stay in L2 (the data is reused soon), making the access contiguous may not speed up an HBM-bound
  kernel. Judge a fix by HBM bytes and kernel time before and after (metrix
  `memory.bytes_transferred_hbm`, `memory.hbm_read_bandwidth`, kernel duration), not by excess lines
  alone. Loads of index arrays are memory traffic too.
- A load and a store on the same line are separate sites.

## Blind spots — when Omniprobe reports nothing for an access you suspect

- **Only global and LDS accesses are analyzed.** Scratch (private) and constant accesses are not
  reported.
- **Generic (flat) pointers** — every access at `-O0`, and accesses through a pointer loaded from
  memory (`float* a = table->field[k]; a[i]`) — are classified per wave at run time and analyzed as
  global or LDS (Omniprobe e807927 and later). A wave whose lanes point into different address
  spaces is counted in `unanalyzed_flat_accesses` and not analyzed (warned). A binary instrumented by
  an older Omniprobe counts all its flat accesses there.
- **Code built without `-g`**: global accesses cannot be matched to a source line and are dropped;
  the summary warns with per-kernel `unattributed_accesses`. Rebuild with `-g`. A few accesses can
  stay unattributed in a kernel that has `-g` (the warning then says `-g` is not missing).
- **Kernels that ran uninstrumented** (built without the plugin, or excluded by `--kernel`) are
  absent; `warnings` / `uninstrumented_kernels` name the first kind.
- **Architecture**: cache-line analysis exists for gfx906/908 (64-B lines) and gfx90a/940-942
  (128-B). Elsewhere only LDS analysis is reported.
- If a kernel is absent from the results entirely, check Omniprobe's stderr for
  `No instrumented alternative found for <kernel>` and see Troubleshooting.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `NoInstrumentedKernelsError` / "no instrumented kernels" | Binary built without the plugin, or wrong plugin | Rebuild with `$(omniprobe-ik flags memory)`; `make clean` first |
| "omniprobe executable not found on PATH" / `ImportError: omniprobe.api` | `<prefix>/omniprobe/bin` of an Omniprobe install is not on `PATH` (MCP servers start with a minimal env), or Omniprobe is not installed | `omniprobe-ik check` reports what is missing; the error message names the install page |
| Link error `duplicate symbol: __hip_cuid_...` | `-fgpu-rdc` with ≥ 2 files on Omniprobe older than 3003058 | Update Omniprobe, or drop `-fgpu-rdc` unless the project needs it |
| `WARNING: … ran WITHOUT instrumentation` / `uninstrumented_kernels` non-empty | That kernel's file was built without the plugin | Rebuild that file with the flags (every file that defines a kernel) |
| A kernel or line is missing | Blind spot above, or `--kernel` regex too narrow | See Blind spots; filter on demangled names |
| `KernelFilterMatchedNothingError` / "matched none of the instrumented kernels" | The `--kernel` regex matches no dispatched kernel (e.g. `^name` on a template kernel `void name<…>`) | The error lists the dispatched kernels; the binary is instrumented, do not rebuild |
| `RuntimeError: omniprobe exited with code N` | The application itself failed | Run the binary alone first; fix arguments/inputs |
| "no instrumented kernels" / "no results" naming `ROCM_PATH` or `llvm-objdump`, though the binary was built with the flags | Omniprobe found no ROCm with `llvm/bin/llvm-objdump` (Omniprobe 149cf2c+ searches `ROCM_PATH`, the ROCm it links against, `hipcc`/`rocminfo` on PATH, `/opt/rocm*`; older builds use only `ROCM_PATH` or `/opt/rocm`) | Set `ROCM_PATH=/opt/rocm-<version>` in the shell or MCP `env`; do not rebuild |
| `omniprobe exited with code 1: omniprobe: ERROR: ...` | Omniprobe rejected its configuration before the application ran (unknown analyzer, missing handler `.so`, unusable Python handler) | The ERROR line names the problem |
| No GPU / runs on the wrong GPU | `HIP_VISIBLE_DEVICES` set to the same id as `ROCR_VISIBLE_DEVICES` selects nothing | Set only `ROCR_VISIBLE_DEVICES` |

## Other analyzers, and writing your own

`omniprobe-ik run <analyzer> ./app` (MCP `run_analyzer`, Python `run_analyzer`) runs any Omniprobe
analyzer: `Heatmap` (accesses per 1 MiB page), `AddressLogger` (every wave's addresses; Python
`log_addresses`), `BasicBlockLogger` (every wave's block entries; `log_basic_blocks`) — memory build
for the first two, basic-block build for the last.

When no analyzer counts what you need (e.g. executions by partially active waves, a stride
histogram, accesses to one buffer), **write a message handler**: about 60 lines of C++ (below), or
a few lines of Python (after it), which sees the same per-wave fields.

```cpp
#include "dh_comms.h"           // Omniprobe installs these in <prefix>/omniprobe/include/dh_comms
#include "message_handlers.h"
class my_counter_t : public dh_comms::message_handler_base {
 public:
  my_counter_t(std::string kernel, uint64_t dispatch) : kernel_(std::move(kernel)) {}
  bool handle(const dh_comms::message_t &m) override {
    const auto &h = m.wave_header();
    if (h.user_type != dh_comms::message_type::address) return false;  // not ours
    unsigned rw = h.user_data & 0x3;            // 1 read, 2 write
    unsigned space = (h.user_data >> 2) & 0xf;  // dh_comms::address_space::global / shared / constant / flat
    unsigned bytes = h.user_data >> 6;          // access size
    size_t lanes = m.no_data_items();           // active lanes; m.data_item(i) -> const uint64_t* address
    // h.exec (mask), h.dwarf_line, h.dwarf_column, h.timestamp are also available
    /* count something */ return true;
  }
  void report() override { std::printf("MY_COUNTER {...}\n"); }  // once per dispatch
 private:
  std::string kernel_;
};
extern "C" __attribute__((visibility("default"))) void getMessageHandlers(const std::string &kernel, uint64_t dispatch,
    std::vector<dh_comms::message_handler_base *> &out) { out.push_back(new my_counter_t(kernel, dispatch)); }
```

- Build it as a shared library with the command `omniprobe-ik handler-flags` prints (hipcc,
  `-fPIC -shared`, Omniprobe's `include/dh_comms`, `-ldh_comms`; Python `handler_build_flags()`).
- Build the application with `$(omniprobe-ik flags memory)` (address messages) or
  `basic-blocks` (block messages, `user_type == basic_block_start`), then run
  `omniprobe-ik run /absolute/path/libmy_counter.so ./app`. What `report()` prints is on the
  run's stdout (Python `run.stdout`, MCP `stdout`); Omniprobe's own messages are on stderr.
- **Test it first** on a small kernel whose answer you can compute by hand (e.g. 1000 threads in
  blocks of 256 → 16 waves, the last with 40 active lanes), then run it on the real application.
- Every wave also loads its kernel arguments (`constant` space) — filter on `space` as needed.

**Python handlers — the quicker path.** A Python module runs through Omniprobe's `PythonHandler`:
`omniprobe-ik run ./my_handler.py ./app` (Python `run_analyzer("my_handler.py", "./app")`, MCP
`run_analyzer` with the `.py` path), on a **memory** build with `-g`. It defines at least one of
these (Omniprobe b9492cd and later for `handleAddressWave`), and its prints are on stdout:

| Function | Called | `msg` keys |
|---|---|---|
| `handleAddressWave(msg)` | once per wave-level memory instruction | `addresses` (one int per active lane), `lanes`, `exec`, `access_type` (`"read"`/`"write"`/`"readwrite"`), `memory_space` (`"global"`, `"shared"`, `"constant"`, `"scratch"`, `"flat"`), `access_size` (bytes), `source_file`, `line`, `column`, `block_idx`, `wave_num`, `cu_id`, `se_id`, `xcc_id`, `timestamp` (`s_memtime` ticks), `kernel`, `dispatch_id` |
| `handleAddressMessage(msg)` | once per active lane | `address`, `kernel`, `dispatch_id`, `kernel_dispatch` |
| `handleReport(msg)` | once per dispatch (optional) | `kernel`, `dispatch_id`, `kernel_dispatch` |

```python
from collections import Counter
sizes = Counter()                       # module state persists across dispatches
def handleAddressWave(msg):             # once per wave-level access
    if msg["memory_space"] == "global":
        sizes[(msg["line"], msg["access_type"], msg["access_size"])] += 1
def handleReport(msg):                  # end of each dispatch
    print("MY_SIZES", msg["kernel"], msg["dispatch_id"], dict(sizes), flush=True); sizes.clear()
```

- Unflushed prints are lost: print with `flush=True`.
- Kernel-argument loads arrive as `memory_space == "constant"` (all lanes of every wave).
- ~10 µs per call, so a per-wave call is up to 64× cheaper than per-lane calls; narrow with
  `--kernel` and `--dispatches 1` on large applications. Write the C++ handler above for high
  message rates.
- A module that cannot be imported or defines none of the message functions stops the run (exit 1,
  `omniprobe: ERROR: ... PythonHandler: ...`).
- Untested: a Python handler inside an application that is itself Python (e.g. Triton) may
  deadlock on the GIL.
- Test it like the C++ handler: `omniprobe/examples/06_python_handler/` checks a per-line counter on
  a kernel with hand-computed counts (16 waves, the last with 40 active lanes).

## Analyses and their plugins

| Analysis | Build flags | CLI | MCP tool | Python |
|---|---|---|---|---|
| Memory (coalescing + LDS bank conflicts) | `omniprobe-ik flags memory` (AddressMessages plugin) | `omniprobe-ik memory` | `analyze_memory` | `analyze_memory`, `summarize_memory` |
| Basic blocks (execution counts) | `omniprobe-ik flags basic-blocks` (BBStart plugin) — a **separate build** | `omniprobe-ik basic-blocks` | `analyze_basic_blocks` | `analyze_basic_blocks` |

`list_analyzers` (MCP) reports these and the blind spots.

**Basic-block analysis = exact per-block execution counts with correct source lines; not timing.**
Each block reports `wave_visits` (how many times a wave entered it) and `active_lanes` (lanes
summed over those entries; `active_lanes / wave_visits` = mean occupancy, low means divergence).
There are no cycle counts: for where time goes use metrix or rocprof kernel timing, and linex for
per-line cycles. Build with `omniprobe-ik flags basic-blocks`; never run it on a binary built for
memory analysis.

Further detail: `docs/tools/omniprobe.md` in the IntelliKit repository
(https://github.com/AMDResearch/intellikit/blob/main/docs/tools/omniprobe.md).
