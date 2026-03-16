# kerncap

Kernel extraction and isolation tool for HIP and Triton applications on AMD GPUs.

kerncap profiles a running application, intercepts a target kernel dispatch, captures its complete runtime state (full device memory snapshot, kernarg buffer, HSACO), and generates a standalone reproducer that can replay the kernel in isolation using VA-faithful HSA dispatch.

## How it works

```
1. Profile          rocprofv3 --kernel-trace --stats → rank kernels by duration
2. Capture          HIP:    HSA_TOOLS_LIB=libkerncap.so → intercept target dispatch,
                            snapshot all tracked device memory + kernarg buffer + HSACO
                    Triton: Python-level hook on JITFunction.run → capture all tensor,
                            scalar, and constexpr args; pin autotuner config
3. Find source      HIP: __global__ grep + #include tracing
                    Triton: @triton.jit AST match + import tracing (incl. relative imports)
4. Generate         Jinja2 templates → standalone .hip+Makefile or .py reproducer
5. Validate         Build, run reproducer, np.allclose against captured reference
```

## Extract methodology

The extract stage is where most of the interesting work happens. It takes a
kernel name and a runnable command, and produces a fully self-contained
reproducer project. Under the hood it runs three sub-stages — capture, find
source, generate — each with language-specific paths for HIP and Triton.

### Capture

Snapshots the full runtime state of a single kernel dispatch for later replay.

**HIP kernels** are captured at the HSA level. `libkerncap.so` (loaded via
`HSA_TOOLS_LIB`) hooks `hsa_queue_create` to install a packet intercept
callback. When the target dispatch arrives, kerncap interposes a completion
signal, waits for the kernel to finish, then walks the kernarg buffer.
All device memory allocations are tracked via `hsa_amd_memory_pool_allocate`
and `hsa_amd_vmem_*` hooks. At capture time, a full device memory snapshot is
taken — every tracked allocation is D2H copied. The replay binary restores all
memory at the original virtual addresses using HSA VMEM APIs, then dispatches
the kernel with the captured HSACO. No DWARF metadata or argument parsing needed.

**Triton kernels** are captured at the Python level via monkey-patching
`JITFunction.run` and `Autotuner.run`. Inputs (tensors, scalars, constexprs)
are serialized before launch and reference outputs saved after. For autotuned
kernels the winning config is recorded so the reproducer can pin it exactly.

### Find source

Locates the kernel's source so the reproducer can compile (HIP) or import (Triton) it.

**HIP**: Searches the source tree for `__global__` declarations matching the
demangled kernel name, then traces local `#include "..."` directives
recursively (depth 5) to collect all required headers.

**Triton**: Parses Python files under `--source-dir` with the `ast` module,
matching `@triton.jit`/`@triton.autotune` decorators. `ImportFrom` nodes
(including relative imports) are traced to resolve the full dependency set.

### Generate

The captured data and located source files are assembled into a standalone
project using Jinja2 templates.

**HIP kernels** produce a VA-faithful replay project using `kerncap-replay`.
The captured HSACO, kernarg buffer, and full device memory snapshot are stored
in `capture/`. `make run` replays the kernel at the original virtual addresses
using HSA VMEM APIs — no kernel source compilation needed.

When `--source-dir` is provided, kerncap additionally finds the `.cu`
translation unit (via `compile_commands.json` or reverse-include search)
and produces:

- `kernel_variant.cpp` — a copy of the main kernel source file for editing
- `deps/` — copies of all `#include` dependency headers (traced up to 5 levels deep)
- `vfs.yaml` — a Clang Virtual File System overlay that maps all local copies over the originals during recompilation

This enables the [optimization workflow](#optimization-workflow) below.

| Output | Always | With `--source-dir` |
|--------|--------|---------------------|
| Captured state (`capture/`) | Yes | Yes |
| Editable source (`kernel_variant.cpp`) | — | Yes |
| Dependency headers (`deps/`) | — | Yes (when deps exist) |
| VFS overlay (`vfs.yaml`) | — | Yes |
| Makefile | Yes | Yes |

**Triton kernels** produce a `reproducer.py` that imports the kernel from the
copied source tree, loads tensor arguments from binary dumps, and calls the
kernel. For autotuned kernels, the reproducer calls `kernel.fn` directly with
pinned config kwargs, bypassing re-tuning entirely (see
[Triton autotuner and reproducibility](#triton-autotuner-and-reproducibility)).

## Install

Builds `libkerncap.so` from source against the host ROCm (requires `hipcc`, `cmake`, HSA headers — all present in standard ROCm images). No PyTorch or Triton dependency. No network access needed during the C++ build (nlohmann/json is vendored).

```bash
# From local source
pip install .

# Editable install for development
pip install -e .[dev]
```

## Usage

### Profile

```bash
# Rank kernels by total GPU time
kerncap profile -- ./my_app --args

# Save profile to JSON for scripting or later analysis
kerncap profile --output profile.json -- python train.py --batch-size 64
```

### Extract

```bash
# Triton kernel — language is auto-detected from source
kerncap extract flash_attn_fwd \
  --cmd "python train.py --batch-size 64" \
  --source-dir ./flash_attn \
  --output ./isolated/flash_attn_fwd

# HIP kernel with preprocessor defines (e.g., llama.cpp/ggml)
kerncap extract mul_mat_q \
  --cmd "./llama-bench -m model.gguf -p 512" \
  --source-dir ./ggml/src \
  -D GGML_USE_HIP -D GGML_CUDA_FA_ALL_QUANTS

# Capture-only — no source lookup, just HSACO + memory snapshot for replay
kerncap extract mul_mat_q \
  --cmd "./llama-bench -m model.gguf -p 512"

# Capture the 3rd dispatch of a kernel (0-indexed)
kerncap extract gemm_kernel \
  --cmd "./my_app" \
  --source-dir ./src \
  --dispatch 2

# Force language when auto-detection is ambiguous
kerncap extract my_kernel \
  --cmd "./my_app" \
  --source-dir ./src \
  --language hip
```

### Replay

```bash
# Replay a captured kernel
kerncap replay ./isolated/mul_mat_q

# Replay with a recompiled HSACO (after editing kernel_variant.cpp)
kerncap replay ./isolated/mul_mat_q --hsaco optimized.hsaco

# Benchmark: run 100 iterations for stable timing
kerncap replay ./isolated/mul_mat_q --iterations 100

# Machine-readable JSON output (for scripting / CI)
kerncap replay ./isolated/mul_mat_q --json

# Dump post-execution device memory for external comparison
kerncap replay ./isolated/mul_mat_q --dump-output
```

### Validate

```bash
# Smoke test: confirm a VA-faithful reproducer replays without crashing
kerncap validate ./isolated/mul_mat_q

# Correctness check: compare recompiled variant against captured baseline
# (runs replay twice — captured HSACO vs variant — and fails on byte differences)
kerncap validate ./isolated/mul_mat_q --hsaco optimized.hsaco

# Triton reproducers: compare outputs against captured reference
kerncap validate ./isolated/flash_attn_fwd
kerncap validate ./isolated/flash_attn_fwd --tolerance 1e-3 --rtol 1e-2

# Verbose logging — works with any command
kerncap -v extract ...
kerncap -v replay ./isolated/mul_mat_q --iterations 10
```

> **VA-faithful vs Triton validation**: For HIP kernels (VA-faithful captures),
> baseline `kerncap validate` is a smoke test only. To validate correctness,
> use `--hsaco` to compare a variant against the captured baseline (byte-exact).
> For Triton reproducers, `kerncap validate` compares outputs against captured
> reference data using `--tolerance` (atol) and `--rtol`.

## Optimization workflow

When `--source-dir` is provided, the extract output is designed for a tight
edit-recompile-validate loop — either manual or LLM-assisted.

```
kernel_variant.cpp      Editable kernel source (main translation unit)
deps/                   Editable dependency headers (#include chain)
vfs.yaml                Clang VFS overlay (maps all local copies over originals)
capture/                VA-faithful memory snapshot + dispatch metadata
Makefile                make run | make recompile | make run-variant
```

The loop:

```bash
cd ./isolated/my_kernel

# 1. Confirm the captured kernel replays correctly
make run

# 2. Edit kernel_variant.cpp and/or files in deps/
#    (Do NOT change the kernel function signature)

# 3. Recompile into a new HSACO (fast — single kernel, no application rebuild)
#    Uses the VFS overlay to substitute your edited files during a hijacked
#    recompile of the original translation unit, preserving all flags and deps.
make recompile

# 4. Replay with the optimized HSACO
make run-variant

# 5. Validate correctness (compares captured vs variant replay byte-for-byte)
kerncap validate . --hsaco optimized.hsaco
```

**`kernel_variant.cpp`** is a copy of the main kernel source file.
**`deps/`** contains copies of all `#include` dependency headers traced from
the main file (up to 5 levels deep). The VFS overlay (`vfs.yaml`) maps every
local copy over its original path during recompilation, so edits to
`kernel_variant.cpp` or any file in `deps/` take effect on `make recompile`
while preserving the exact compiler flags and include paths from the original
build. The kernel function signature must be preserved (the replay binary
dispatches arguments by position and type).

`capture/dispatch.json` provides the launch configuration (grid/block dims,
kernarg size, GPU architecture) and is useful context for an LLM doing the
optimization.

## Project structure

```
src/kerncap.{hip,hpp}     HSA tool loaded via HSA_TOOLS_LIB (capture)
src/replay.cpp             VA-faithful HSA kernel replay binary (kerncap-replay)
kerncap/                   Python package (CLI, profiler, capturer, source finder,
                           reproducer generator, validator)
kerncap/templates/         Jinja2 templates for HIP and Triton reproducers
vendor/                    Vendored nlohmann/json headers
tests/                     Unit + integration tests (see tests/README.md)
```

## AI-assisted testing

A [Cursor agent skill](/.cursor/skills/test-kerncap/SKILL.md) is included that can run the full kerncap pipeline (reinstall, profile, extract, compile, validate) against any HIP or Triton workload on a live GPU. Ask Cursor to "test kerncap against `<app_cmd>`" and it will drive the end-to-end workflow automatically.

## Embedded device pointers and batched operations

kerncap uses VA-faithful replay: all device memory is captured in a full
snapshot and restored at the original virtual addresses during replay.
Embedded device pointers (e.g. `T**` in batched BLAS, structs with pointer
members) work automatically because the entire GPU address space is
reconstructed at its original layout — no pointer patching or relocation
tables needed.

## Triton autotuner and reproducibility

Triton's `@triton.autotune` selects a config by benchmarking (e.g.
`BLOCK_M=128, num_warps=4`). Different configs change FP accumulation order,
which can cause large numerical differences in FP16. kerncap captures the
winning config and pins it in the reproducer (`kernel.fn[grid](...)` with
explicit config kwargs), bypassing re-tuning entirely. Without this, the
reproducer could select a different config and produce outputs differing by
the full value range (observed `max_error ≈ 7` for Flash Attention LSE).

If validation fails with tight tolerances, use
`kerncap validate --tolerance <atol>` to relax the threshold.

> **NaN in validation output**: Common causes are uninitialized device memory,
> FP16 overflow, or wrong dtype inference. The validator reports NaN counts
> per argument and sets `max_error` to `nan`.

## Validation targets

- **Triton**: Flash Attention forward kernel (`ROCm/flash-attention`, Triton backend) in `rocm/pytorch` container
- **HIP**: Composable Kernel GEMM XDL FP16 (`ROCm/composable_kernel`) in `rocm/composable_kernel:ck_pytorch` container
- **HIP (embedded pointers)**: Batched vector scale kernel in local ROCm environment, testing T** (double-pointer) arguments via VA-faithful replay

llama.cpp/ggml kernels (template-qualified names like `mul_mat_q<(ggml_type)7, 32, true>`) are also supported via the `-D` flag for preprocessor defines.
