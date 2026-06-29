# Kerncap

Kerncap profiles a running application, intercepts a target kernel dispatch, captures its complete runtime state (full device memory snapshot, kernarg buffer, and HSACO), and generates a standalone reproducer that can replay the kernel in isolation using VA-faithful HSA dispatch.

## How it works

```
1. Profile       rocprofv3 --kernel-trace --stats → rank kernels by duration
2. Capture       HIP:    LD_PRELOAD=libkerncap.so → intercept target dispatch,
                         snapshot all tracked device memory + kernarg buffer + HSACO
                 Triton: LD_PRELOAD capture with code-object metadata +
                         compile-shim name mapping
3. Find source   HIP: __global__ grep + #include tracing
                 Triton: @triton.jit AST match + import tracing (incl. relative imports)
4. Generate      Jinja2 templates → standalone .hip+Makefile or .py reproducer
5. Validate      Build, run reproducer, np.allclose against captured reference
```

## Installation

Kerncap builds `libkerncap.so` from source against the host ROCm 7.0+ environment. It requires `hipcc`, `cmake`, HSA headers, and `rocprofiler-sdk`, which are included in standard ROCm images.

Install Kerncap using the following methods:

```bash
# From local source
pip install .

# Editable install for development
pip install -e .[dev]
```

## Usage

Kerncap provides a versatile interface where each operation is accessible through a Python API and a CLI command.

### Profile

Rank kernels by their total GPU execution time.

**Python API:**

```python
from kerncap import Kerncap

kc = Kerncap()
profile = kc.profile(["./my_app", "--args"])
for kernel in profile[:5]:
    print(f"{kernel.name}: {kernel.total_duration_ns / 1e6:.1f} ms ({kernel.percentage:.1f}%)")
```

**CLI:**

```bash
# Profile and print top kernels
kerncap profile -- ./my_app --args

# Save profile to JSON
kerncap profile --output profile.json -- ./my_app
```

### Extract

Capture a kernel's full runtime state and generate a standalone reproducer.

**Python API:**

```python
# HIP kernel with source (enables recompile workflow)
result = kc.extract(
    kernel_name="mul_mat_q",
    cmd=["./llama-bench", "-m", "model.gguf", "-p", "512"],
    source_dir="./ggml/src",
    output="./isolated/mul_mat_q",
    defines=["GGML_USE_HIP", "GGML_CUDA_FA_ALL_QUANTS"],
)
print(f"Output: {result.output_dir}  has_source: {result.has_source}")

# Triton kernel — language auto-detected from source
result = kc.extract(
    kernel_name="flash_attn_fwd",
    cmd=["python", "train.py", "--batch-size", "64"],
    source_dir="./flash_attn",
    output="./isolated/flash_attn_fwd",
)
```

**CLI:**

```bash
# HIP with source
kerncap extract mul_mat_q --cmd "..." --source-dir ./ggml/src -D GGML_USE_HIP

# Triton
kerncap extract flash_attn_fwd --cmd "..." --source-dir ./flash_attn

# Capture-only (no source)
kerncap extract mul_mat_q --cmd "..."

# Specific dispatch
kerncap extract gemm_kernel --cmd "..." --dispatch 2
```

**Language detection**: Kerncap auto-detects whether a kernel is HIP or Triton from `--source-dir` contents. To override, use `--language hip` or `--language triton` in the CLI or provide `language="triton"` in the Python API.

**Triton backend compatibility**: The `--triton-backend` flag is available for compatibility with the legacy Python capture backend. HSA is the new default, and the Python backend will soon be deprecated.

### Replay

Replay a captured kernel in isolation.

**Python API:**

```python
# Replay with captured HSACO
baseline = kc.replay("./isolated/mul_mat_q")
print(f"Baseline: {baseline.timing_us:.1f} us")

# Replay with a variant HSACO and compare
variant = kc.replay("./isolated/mul_mat_q", hsaco="./isolated/mul_mat_q/optimized.hsaco")
print(f"Variant:  {variant.timing_us:.1f} us")
print(f"Speedup:  {baseline.timing_us / variant.timing_us:.2f}x")
```

**CLI:**

```bash
# Replay with captured HSACO
kerncap replay ./isolated/mul_mat_q

# Replay with a variant HSACO
kerncap replay ./isolated/mul_mat_q --hsaco optimized.hsaco

# Benchmark over multiple iterations
kerncap replay ./isolated/mul_mat_q --iterations 100
```

**HIP launch mode**: If replay conflicts with `rocprofv3`, such as when profiling the reproducer itself, use `--hip-launch` to switch from the default HSA dispatch to the HIP runtime launch path.

### Validate

Validate the correctness of a reproducer or variant HSACO.

**Python API:**

```python
# Smoke test — confirm baseline replays without error
result = kc.validate("./isolated/mul_mat_q")
print("Passed:", result.passed)

# Correctness check — compare recompiled variant against captured baseline
result = kc.validate("./isolated/mul_mat_q", hsaco="./isolated/mul_mat_q/optimized.hsaco")
print("Passed:", result.passed)

# Triton — compare against captured reference with tolerance
result = kc.validate("./isolated/flash_attn_fwd", tolerance=1e-3, rtol=1e-2)
print("Passed:", result.passed)

# Edit loops auto-detect candidate.hsaco / optimized.hsaco
result = kc.validate("./isolated/flash_attn_fwd")
print("Passed:", result.passed)
```

**CLI:**

```bash
# Smoke test — confirm baseline replays without error
kerncap validate ./isolated/mul_mat_q

# Correctness check — compare variant against captured baseline
kerncap validate ./isolated/mul_mat_q --hsaco optimized.hsaco

# Triton — compare with relaxed tolerance
kerncap validate ./isolated/flash_attn_fwd --tolerance 1e-3 --rtol 1e-2

# Edit loops auto-detect candidate.hsaco / optimized.hsaco
kerncap validate ./isolated/flash_attn_fwd
```

**Validation modes**: For VA-faithful captures, the baseline `validate` operation acts as a smoke test until a rebuilt HSACO is available. To compare captured and rebuilt execution byte-for-byte, the `--hsaco` flag, or let Kerncap auto-detect `candidate.hsaco` or `optimized.hsaco`.

`kerncap validate <dir>` auto-detects rebuilt HSACOs from the edit loop: `candidate.hsaco` from Triton `python3 reproducer.py`, or `optimized.hsaco` from HIP `make recompile`. Validation is performed by comparing the captured kernel execution against the rebuilt HSACO, producing a byte-exact memory-region summary. 

The CLI prints a canonical `Result: PASS/FAIL (...)` footer for replay and validation. Per-region PASS lines are hidden unless validation fails or `kerncap validate -v` is used.

## Extract methodology

The extract stage takes a kernel name and a runnable command, producing a fully self-contained reproducer project. Under the hood, it runs three sub-stages (capture, find source, and generate), applying language-specific paths for HIP and Triton.

### Capture

Snapshots the full runtime state of a single kernel dispatch for later replay.

HIP kernels are captured at the HSA level. `libkerncap.so` is loaded via `LD_PRELOAD` with `rocprofiler-sdk` registration. Kerncap hooks into `hsa_queue_create` to install a packet intercept callback. When the target dispatch arrives, Kerncap interposes a completion signal, waits for execution to complete, and walks the kernarg buffer. All device memory allocations are tracked via `hsa_amd_memory_pool_allocate` and `hsa_amd_vmem_*` hooks. At capture time, a full device memory snapshot is taken and every tracked allocation is D2H copied. The replay binary restores all memory at the original virtual addresses using HSA VMEM APIs, then dispatches the kernel with the captured HSACO. No DWARF metadata or argument parsing is required.

Triton kernels are captured with `libkerncap.so` loaded via `LD_PRELOAD` and a Python compile shim installed through `sitecustomize.py`. The shim records `name_map.json` rows that connect Triton's user-facing function name, signature, constant expressions, launch attributes, source snapshot, and HSACO SHA-256 to the captured dispatch. A short-lived run observer records tensor layout metadata, so the editable Python reproducer can rebuild accurate `torch.as_strided` views. The captured kernarg buffer and memory snapshot remain the source of truth.

### Find source

Locates the kernel's source so the reproducer can compile (HIP) or import (Triton) it.

HIP searches the source tree for `__global__` declarations matching the demangled kernel name, then traces local `#include "..."` directives recursively (depth 5) to collect all required headers. For projects built with CMake, Kerncap uses `compile_commands.json` to find the translation unit and extract the exact compiler flags required for the `make recompile` target.

`compile_commands.json` is optional. It is required by the `make recompile` target (for editing and rebuilding kernels), but not by the capture/replay/validate workflows. You can generate `compile_commands.json` by adding `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` when configuring your CMake project. If missing, Kerncap displays: `"No compile command found (compile_commands.json missing or has no entry for this file). The 'make recompile' target will not be available."` In this case, basic validation of the reproducer will still work, but kernel edits cannot be tested.

Triton parses Python files under `--source-dir` with the `ast` module by matching `@triton.jit` or `@triton.autotune` decorators. `ImportFrom` nodes (including relative imports) are traced to resolve the full dependency set. Kerncap can also fall back to the source snapshots written by the compile shim if `--source-dir` is omitted or the original file comes from a temporary `torch.compile`or Inductor path.

If source lookup fails, Kerncap classifies the failure instead of emitting a generic warning. It distinguishes cases where DWARF source paths exist but `--source-dir` is wrong, scenarios where Triton kernels were routed as HIP, or cases where the captured code object has no source trail at all, such as Tensile, hand-written assembly, JIT-generated binaries, or third-party HSACO blobs.

### Generate

The captured data and located source files are assembled into a standalone project using Jinja2 templates.

HIP kernels generate a VA-faithful replay project using `kerncap-replay`. The project includes the captured HSACO, kernarg buffer, and a full snapshot of device memory stored in `capture/`. Running `make run` replays the kernel at its original virtual addresses using HSA VMEM APIs. There is no need to recompile the kernel source.

When `--source-dir` is provided, Kerncap goes a step further by locating the `.cu` translation unit (via `compile_commands.json` or reverse-include search) and generating:

- `kernel_variant.cpp`: a copy of the main kernel source file for editing
- `deps/`: copies of all `#include` dependency headers, traced up to 5 levels deep
- `vfs.yaml`: a Clang Virtual File System (VFS) overlay file that maps all local copies over the originals during recompilation

These artifacts enable the [optimization workflow](#optimization-workflow):

| Output | Always | With `--source-dir` |
|--------|--------|---------------------|
| Captured state (`capture/`) | Yes | Yes |
| Editable source (`kernel_variant.cpp`) | — | Yes |
| Dependency headers (`deps/`) | — | Yes (if dependencies exist) |
| VFS overlay (`vfs.yaml`) | — | Yes |
| Makefile | Yes | Yes |

Triton kernels generate a `reproducer.py` script that imports the kernel from the copied source tree, loads tensor arguments from binary dumps, and calls the kernel. The generated project also includes `capture/` and `reference_output/`. Editing `kernel_variant.py` (or the copied source file) and rerunning `python3 reproducer.py` re-JITs the kernel through Triton and generates `candidate.hsaco` for further validation.

## Optimization workflow

When `source_dir` is provided, `extract` generates a self-contained project for a fast edit-recompile-validate loop:

```
kernel_variant.cpp      Editable copy of the main kernel source file
deps/                   Copies of all #include dependency headers (up to 5 levels)
vfs.yaml                Clang VFS overlay — maps local copies over originals at compile time
capture/                VA-faithful memory snapshot, dispatch metadata, baseline HSACO
Makefile                make run | make recompile | make run-variant | make validate-variant
```

**Prerequisites for `make recompile`**: 

The recompilation workflow requires `compile_commands.json` from your project's build directory. If your project is built with CMake, regenerate it with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. Without this file, `make run` and `kerncap validate` can still work by using the captured HSACO, but `make recompile` won't be available.

For Triton captures, the edit loop is Python-native rather than Makefile-based:

```bash
cd ./isolated/flash_attn_fwd
# edit kernel_variant.py or the copied source file
python3 reproducer.py     # re-JITs through Triton and writes candidate.hsaco
kerncap validate .        # auto-detects candidate.hsaco
```

**Python API:**

```python
import subprocess, os
from kerncap import Kerncap

kc = Kerncap()

# 1. Extract (once)
result = kc.extract("mul_mat_q", cmd=[...], source_dir="./ggml/src", output="./isolated/mul_mat_q")
reproducer_dir = result.output_dir

# 2. Edit kernel_variant.cpp or files in deps/ (do not change the kernel signature)

# 3. Recompile — single kernel, no application rebuild
subprocess.run(["make", "recompile"], cwd=reproducer_dir, check=True)

# 4. Compare baseline vs variant
baseline = kc.replay(reproducer_dir)
variant  = kc.replay(reproducer_dir, hsaco=os.path.join(reproducer_dir, "optimized.hsaco"))
print(f"Baseline: {baseline.timing_us:.1f} us  Variant: {variant.timing_us:.1f} us")
print(f"Speedup: {baseline.timing_us / variant.timing_us:.2f}x")

# 5. Validate correctness
result = kc.validate(reproducer_dir, hsaco=os.path.join(reproducer_dir, "optimized.hsaco"))
print("Passed:", result.passed)
```

**CLI:**

```bash
cd ./isolated/mul_mat_q

make run            # replay baseline
# edit kernel_variant.cpp and/or deps/
make recompile      # recompile into optimized.hsaco
make run-variant    # replay variant
kerncap validate .   # auto-detects optimized.hsaco for correctness check
```

## AI-assisted optimization

The Python API is designed for LLM-driven workflows. A Cursor agent (or any LLM with code execution capabilities) can drive the full pipeline, which includes profiling, extracting, recompiling, benchmarking, and validating, entirely through the `Kerncap` class without shell scripting. The key inputs for this workflow are:

- The application command (`cmd`)
- The source directory (`source_dir`) and any preprocessor defines (`defines`)
- The reproducer directory for the edit-recompile-validate loop

`capture/dispatch.json` provides useful context for an LLM. It contains the kernel's launch configuration (grid/block dims, kernarg layout, and GPU architecture) with the captured HSACO, giving a complete picture of what the kernel does and how it is launched before any source is read.

## Technical details

### Embedded device pointers

Kerncap uses VA-faithful replay, ensuring all device memory is captured in a full snapshot and restored to its original virtual addresses during replay. Embedded device pointers (for example, `T**` in batched BLAS or structs with pointer members) work automatically. No pointer patching or relocation tables is required.

### Module variables and constant memory

Some frameworks launch kernels through module-scope variables instead of ordinary dynamic allocations. For example, Kokkos's `hip_parallel_launch_constant_memory`, used by applications such as LAMMPS, SPARTA, and ExaMPM, stores kernel arguments in a `__constant__`-style HSACO module variable populated by `hipMemcpyToSymbol`. Kerncap snapshots these executable module variables into `module_variables.json` with binary blobs. During replay, Kerncap restores only the variables whose executable SHA-256 matches the captured kernel.

### Large snapshots

Device-memory snapshots are streamed through a bounded staging buffer instead of copied as one full host allocation per region. The default chunk size is 64 MiB. Set `KERNCAP_SNAPSHOT_CHUNK_BYTES` when debugging captures with very large allocations.

### Triton autotuner reproducibility

Triton's `@triton.autotune` selects a configuration by benchmarking (for example, `BLOCK_M=128, num_warps=4`). Different configurations change FP accumulation order, which can cause large numerical differences in FP16. Kerncap captures the winning configuration and pins it in the reproducer, bypassing the re-tuning process.

If validation fails with tight tolerances, use `kerncap validate --tolerance <atol>` to relax the threshold.

Common causes of NaN in validation output include uninitialized device memory, FP16 overflow, or incorrect data type inference. The validator reports NaN counts per argument and sets `max_error` to `nan`.

### Validation targets

Kerncap supports a range of validation targets for different use cases:

- **Triton**: Flash Attention forward kernel (`ROCm/flash-attention`) in `rocm/pytorch` container
- **HIP**: Composable Kernel GEMM XDL FP16 (`ROCm/composable_kernel`) in `rocm/composable_kernel:ck_pytorch` container
- **HIP with embedded pointers**: Batched vector scale kernel in local ROCm environment, testing T** (double-pointer) arguments via VA-faithful replay

llama.cpp/ggml kernels (with template-qualified names like `mul_mat_q<(ggml_type)7, 32, true>`) are also supported via the `-D` flag to provide preprocessor definitions.

## Project structure

```
src/kerncap.{hip,hpp}     HSA tool loaded via LD_PRELOAD (`rocprofiler-sdk` registration)
src/replay.cpp             VA-faithful HSA kernel replay binary (kerncap-replay)
src/kernarg_metadata.*     AMDGPU code-object kernarg metadata parser
kerncap/                   Python package (CLI, profiler, capturer, source finder,
                           reproducer generator, validator)
kerncap/templates/         Jinja2 templates for HIP and Triton reproducers
vendor/                    Vendored nlohmann/json headers
tests/                     Unit + integration tests
```

## Related links

- [Kerncap: Automated Kernel Extraction and Isolation for AMD GPUs](https://arxiv.org/abs/2605.03208) (arXiv)
