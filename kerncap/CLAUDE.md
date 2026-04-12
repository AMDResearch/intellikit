# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is kerncap

kerncap is a kernel extraction and isolation tool for HIP and Triton applications on AMD GPUs. It profiles running applications, intercepts kernel dispatches at the HSA level (for HIP) or Python level (for Triton), captures complete runtime state (grid/block dims, all kernel arguments including device buffers), and generates standalone reproducer projects that can rebuild and replay kernels in isolation.

The extraction workflow: **Profile** → **Capture** → **Find Source** → **Generate** → **Validate**

## Build and Installation

kerncap is a hybrid C++/Python project using scikit-build-core to build the HIP interception library (`libkerncap.so`) and install the Python CLI.

```bash
# Standard install (builds libkerncap.so from src/)
pip install .

# Editable install for development
pip install -e .[dev]
```

The build uses CMake to compile `src/kerncap.hip` into `libkerncap.so`, which is then installed into the Python package wheel at `kerncap/lib/libkerncap.so`.

## Running Tests

```bash
# Unit tests (no GPU required) - test Python logic only
PYTHONPATH=. pytest tests/unit/ -v

# Local integration test (requires ROCm + AMD GPU)
PYTHONPATH=. pytest tests/integration/test_vector_add.py -v

# Docker-based integration tests (requires Docker + AMD GPU)
# These run full pipeline in ROCm containers against real workloads
PYTHONPATH=. pytest tests/integration/test_flash_attn.py -v      # Triton Flash Attention
PYTHONPATH=. pytest tests/integration/test_ck_gemm.py -v         # Composable Kernel GEMM

# All Docker tests
PYTHONPATH=. pytest tests/integration/ -m docker -v

# Skip Docker tests
PYTHONPATH=. pytest tests/integration/ -m "not docker" -v

# Show container output in real time (useful for debugging Docker tests)
PYTHONPATH=. pytest tests/integration/test_flash_attn.py -v -s
```

## CLI Commands

```bash
# Profile application to rank kernels by execution time
kerncap profile -- ./my_app --args
kerncap profile --output profile.json -- ./my_app

# Extract a kernel into standalone reproducer
kerncap extract flash_attn_fwd \
  --cmd "./my_app --args" \
  --source-dir ./src \
  --output ./isolated/flash_attn_fwd \
  --language triton \
  --dispatch 0

# Extract HIP kernel with extra preprocessor defines (e.g., llama.cpp/ggml)
kerncap extract mul_mat_q \
  --cmd "./llama-bench -m model.gguf -p test" \
  --source-dir ./ggml/src \
  -D GGML_USE_HIP -D GGML_CUDA_FA_ALL_QUANTS

# Replay captured kernel
kerncap replay ./isolated/mul_mat_q
kerncap replay ./isolated/mul_mat_q --hsaco optimized.hsaco

# Validate: smoke test for VA-faithful (HIP) reproducers
kerncap validate ./isolated/mul_mat_q

# Validate: correctness check — compare variant HSACO against captured baseline
kerncap validate ./isolated/mul_mat_q --hsaco optimized.hsaco

# Validate: Triton reproducers (compares outputs with tolerance)
kerncap validate ./isolated/flash_attn_fwd --tolerance 1e-4 --rtol 1e-3

# Enable verbose (DEBUG) logging for any command
kerncap -v extract ...
```

## Architecture

### Python Package (`kerncap/`)

- **cli.py**: Click-based CLI with `profile`, `extract`, `validate` commands
- **profiler.py**: Wrapper around `rocprofv3 --kernel-trace --stats`, parses CSV output
- **capturer.py**: Orchestrates capture process - dispatches to HIP (via `libkerncap.so`) or Triton (via Python hook)
- **triton_capture.py**: Generates hook script that monkey-patches `triton.runtime.jit.JITFunction.run` to intercept kernel launches
- **source_finder.py**: Locates kernel source files and translation units
  - **HIP**: Searches for `__global__` declarations, traces local `#include` dependencies recursively, finds the `.cu` translation unit via `compile_commands.json` or reverse-include search
  - **Triton**: Parses Python AST for `@triton.jit`/`@triton.autotune` decorators, traces imports (including relative imports)
- **reproducer.py**: Generates standalone reproducer projects with capture data, VFS overlay for recompilation, and Makefile
- **validator.py**: For VA-faithful (HIP) captures: baseline validation is a smoke test (replay succeeds); with `--hsaco`, runs two replays (captured vs variant) and compares byte-for-byte. For Triton: compares reproducer outputs against captured reference with tolerance

### HIP Interception Library (`src/`)

- **kerncap.hip**: HSA tool library loaded via `LD_PRELOAD` (rocprofiler-sdk registration)
  - Hooks `hsa_queue_create` to install packet intercept callback (same pattern as rocscope/Accordo)
  - Hooks `hsa_amd_memory_pool_allocate` to track device buffer sizes
  - On target kernel dispatch: interposes completion signal, waits for kernel finish, walks kernarg buffer
  - Snapshots all tracked device memory regions to disk for VA-faithful replay
  - Writes captured state to JSON + binary dumps on disk
- **kerncap.hpp**: C++ declarations
- **kerncap_log.hpp**: Logging macros

Built with CMake + HIP language support. Requires ROCm 7.0+ (HSA headers and rocprofiler-sdk, present in standard ROCm installations).

### Key Technical Details

**Triton autotuner reproducibility**: Triton's `@triton.autotune` selects configs by benchmarking, but different tile sizes change floating-point accumulation order, causing large numerical differences in FP16. The capturer records the winning config and the reproducer pins it exactly by calling `kernel.fn[grid](**config)` directly, bypassing autotuner re-execution.

**HIP argument capture**: kerncap performs a full device memory snapshot at capture time, copying all tracked GPU allocations to disk. Embedded device pointers are inherently captured as part of the full memory snapshot — no DWARF metadata or pointer chasing needed.

**Translation unit discovery**: For HIP kernels, the source finder locates the `.cu` translation unit that actually compiles the kernel (not just the `.cuh` header where it's defined). It searches `compile_commands.json` for entries whose source includes the kernel header, and for templated kernels with multiple instantiation files (e.g., llama.cpp's `mmq-instance-*.cu`), uses the mangled name from `dispatch.json` to select the correct one. The reproducer copies the main translation unit as `kernel_variant.cpp` and all traced `#include` dependencies into `deps/`, then generates a Clang Virtual File System (`vfs.yaml`) overlay that maps every local copy over its original path during a hijacked recompile, ensuring 100% flag and dependency fidelity. Edits to `kernel_variant.cpp` or any file in `deps/` take effect on `make recompile`.

**Source finding depth**: HIP `#include` tracing goes 5 levels deep by default (local includes only; system includes like `<hip/hip_runtime.h>` are assumed present in ROCm). Triton import tracing follows the full transitive closure of `ImportFrom` nodes.

## Validation Targets

The integration tests verify kerncap against:
- **Triton**: Flash Attention forward kernel from ROCm/flash-attention (Triton backend) in `rocm/pytorch` container
- **HIP**: Composable Kernel GEMM XDL FP16 from ROCm/composable_kernel in `rocm/composable_kernel:ck_pytorch` container
- **HIP (embedded pointers)**: Batched vector scale kernel in local ROCm environment, testing T** (double-pointer) arguments via VA-faithful replay

## Common Issues

**Validation failures with tight tolerances**: Almost always due to Triton autotuner config differences. Use `--tolerance` to relax `atol` when comparing across different configs or hardware.

**NaN values in validation output**: The validator detects NaN elements and reports them explicitly. Common causes: uninitialized device memory, half-precision overflow, or buffer size misinterpretation (wrong dtype). When NaNs are present, the validator reports the count and sets `max_error` to `nan`.


**Docker test failures**: Ensure `/dev/kfd` (AMD GPU device) is accessible and Docker can mount it. Tests automatically skip if Docker or GPU is unavailable.
