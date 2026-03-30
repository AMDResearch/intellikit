# kerncap

Kernel extraction and isolation tool for HIP and Triton applications on AMD GPUs.

kerncap profiles a running application, intercepts a target kernel dispatch, captures its complete runtime state (full device memory snapshot, kernarg buffer, HSACO), and generates a standalone reproducer that can replay the kernel in isolation using VA-faithful HSA dispatch.

> For full documentation, see [amdresearch.github.io/intellikit/tools/kerncap](https://amdresearch.github.io/intellikit/tools/kerncap/).

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

## Install

Builds `libkerncap.so` from source against the host ROCm (requires `hipcc`, `cmake`, HSA headers — all present in standard ROCm images). No PyTorch or Triton dependency. No network access needed during the C++ build (nlohmann/json is vendored).

```bash
# From local source
pip install .

# Editable install for development
pip install -e .[dev]
```

## Usage

Each operation is available as both a Python API and a CLI command.

### Profile

Rank kernels by total GPU execution time.

```bash
# Profile and print top kernels
kerncap profile -- ./my_app --args

# Save profile to JSON
kerncap profile --output profile.json -- ./my_app
```

```python
from kerncap import Kerncap

kc = Kerncap()
profile = kc.profile(["./my_app", "--args"])
for kernel in profile[:5]:
    print(f"{kernel.name}: {kernel.total_duration_ns / 1e6:.1f} ms ({kernel.percentage:.1f}%)")
```

### Extract

Capture a kernel's full runtime state and generate a standalone reproducer.

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

```python
result = kc.extract(
    kernel_name="mul_mat_q",
    cmd=["./llama-bench", "-m", "model.gguf", "-p", "512"],
    source_dir="./ggml/src",
    output="./isolated/mul_mat_q",
    defines=["GGML_USE_HIP", "GGML_CUDA_FA_ALL_QUANTS"],
)
```

> **Language detection**: kerncap auto-detects whether a kernel is HIP or Triton from `--source-dir` contents. To override, pass `--language hip` or `--language triton` on the CLI (or `language="triton"` in the Python API).

### Replay

Replay a captured kernel in isolation.

```bash
# Replay with captured HSACO
kerncap replay ./isolated/mul_mat_q

# Replay with a variant HSACO
kerncap replay ./isolated/mul_mat_q --hsaco optimized.hsaco

# Benchmark over multiple iterations
kerncap replay ./isolated/mul_mat_q --iterations 100
```

```python
# Replay baseline vs variant and compare
baseline = kc.replay("./isolated/mul_mat_q")
variant = kc.replay("./isolated/mul_mat_q", hsaco="optimized.hsaco")
print(f"Speedup: {baseline.timing_us / variant.timing_us:.2f}x")
```

> **HIP launch mode**: If replay conflicts with `rocprofv3` (e.g. when profiling the reproducer itself), pass `--hip-launch` to use the HIP runtime launch path instead of the default HSA dispatch.

### Validate

Check correctness of a reproducer or variant HSACO.

```bash
# Smoke test — confirm baseline replays without error
kerncap validate ./isolated/mul_mat_q

# Correctness check — compare variant against captured baseline
kerncap validate ./isolated/mul_mat_q --hsaco optimized.hsaco

# Triton — compare with relaxed tolerance
kerncap validate ./isolated/flash_attn_fwd --tolerance 1e-3 --rtol 1e-2
```

```python
# Correctness check — compare variant against captured baseline
result = kc.validate("./isolated/mul_mat_q", hsaco="optimized.hsaco")
print("Passed:", result.passed)
```

> **HIP vs Triton validation**: For HIP kernels, baseline `validate` is a smoke test only. Pass `hsaco` to compare a recompiled variant against the captured baseline. For Triton reproducers, `validate` compares outputs against captured reference data using `np.allclose`.

## Optimization workflow

When `source_dir` is provided, `extract` produces a self-contained project for a tight edit-recompile-validate loop:

```bash
cd ./isolated/mul_mat_q

make run            # replay baseline
# edit kernel_variant.cpp and/or deps/
make recompile      # recompile into optimized.hsaco
make run-variant    # replay variant
kerncap validate . --hsaco optimized.hsaco  # correctness check
```

See [full documentation](https://amdresearch.github.io/intellikit/tools/kerncap/) for the complete Python API workflow and details on the generated project layout.

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
