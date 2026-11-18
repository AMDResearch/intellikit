---
name: kerncap-isolate
description: Isolate a GPU kernel from a running HIP or Triton application into a standalone reproducer that can be replayed, edited, recompiled, and validated on its own. Use when extracting or capturing a hot kernel, building a minimal kernel reproducer, or iterating on one kernel without rerunning the whole application.
---

# Kerncap: GPU Kernel Isolation

Profile a running application, capture one kernel dispatch's complete runtime state (full device memory snapshot, kernarg buffer, HSACO), and generate a standalone reproducer that replays the kernel in isolation at its original virtual addresses. With source available, the reproducer becomes an edit-recompile-validate project for that single kernel.

## When to Use

- User wants to pull the hottest (or a named) kernel out of a large HIP or Triton application (e.g. llama.cpp, PyTorch/Triton, Composable Kernel)
- Need a minimal, standalone reproducer for a kernel bug or performance report
- Iterating on one kernel's performance without rebuilding or rerunning the whole application
- Checking that an edited kernel still produces byte-identical outputs to the captured original

## Instructions

1. **Ensure prerequisites:** ROCm 7.2.x or later, Ubuntu 22.04/24.04, Python 3.10+, and an AMD GPU (Instinct MI300X, MI325X, MI355X recommended; Radeon supported). `hipcc`, `cmake`, and `rocprofv3` must be on `PATH`. Install with `pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=kerncap"`, which builds `libkerncap.so` against the host ROCm.
2. **To edit and rebuild a HIP kernel, the application's build must provide `compile_commands.json`** (configure CMake with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`). Without it, capture, replay, and validate still work, but the `make recompile` target is not generated. Check this before extracting if the user wants to modify the kernel.
3. **Choose execution path:**
   - If a Kerncap MCP server (`kerncap-mcp`) is available, use `profile_kernels` to rank kernels, `extract_kernel` to generate the reproducer, and `validate_reproducer` to check it. There is no MCP replay tool; use the CLI or Python API for replay and timing.
   - Otherwise use the CLI or Python API from the environment where Kerncap is installed.

### CLI

```bash
# Rank kernels by total GPU time
kerncap profile -- ./my_app --args
kerncap profile --output profile.json -- ./my_app

# Extract into a standalone reproducer (HIP with source enables the edit loop)
kerncap extract mul_mat_q --cmd "./llama-bench -m model.gguf -p 512" \
  --source-dir ./ggml/src --output ./isolated/mul_mat_q -D GGML_USE_HIP

# Triton (language auto-detected from --source-dir)
kerncap extract flash_attn_fwd --cmd "python train.py" --source-dir ./flash_attn

# Replay in isolation; benchmark over many iterations
kerncap replay ./isolated/mul_mat_q --iterations 100
kerncap replay ./isolated/mul_mat_q --hsaco ./isolated/mul_mat_q/optimized.hsaco

# Validate (auto-detects optimized.hsaco / candidate.hsaco from the edit loop)
kerncap validate ./isolated/mul_mat_q
```

Options: `extract` takes `--cmd` (required), `--source-dir`, `--output`/`-o`, `--language` (`hip`|`triton`), `--dispatch` (index of the matching dispatch; default first match), `--defines`/`-D` (repeatable), `--timeout` (default 300 s). `replay` takes `--hsaco`, `--iterations`/`-n`, `--json`, `--dump-output`, `--hip-launch`. `validate` takes `--hsaco`, `--tolerance`/`-t`, `--rtol`, `-v`/`--verbose`. `profile` takes `--output`/`-o`, `--timeout`.

### Python API

```python
import os, subprocess
from kerncap import Kerncap

kc = Kerncap()

for k in kc.profile(["./my_app", "--args"])[:5]:
    print(f"{k.name}: {k.total_duration_ns / 1e6:.1f} ms ({k.percentage:.1f}%)")

result = kc.extract(
    "mul_mat_q",
    cmd=["./llama-bench", "-m", "model.gguf", "-p", "512"],
    source_dir="./ggml/src",
    output="./isolated/mul_mat_q",
    defines=["GGML_USE_HIP"],
)
print(result.output_dir, result.language, result.has_source, result.generated_files)

# After editing kernel_variant.cpp:
subprocess.run(["make", "recompile"], cwd=result.output_dir, check=True)
variant_hsaco = os.path.join(result.output_dir, "optimized.hsaco")

baseline = kc.replay(result.output_dir, iterations=100)
variant = kc.replay(result.output_dir, hsaco=variant_hsaco, iterations=100)
print(f"Speedup: {baseline.timing_us / variant.timing_us:.2f}x")

check = kc.validate(result.output_dir, hsaco=variant_hsaco)
print("Passed:", check.passed, "max_error:", check.max_error)
```

### Reproducer layout

| Output | HIP | Triton |
|--------|-----|--------|
| `capture/` (memory snapshot, `dispatch.json`, baseline HSACO) | Always | Always |
| `Makefile` (`run`, `replay`, `validate`) | Always | — |
| `kernel_variant.cpp`, `deps/`, `vfs.yaml` | With `--source-dir` | — |
| `recompile`, `run-variant`, `validate-variant` targets | With `compile_commands.json` | — |
| `reproducer.py`, `reference_output/` | — | Always |
| `kernel_variant.py` (or a copy of the original source file) | — | Always |

Read `capture/dispatch.json` and `capture/metadata.json` first, before reading source: `dispatch.json` holds the grid/block dimensions, total kernarg size, and GPU architecture; `metadata.json` holds the per-argument layout (`kernarg_slots`: offsets, sizes, kinds, types) when available.

## Workflow

1. **Pick the target.** If the user named a kernel, use it. Otherwise run `kerncap profile` and choose the kernel with the highest total time; tell the user which one and why. Kernel names are matched as substrings and `extract` captures the *first* matching dispatch, so for templated kernels use a long, template-qualified substring (e.g. `mul_mat_q<(ggml_type)7, 32, true>` rather than `mul_mat_q`). Use `--dispatch N` to capture a later dispatch.
2. **Extract.** Pass `--source-dir` and any `-D` defines the application is built with to get an editable project. Without source, the capture is still replayable and validatable, but not editable. For Triton, `--source-dir` may be omitted only if you pass `--language triton`; Kerncap then falls back to source snapshots recorded at JIT time (including `torch.compile`/Inductor kernels). Without either flag, the language is not detected and no snapshots are taken.
3. **Replay the baseline.** HIP: `make run` in the reproducer directory. Triton: `python3 reproducer.py`. For stable timing, use `kerncap replay <dir> --iterations 100`.
4. **Edit loop.** Do not change the kernel signature.
   - HIP: edit `kernel_variant.cpp` and/or files in `deps/`, run `make recompile` (writes `optimized.hsaco`), then `make run-variant`.
   - Triton: edit `kernel_variant.py` (or the copied source file), then rerun `python3 reproducer.py` (re-JITs and writes `candidate.hsaco`).
5. **Validate.** Run `kerncap validate <dir>`. It auto-detects `optimized.hsaco` or `candidate.hsaco` and compares output memory regions byte-for-byte against the captured run. Read the `Result: PASS/FAIL (...)` footer; add `-v` for per-region detail.
6. **Report:** the kernel chosen and why, the reproducer path, baseline vs variant timing (speedup), and the validation result, including any extraction warnings.
7. Use relative paths for commands and output directories so the skill is portable.

## Notes

- Without a rebuilt HSACO, `kerncap validate` is a smoke test only: it confirms the captured kernel replays, not that any change is correct.
- A "source not found" warning names its cause: a wrong `--source-dir`, a Triton kernel routed as HIP (pass `--language triton`), or a kernel with no source trail (Tensile, hand-written assembly, JIT-generated or vendor HSACO blobs). The capture is still replayable.
- Embedded device pointers (`T**` in batched BLAS, structs with pointer members) and Kokkos constant-memory launches work automatically; replay restores all memory at its original virtual addresses.
- Validation of default captures (those with `capture/dispatch.json`, HIP and Triton alike) is byte-exact; `--tolerance`/`--rtol` are ignored. A FAIL means the rebuilt kernel really produces different bytes (e.g. a changed FP accumulation order), so do not retry with looser tolerances. Tolerances apply only to legacy captures without `dispatch.json`, such as `--triton-backend python`.
- Triton autotune configurations are pinned at capture time, so replay uses the configuration the application selected.
- To profile the isolated kernel with Metrix or Linex (`rocprofv3`), replay with `--hip-launch` to avoid conflicts with the default HSA dispatch path.
- For very large device-memory snapshots, tune the staging buffer with `KERNCAP_SNAPSHOT_CHUNK_BYTES` (default 64 MiB). Raise `extract --timeout` for long-running applications.
