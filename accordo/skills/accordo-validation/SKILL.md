---
name: accordo-validation
description: Validate GPU kernel correctness by comparing reference and optimized outputs. Use when verifying that an optimized or modified kernel matches a reference implementation, including checking several tuned variants against one baseline before benchmarking them.
---

# Accordo: GPU Kernel Validation

Capture and compare kernel outputs from reference and optimized binaries to validate correctness. Uses kernelDB for automatic kernel extraction; supports configurable tolerance and execution-time comparison.

## When to Use

- User has a reference and an optimized (or modified) GPU kernel and wants to check they produce the same results
- Regression testing after kernel or build changes
- Validating multiple optimization variants against one baseline

## Prerequisites

- AMD GPU with ROCm 7.2 or later, Ubuntu 22.04 or 24.04, and Python 3.10 or later.
- `cmake`, `libdwarf-dev`, and `libzstd-dev` for the C++ build; the install fails without them.
- Install: `pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=accordo"`

## Instructions

1. **Require two or more binaries:** one reference (e.g. `./app_ref`) and one or more to validate (e.g. `./app_opt`). All must expose the same kernel by name.
2. **Ensure binaries are built with debug symbols** (`-g`) so kernel arguments can be extracted.
3. **Choose execution path:**
   - If an Accordo MCP server is available, call its `accordo-mcp:validate_kernel_correctness` tool, which performs capture-and-compare with the same semantics described below.
   - Otherwise use the Python API or the `accordo validate` CLI (`accordo validate --help` for flags: `--kernel-name`, `--ref-binary`, `--opt-binary`, `--atol`, `--rtol`, `--equal-nan`, `--timeout`, `--working-dir`, `--kernel-args`, `--log-level`).

### Python API

```python
from accordo import Accordo

# Validator for the kernel to validate (binary used to extract signature)
validator = Accordo(binary="./app_ref", kernel_name="reduce_sum")

# Optional: set working directory if binaries expect it
validator = Accordo(binary="./app_ref", kernel_name="reduce_sum", working_directory="./run")

# Capture snapshots
ref = validator.capture_snapshot(binary="./app_ref")
opt = validator.capture_snapshot(binary="./app_opt")

# Compare with numpy.allclose semantics (defaults atol=1e-8, rtol=1e-5)
result = validator.compare_snapshots(ref, opt, atol=1e-8, rtol=1e-5)

if result.is_valid:
    print("PASS:", result.num_arrays_validated, "arrays matched")
else:
    print(result.summary())
```

For multiple optimizations, capture the reference once and compare each optimized snapshot against it.

### Snapshot and result attributes

- **Snapshot:** `arrays`, `execution_time_ms`, `grid_size`, `block_size`
- **ValidationResult:** `is_valid`, `num_arrays_validated`, `num_mismatches`, `mismatches`, `success_rate`; use `summary()` for a human-readable report.

## Workflow

1. Build reference and optimized binaries with the same kernel name and `-g`.
2. Create an `Accordo(binary=ref_binary, kernel_name="...")` validator; set `working_directory` if needed.
3. Capture reference snapshot with `capture_snapshot(binary=ref_binary)`.
4. For each variant, capture with `capture_snapshot(binary=opt_binary)` and compare with `compare_snapshots(ref, opt, atol=..., rtol=...)`.
5. If `result.is_valid` is false, use `result.summary()` and `result.mismatches` to diagnose.
6. Use relative paths for binaries and working directory so the skill is portable.

## Notes

- kernelDB is used automatically; no separate kernelDB setup is required when using the Python API.
- Loosen `atol`/`rtol` for floating-point comparisons when appropriate (e.g. `rtol=1e-4` when the optimization reorders single-precision reductions). `tolerance=` still works as a legacy alias for `atol`.
- Use `timeout_seconds` in `capture_snapshot` if the run may hang.
