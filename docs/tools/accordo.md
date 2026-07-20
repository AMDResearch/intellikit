---
myst:
    html_meta:
        "description": "Accordo validates GPU kernel correctness by capturing and comparing outputs from reference and optimized implementations on AMD GPUs with configurable tolerance."
        "keywords": "Accordo, AMD GPU, ROCm, kernel validation, correctness, optimization, HIP, KernelDB"
---

# Accordo (IntelliKit)

Accordo automatically validates GPU kernel correctness by capturing and comparing kernel outputs from reference and optimized implementations.

## Features

Accordo provides the following capabilities.

- **Automatic kernel extraction**: uses KernelDB to extract kernel signatures from binaries
- **Snapshot-based validation**: captures kernel outputs once and validates against multiple optimizations
- **Configurable tolerance**: sets precision requirements for floating-point comparisons (`atol`, `rtol`, `equal_nan`)
- **Performance tracking**: measures and compares execution times

## Requirements

Accordo requires the following.

- Python >= 3.10
- ROCm >= 7.2.0
- KernelDB (automatically installed)

## Installation

Accordo compiles C++ code (through KernelDB) during installation. Ensure `cmake`, `libdwarf-dev`, and `libzstd-dev` are pre-installed:

```bash
# System prerequisites (Debian/Ubuntu)
sudo apt-get update && sudo apt-get install -y cmake libdwarf-dev libzstd-dev

# Install
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=accordo"
```

## Quick start

Use the Python API or CLI to validate GPU kernels.

### Python API

```python
from accordo import Accordo

# Create validator for a specific kernel
validator = Accordo(binary="./app_ref", kernel_name="reduce_sum")

# Capture snapshots from reference and optimized binaries
ref = validator.capture_snapshot(binary="./app_ref")
opt = validator.capture_snapshot(binary="./app_opt")

# Compare with allclose-style controls
result = validator.compare_snapshots(ref, opt, atol=1e-6, rtol=1e-5)

if result.is_valid:
    print(f"PASS: {result.num_arrays_validated} arrays matched")
else:
    print(result.summary())
```

### CLI

```bash
accordo validate \
  --kernel-name reduce_sum \
  --ref-binary ./app_ref \
  --opt-binary ./app_opt \
  --atol 1e-6 --rtol 1e-5
```

### Testing multiple optimizations

You can capture a single reference snapshot and compare it against multiple optimized variants.

```python
validator = Accordo(binary="./ref", kernel_name="matmul")
ref = validator.capture_snapshot(binary="./ref")

for opt_binary in ["./opt_v1", "./opt_v2", "./opt_v3"]:
    opt = validator.capture_snapshot(binary=opt_binary)
    result = validator.compare_snapshots(ref, opt, atol=1e-6, rtol=1e-5)
    print(f"{opt_binary}: {'PASS' if result.is_valid else 'FAIL'}")
```

## CLI reference

The Accordo CLI provides the following options.

```
accordo validate \
  --kernel-name NAME \
  --ref-binary PATH_TO_EXECUTABLE \
  --opt-binary PATH_TO_EXECUTABLE \
  [--tolerance FLOAT]              # legacy alias for --atol
  [--atol FLOAT]                   # absolute tolerance (default: 1e-08)
  [--rtol FLOAT]                   # relative tolerance (default: 1e-05)
  [--equal-nan]                    # treat NaN == NaN
  [--timeout SECONDS]              # per snapshot, default: 30
  [--working-dir DIR]              # default: .
  [--kernel-args 'n1:t1,n2:t2,...']
  [--log-level DEBUG|INFO|WARNING|ERROR]
```

## API reference

The following describes the Accordo Python API classes and methods.

### `Accordo(binary, kernel_name, **options)`

**Parameters:**
- `binary` (str | list): binary path to extract the kernel signature from
- `kernel_name` (str): name of the kernel to validate
- `kernel_args` (list[tuple] | None): manual kernel arguments such as `[(name, type), ...]`. Auto-extracted if set to `None`.
- `working_directory` (str) — working directory (default: `"."`)
- `force_rebuild` (bool): force rebuild even if the library exists (default: `False`)
- `parallel_jobs` (int) — number of parallel build jobs (default: `16`)
- `log_level` (str) — logging level (default: `"WARNING"`)

**Methods:**
- `capture_snapshot(binary, timeout_seconds=30, dispatch_id=None)` -> `Snapshot`
- `compare_snapshots(reference, optimized, tolerance=None, *, atol=1e-08, rtol=1e-05, equal_nan=False)` -> `ValidationResult`

### `Snapshot`

| Attribute | Type | Description |
|-----------|------|-------------|
| `arrays` | `list[np.ndarray]` | Captured output arrays (first dispatch) |
| `dispatch_arrays` | `list[list[np.ndarray]] \| None` | Per-dispatch output arrays |
| `execution_time_ms` | `float` | Execution time |
| `grid_size` | `dict \| None` | Kernel grid dimensions |
| `block_size` | `dict \| None` | Kernel block dimensions |

### `ValidationResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `is_valid` | `bool` | Whether validation passed |
| `num_arrays_validated` | `int` | Total arrays checked |
| `num_mismatches` | `int` | Failed comparisons |
| `mismatches` | `list[ArrayMismatch]` | Detailed mismatch information |

**Methods:**
- `summary()` -> `str`: human-readable validation summary

