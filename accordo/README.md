# Accordo: Automated GPU Kernel Validation

Accordo automatically validates GPU kernel correctness by capturing and comparing kernel outputs from reference and optimized implementations.

## Features

- **Automatic kernel extraction**: Uses kernelDB to extract kernel signatures from binaries
- **Snapshot-based validation**: Capture once, compare against multiple optimizations
- **Configurable tolerance**: Set precision requirements for floating-point comparisons (`atol`, `rtol`, `equal_nan`)
- **Performance tracking**: Measure and compare execution times

## Installation

### System prerequisites

Accordo compiles C++ code (via [KernelDB](https://github.com/AMDResearch/KernelDB)) during installation.
You need `cmake`, `libdwarf-dev`, and `libzstd-dev` installed first:

```bash
# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y cmake libdwarf-dev libzstd-dev

# Fedora / RHEL
sudo dnf install -y cmake libdwarf-devel libzstd-devel
```

### Install via pip

```bash
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=accordo"
```

## Quick Start

```python
from accordo import Accordo

# Create validator for a specific kernel
validator = Accordo(binary="./app_ref", kernel_name="reduce_sum")

# Capture snapshots from reference and optimized binaries
ref = validator.capture_snapshot(binary="./app_ref")
opt = validator.capture_snapshot(binary="./app_opt")

# Compare with allclose-style controls
result = validator.compare_snapshots(ref, opt, atol=1e-6, rtol=1e-5, equal_nan=False)

if result.is_valid:
    print(f"PASS: {result.num_arrays_validated} arrays matched")
else:
    print(result.summary())
```

### Testing Multiple Optimizations

```python
validator = Accordo(binary="./ref", kernel_name="matmul")
ref = validator.capture_snapshot(binary="./ref")

for opt_binary in ["./opt_v1", "./opt_v2", "./opt_v3"]:
    opt = validator.capture_snapshot(binary=opt_binary)
    result = validator.compare_snapshots(ref, opt, atol=1e-6, rtol=1e-5)
    print(f"{opt_binary}: {'PASS' if result.is_valid else 'FAIL'}")
```

## Command line

The `accordo` entry point exposes JSON on stdout (logging on stderr). Subcommands:

```
accordo validate \
  --kernel-name NAME \
  --ref-binary PATH_TO_EXECUTABLE \
  --opt-binary PATH_TO_EXECUTABLE \
  [--tolerance FLOAT]               # legacy alias for --atol
  [--atol FLOAT]                    # absolute tolerance (default: 1e-08)
  [--rtol FLOAT]                    # relative tolerance (default: 1e-05)
  [--equal-nan]                     # treat NaN == NaN
  [--timeout SECONDS]               # per snapshot, default: 30
  [--working-dir DIR]               # default: .
  [--kernel-args 'n1:t1,n2:t2,...']
  [--log-level DEBUG|INFO|WARNING|ERROR]  # default: WARNING
```

Example: `accordo validate --kernel-name reduce_sum --ref-binary ./app_ref --opt-binary ./app_opt`

The CLI passes each flag as a **single executable path** (no embedded spaces or extra argv). For runs that need arguments, use a wrapper script or the Python API (`capture_snapshot` accepts `binary` as a list, e.g. `["./app", "--flag"]`).

## API Reference

### `Accordo(binary, kernel_name, **options)`

**Parameters:**
- `binary` (str | list): Binary path to extract kernel signature from
- `kernel_name` (str): Name of the kernel to validate
- `kernel_args` (list[tuple] | None): Manual kernel args as `[(name, type), ...]`. Auto-extracted if None.
- `working_directory` (str): Working directory (default: `"."`)
- `force_rebuild` (bool): Force rebuild even if library exists (default: `False`)
- `parallel_jobs` (int): Number of parallel build jobs (default: `16`)
- `log_level` (str): Logging level (default: `"WARNING"`)

**Methods:**
- `capture_snapshot(binary, timeout_seconds=30, dispatch_id=None)` -> `Snapshot`
- `compare_snapshots(reference, optimized, tolerance=None, *, atol=1e-08, rtol=1e-05, equal_nan=False)` -> `ValidationResult`

### `Snapshot`

**Attributes:**
- `arrays` (list[np.ndarray]): Captured output arrays (first dispatch)
- `dispatch_arrays` (list[list[np.ndarray]] | None): Per-dispatch output arrays
- `execution_time_ms` (float): Execution time
- `grid_size`, `block_size` (dict | None): Kernel dimensions

### `ValidationResult`

**Attributes:**
- `is_valid` (bool): Whether validation passed
- `num_arrays_validated` (int): Total arrays checked
- `num_mismatches` (int): Failed comparisons
- `mismatches` (list[ArrayMismatch]): Detailed mismatch info

**Methods:**
- `summary()` -> `str`: Human-readable validation summary

## Requirements

- Python >= 3.8
- ROCm toolchain
- kernelDB (automatically installed)

## Examples

See `examples/` directory for complete examples:
- `01_reduction/` - Basic reduction kernel validation
- `02_template_kernel/` - Template kernel validation
- `03_pytorch_tensor_add/` - Element-wise add: HIP ref/opt for Accordo + optional PyTorch `tensor_add.py`
- `04_pytorch_matmul/` - Small square matmul: HIP ref/opt + optional PyTorch `tensor_matmul.py`

## License

MIT License - Copyright (c) 2025 Advanced Micro Devices, Inc.
