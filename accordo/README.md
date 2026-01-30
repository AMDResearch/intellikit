# Accordo: Automated GPU Kernel Validation

Accordo automatically validates GPU kernel correctness by capturing and comparing kernel outputs from reference and optimized implementations.

## Features

- **Automatic kernel extraction**: Uses kernelDB to extract kernel signatures from binaries
- **Snapshot-based validation**: Capture once, compare against multiple optimizations
- **Configurable tolerance**: Set precision requirements for floating-point comparisons
- **Performance tracking**: Measure and compare execution times

## Installation

```bash
# Install from IntelliKit
pip install git+https://github.com/AMDResearch/intellikit.git

# Or install accordo only
pip install git+https://github.com/AMDResearch/intellikit.git#subdirectory=accordo
```

## Quick Start

```python
from accordo import Accordo

# Create validator for a specific kernel
validator = Accordo(binary="./app_ref", kernel_name="reduce_sum")

# Capture snapshots from reference and optimized binaries
ref = validator.capture_snapshot(binary="./app_ref")
opt = validator.capture_snapshot(binary="./app_opt")

# Compare with specified tolerance
result = validator.compare_snapshots(ref, opt, tolerance=1e-6)

if result.is_valid:
    print(f"✓ PASS: {result.num_arrays_validated} arrays matched")
else:
    print(f"✗ FAIL: {result.summary()}")
```

### Testing Multiple Optimizations

```python
validator = Accordo(binary="./ref", kernel_name="matmul")
ref = validator.capture_snapshot(binary="./ref")

for opt_binary in ["./opt_v1", "./opt_v2", "./opt_v3"]:
    opt = validator.capture_snapshot(binary=opt_binary)
    result = validator.compare_snapshots(ref, opt, tolerance=1e-6)
    print(f"{opt_binary}: {'✓ PASS' if result.is_valid else '✗ FAIL'}")
```

## API Reference

### `Accordo(binary, kernel_name, **options)`

**Parameters:**
- `binary` (str | list): Binary path to extract kernel signature from
- `kernel_name` (str): Name of the kernel to validate
- `working_directory` (str): Working directory (default: `"."`)
- `log_level` (str): Logging level (default: `"WARNING"`)

**Methods:**
- `capture_snapshot(binary, timeout_seconds=30)` → `Snapshot`
- `compare_snapshots(reference, optimized, tolerance=1e-6)` → `ValidationResult`

### `Snapshot`

**Attributes:**
- `arrays` (list[np.ndarray]): Captured output arrays
- `execution_time_ms` (float): Execution time
- `grid_size`, `block_size` (dict | None): Kernel dimensions

### `ValidationResult`

**Attributes:**
- `is_valid` (bool): Whether validation passed
- `num_arrays_validated` (int): Total arrays checked
- `num_mismatches` (int): Failed comparisons
- `mismatches` (list[ArrayMismatch]): Detailed mismatch info

**Methods:**
- `summary()` → `str`: Human-readable validation summary

## Requirements

- Python >= 3.8
- ROCm toolchain
- kernelDB (automatically installed)

## Examples

See `examples/` directory for complete examples:
- `01_reduction/` - Basic reduction kernel validation
- `02_template_kernel/` - Template kernel validation

## License

MIT License - Copyright (c) 2025 Advanced Micro Devices, Inc.
