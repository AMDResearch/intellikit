# Accordo: Automated GPU Kernel Validation

Accordo is an automated side-by-side correctness validation tool for GPU kernels. It captures kernel arguments and outputs from reference and optimized implementations, then compares them to ensure correctness.

## Features

- **Snapshot-based validation**: Capture kernel I/O once, compare against multiple optimizations
- **Configurable tolerance**: Set precision requirements for floating-point comparisons
- **Performance tracking**: Measure and compare execution times
- **Detailed error reporting**: Get precise information about validation failures

## Installation

### From Git Repository (Subdirectory)

```bash
pip install git+https://github.com/AMDResearch/nexus.git#subdirectory=accordo
```

### From IntelliKit (All Tools)

```bash
pip install git+https://github.com/AMDResearch/nexus.git
```

### Development Installation

```bash
git clone https://github.com/AMDResearch/nexus.git
cd nexus/accordo
pip install -e .
```

## Quick Start

### Simple Validation

```python
from accordo import Accordo

# Create validator for a specific kernel
# This extracts signature and builds the library once
validator = Accordo(
    binary="./app_ref",
    kernel_name="reduce_sum"
)

# Capture reference snapshot
ref = validator.capture_snapshot(binary="./app_ref")

# Capture optimized snapshot
opt = validator.capture_snapshot(binary="./app_opt")

# Compare with specified tolerance
result = validator.compare_snapshots(ref, opt, tolerance=1e-6)
print(f"Validation: {'✓ PASS' if result.is_valid else '✗ FAIL'}")
```


## Requirements

- Python >= 3.8
- ROCm toolchain
- HIP compiler (hipcc)
- kernelDB (automatically installed as dependency)

## License

MIT License - Copyright (c) 2025 Advanced Micro Devices, Inc.
