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

```python
from accordo import Accordo

# Configure validation
config = Accordo.Config(
    kernel_name="my_kernel",
    kernel_args=[
        Accordo.KernelArg(name="result", type="double*"),
        Accordo.KernelArg(name="input", type="const double*"),
    ],
    tolerance=1e-6
)

# Create validator
validator = Accordo(config)

# Capture reference snapshot
ref_snapshot = validator.capture_snapshot(
    binary=["./app_ref"],
    working_directory=".",
    timeout_seconds=30
)

# Compare against optimized version
opt_snapshot = validator.capture_snapshot(
    binary=["./app_opt"],
    working_directory=".",
    timeout_seconds=30
)

result = validator.compare_snapshots(ref_snapshot, opt_snapshot)
print(f"Validation: {'✓ PASS' if result.is_valid else '✗ FAIL'}")
if result.is_valid:
    print(f"✓ {result.num_arrays_validated} arrays matched within tolerance")
else:
    print(f"✗ {result.num_mismatches} mismatches found")
    print(result.summary())
```

## Requirements

- Python >= 3.8
- ROCm toolchain
- HIP compiler (hipcc)

## License

MIT License - Copyright (c) 2025 Advanced Micro Devices, Inc.

