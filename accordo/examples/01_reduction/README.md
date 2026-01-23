# Example 01: Reduction Kernel Validation

Demonstrates using Accordo to validate an optimized reduction kernel.

## What it does

The Python script:
1. Writes baseline and optimized HIP kernels to `/tmp`
2. Compiles both versions with `hipcc`
3. Uses Accordo to capture kernel outputs
4. Validates they produce identical results

## Run it

```bash
python3 validate.py
```

## Requirements

- ROCm and `hipcc` installed
- Accordo installed: `pip install git+https://github.com/AMDResearch/intellikit.git#egg=intellikit[accordo]`

## What gets validated

- **Baseline kernel**: Simple atomic-based reduction (slow but correct)
- **Optimized kernel**: Shared memory reduction (faster, needs validation)
- Both sum 1M floats and must produce the same result within tolerance

## Expected output

```
Accordo Example: Reduction Kernel Validation
Compiling kernels in /tmp...
✓ Validator ready
✓ Capturing snapshots...
✓ VALIDATION PASSED!
  Speedup: 2.3x
```

See [accordo documentation](../../README.md) for more details.

