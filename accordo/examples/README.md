# Accordo Examples

This directory contains examples demonstrating how to use Accordo for GPU kernel validation.

## Examples

### [01_reduction_validation](01_reduction_validation/)

**Purpose**: Validate an optimized reduction kernel against a baseline implementation

**What it demonstrates**:
- Compiling baseline and optimized kernels
- Configuring Accordo with kernel signature
- Capturing snapshots from both versions
- Comparing outputs for correctness
- Displaying validation results

**Run it**:
```bash
cd 01_reduction_validation
python3 validate_reduction.py
```

## Example Structure

Each example follows this pattern:

```
01_example_name/
├── README.md              # Example documentation
├── kernel_baseline.hip    # Reference implementation
├── kernel_optimized.hip   # Optimized implementation
├── validate.py            # Accordo validation script
└── Makefile              # Build targets
```

## Creating Your Own Example

1. **Write baseline kernel**: Correct reference implementation
2. **Write optimized kernel**: Performance-improved version
3. **Configure Accordo**: Define kernel signature and arguments
4. **Capture snapshots**: Run both kernels via Accordo
5. **Validate**: Compare outputs within tolerance

See the reduction example for a template you can adapt.

## Common Patterns

### Basic Validation
```python
from accordo import Accordo

# Configure
config = Accordo.Config(
    kernel_name="my_kernel",
    kernel_args=[...],
    tolerance=1e-6
)

# Create validator
validator = Accordo(config)

# Capture and compare
ref_snap = validator.capture_snapshot(binary=["./baseline"])
opt_snap = validator.capture_snapshot(binary=["./optimized"])
result = validator.compare_snapshots(ref_snap, opt_snap)

# Check results
if result.is_valid:
    print("✓ Validation passed!")
else:
    print(f"✗ Failed: {result.summary()}")
```

### Batch Validation
```python
# Capture reference once
ref_snap = validator.capture_snapshot(binary=["./baseline"])

# Test multiple optimizations
for opt_binary in optimization_candidates:
    opt_snap = validator.capture_snapshot(binary=[opt_binary])
    result = validator.compare_snapshots(ref_snap, opt_snap)
    print(f"{opt_binary}: {'✓' if result.is_valid else '✗'}")
```

## Tips

1. **Start simple**: Use small inputs first, then scale up
2. **Set appropriate tolerance**: Floating-point needs wiggle room (1e-4 to 1e-6)
3. **Check execution time**: Accordo captures timing in snapshots
4. **Use grid/block metadata**: Available in snapshot for debugging
5. **Batch testing**: Capture reference once, compare many optimizations

## Troubleshooting

**Build failures**: Ensure `hipcc` is available and ROCm is installed

**Validation failures**: Could indicate:
- Actual bug in optimized kernel (good catch!)
- Tolerance too strict for floating-point
- Wrong kernel name in configuration

**Performance**: Accordo adds overhead for capture, but comparison is fast

## Learn More

- [Accordo Documentation](../README.md)
- [IntelliKit Documentation](../../README.md)

