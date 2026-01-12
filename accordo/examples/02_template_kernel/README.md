# Example 02: Template Kernel Validation

This example demonstrates how to validate C++ template kernels with Accordo. It shows:

1. **Template discovery**: Listing all template instantiations in a binary
2. **Multiple validators**: Creating separate validators for each template instantiation
3. **Type-specific validation**: Validating float, double, and other type instantiations

## The Template Kernel

```cpp
template<typename T>
__global__ void scale_values(T* input, T* output, T factor, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * factor;
    }
}
```

This kernel is instantiated with multiple types:
- `scale_values<float>`
- `scale_values<double>`

## Key Concepts

### Each Template Instantiation is a Different Kernel

Different template instantiations have **different signatures**, so each needs its own validator:

```python
# Validator for float instantiation
float_validator = Accordo(
    binary="./app",
    kernel_name="scale_values<float>"  # Note: matches demangled name
)

# Validator for double instantiation
double_validator = Accordo(
    binary="./app",
    kernel_name="scale_values<double>"
)
```

### Discovering Template Instantiations

Use `list_available_kernels()` to discover all instantiations:

```python
from accordo import list_available_kernels

kernels = list_available_kernels("./app")
# Returns: ["scale_values<float>", "scale_values<double>", ...]

# Filter for specific template
scale_kernels = [k for k in kernels if "scale_values" in k]
```

### Validating All Instantiations

```python
for kernel_name in scale_kernels:
    # Create validator for this instantiation
    validator = Accordo(
        binary="./ref",
        kernel_name=kernel_name
    )

    # Capture and compare
    ref = validator.capture_snapshot(binary="./ref")
    opt = validator.capture_snapshot(binary="./opt")
    result = validator.compare_snapshots(ref, opt)

    print(f"{kernel_name}: {'PASS' if result.is_valid else 'FAIL'}")
```

## Running the Example

```bash
python3 validate.py
```

## Expected Output

```
================================================================================
Accordo Example: Template Kernel Validation
================================================================================

Temp directory: /tmp/accordo_template_xxx

Step 1: Compiling kernels...
Compiling reference... OK
Compiling optimized... OK

Step 2: Discovering template instantiations...
Found 2 template instantiation(s):
  - void scale_values<double>(double*, double*, double, int)
  - void scale_values<float>(float*, float*, float, int)

Step 3: Validating template instantiations...

Validating double instantiation...
  Kernel: void scale_values<double>(double*, double*, double, int)
  Arguments: ['input:double*', 'output:double*', 'factor:double', 'N:int']
  Captured: 1 array(s)
  ✓ PASS - Arrays matched within tolerance

Validating float instantiation...
  Kernel: void scale_values<float>(float*, float*, float, int)
  Arguments: ['input:float*', 'output:float*', 'factor:float', 'N:int']
  Captured: 1 array(s)
  ✓ PASS - Arrays matched within tolerance

================================================================================
SUMMARY
================================================================================
Template instantiations validated: 2
Passed: 2
Failed: 0

  double     ✓ PASS
  float      ✓ PASS
================================================================================
```
