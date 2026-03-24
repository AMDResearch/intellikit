# Kerncap Examples

Examples demonstrating how to use kerncap to extract, replay, and validate GPU kernels.

## Examples

### `extract_and_replay.py`

Full kerncap pipeline on a multi-kernel HIP application.

**Run:**
```bash
python examples/extract_and_replay.py
```

**What it does:**
- Compiles `mini_pipeline.hip` (five GPU kernels in a single file)
- Profiles the application to rank kernels by GPU time
- Extracts the target kernel into a standalone reproducer
- Replays the captured kernel in isolation and reports timing
- Validates the reproducer for correctness

**Options:**
```bash
# Extract a different kernel (default: vector_scale)
python examples/extract_and_replay.py --kernel histogram_atomic

# Benchmark with more iterations
python examples/extract_and_replay.py --iterations 50

# Save the reproducer to a specific directory
python examples/extract_and_replay.py --output ./my_reproducer
```

### `mini_pipeline.hip`

A standalone HIP application with five kernels exercising common GPU patterns:

| Kernel | Pattern |
|--------|---------|
| `vector_add` | Elementwise addition |
| `vector_scale` | Scalar multiplication |
| `vector_bias_relu` | Fused bias + ReLU activation |
| `vector_shift` | Elementwise shift |
| `histogram_atomic` | Atomic histogram (different grid size) |

**Compile and run directly:**
```bash
hipcc -O2 -o mini_pipeline examples/mini_pipeline.hip
./mini_pipeline
```

## Prerequisites

- ROCm installed (`hipcc`, `rocprofv3` on PATH)
- AMD GPU (MI300+ recommended)
- kerncap installed: `pip install -e kerncap/`
