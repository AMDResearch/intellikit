# Example 01: HIP mini-pipeline — extract and replay

Multi-kernel **HIP** application (`mini_pipeline.hip`) and a **full kerncap pipeline** script: profile → extract → replay → validate.

This is the numbered “first example” for kerncap, in the same style as Metrix’s [01_basic_profiling](../../../metrix/examples/01_basic_profiling/) and Linex’s [01_simple_sqtt](../../../linex/examples/01_simple_sqtt/).

## What it does

[`extract_and_replay.py`](extract_and_replay.py) uses the Python API to:

1. Compile `mini_pipeline.hip` (five GPU kernels in one file)
2. Profile the binary to rank kernels by GPU time
3. Extract the target kernel into a standalone reproducer
4. Replay the captured kernel in isolation and report timing
5. Validate the reproducer

## Run it

From the `kerncap/` package root:

```bash
python3 examples/01_extract_and_replay/extract_and_replay.py
```

Options:

```bash
python3 examples/01_extract_and_replay/extract_and_replay.py --kernel histogram_atomic
python3 examples/01_extract_and_replay/extract_and_replay.py --iterations 50
python3 examples/01_extract_and_replay/extract_and_replay.py --output ./my_reproducer
```

## `mini_pipeline.hip` kernels

| Kernel | Pattern |
|--------|---------|
| `vector_add` | Elementwise addition |
| `vector_scale` | Scalar multiplication (default extract target) |
| `vector_bias_relu` | Fused bias + ReLU |
| `vector_shift` | Elementwise shift |
| `histogram_atomic` | Atomic histogram |

Compile and run the HIP app alone:

```bash
hipcc -O2 -o mini_pipeline examples/01_extract_and_replay/mini_pipeline.hip
./mini_pipeline
```

## Requirements

- ROCm (`hipcc`, `rocprofv3` on `PATH`)
- AMD GPU (MI300+ recommended)
- kerncap: `pip install -e ./kerncap` from the IntelliKit clone

See the [examples index](../README.md) and the [Kerncap README](../../README.md).
