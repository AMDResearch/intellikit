# Example 02: PyTorch — add two tensors (Linex)

Demonstrates the same **PyTorch** element-wise add workload as in Metrix’s [02_pytorch_tensor_add](../../../metrix/examples/02_pytorch_tensor_add/), but run through **Linex** (SQTT → source lines or ISA).

## What it does

1. Runs `tensor_add.py`: two `(N, N)` tensors on `cuda`, `a + b`, synchronize, checksum print.
2. `profile_with_linex.py` drives `Linex().profile(...)` on that command and prints hotspots (or a short ISA sample).

## Run it

Workload only (from `linex/examples/02_pytorch_tensor_add/`):

```bash
python3 tensor_add.py
python3 tensor_add.py --size 2048
```

End-to-end with Linex:

```bash
python3 profile_with_linex.py
```

Or call `Linex().profile` yourself (see script for the exact `command=` string).

## Requirements

- **Linex** installed: `pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=linex"` or `pip install -e .` from the `linex/` package root
- **ROCm 7.0+** with `rocprofv3` and a supported AMD GPU
- **PyTorch** with a visible GPU (ROCm wheel on AMD hardware)

### Notes

- PyTorch kernel names vary; pass `kernel_filter` as a regex to `profile()` if you want to narrow dispatches.

### Automated tests (optional)

From the Linex package root: `pytest tests/test_pytorch_tensor_add_example.py` (skips without GPU / torch / `rocprofv3`).

See the [Linex package README](../../README.md) and the [examples index](../README.md).
