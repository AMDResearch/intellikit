# Example 03: PyTorch — matrix multiply (Linex)

Same **PyTorch** `A @ B` idea as Metrix’s [03_pytorch_matmul](../../../metrix/examples/03_pytorch_matmul/), profiled with **Linex**.

## What it does

1. Runs `tensor_matmul.py`: `(N, N)` matrices on `cuda`, `A @ B`, synchronize, checksum.
2. `profile_with_linex.py` runs `Linex().profile(...)` on that command and prints hotspots or ISA.

## Run it

Workload only (from `linex/examples/03_pytorch_matmul/`):

```bash
python3 tensor_matmul.py
python3 tensor_matmul.py --size 1024
```

End-to-end with Linex:

```bash
python3 profile_with_linex.py
```

## Requirements

- **Linex** installed: `pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=linex"` or `pip install -e .` from the `linex/` package root
- **ROCm 7.0+** with `rocprofv3` and a supported AMD GPU
- **PyTorch** with a visible GPU

See the [Linex package README](../../README.md) and the [examples index](../README.md).
