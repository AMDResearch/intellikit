# Example 03: PyTorch — matrix multiply

Demonstrates a small **PyTorch** GEMM-style workload (`A @ B`) you can run alone or under **Metrix**, following the same layout as [Example 02: PyTorch tensor add](../02_pytorch_tensor_add/).

## What it does

1. Allocates two random `(N, N)` tensors on `cuda` and multiplies them with `A @ B`.
2. Calls `torch.cuda.synchronize()` so the work finishes before printing.
3. Prints a scalar checksum (`sum(A@B)`).
4. Optionally: wrap the same command with `metrix --time-only` to time GPU dispatches.

## Run it

Workload only (from `metrix/examples/03_pytorch_matmul/`):

```bash
python3 tensor_matmul.py
python3 tensor_matmul.py --size 1024
```

Profile with Metrix:

```bash
metrix --time-only -n 1 "python3 $(pwd)/tensor_matmul.py"
```

From the `metrix` package root, use an absolute or repo-relative path to `tensor_matmul.py` instead of `$(pwd)`.

## Requirements

- PyTorch with GPU support (ROCm wheel on AMD hardware)
- ROCm with `rocprofv3` for Metrix
- Metrix installed: `pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=metrix"` (or `pip install -e .` from the `metrix/` tree)

See the [Metrix package README](../../README.md) and the [examples index](../README.md).
