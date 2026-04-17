# Example 07: PyTorch — matrix multiply (Nexus)

Demonstrates **Nexus** on a **PyTorch** GEMM-style workload (`A @ B`), same layout as [Example 06: PyTorch tensor add](../06_pytorch_tensor_add/).

## What it does

1. `tensor_matmul.py` runs `A @ B` on `cuda`, synchronizes, prints `sum(A@B)`.
2. `trace_pytorch.py` runs `Nexus().run([python, tensor_matmul.py])` and prints kernels plus a short assembly sample.

## Run it

Workload only (from `nexus/examples/07_pytorch_matmul/`):

```bash
python3 tensor_matmul.py
python3 tensor_matmul.py --size 1024
```

Trace with Nexus:

```bash
python3 trace_pytorch.py
```

## Requirements

- **Nexus** installed: `pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=nexus"` or `pip install -e .` from the `nexus/` tree
- **ROCm** on a supported AMD GPU
- **PyTorch** with a visible GPU

### Notes

- Same caveats as Example 06 regarding kernel naming and optional HIP/Triton source.

See the [Nexus package README](../../README.md) and the [examples index](../README.md).
