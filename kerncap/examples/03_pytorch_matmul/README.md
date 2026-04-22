# Example 03: PyTorch — matrix multiply

Minimal **PyTorch** GPU **GEMM-style** workload (`C = A @ B`), aligned with [metrix](../../../metrix/examples/03_pytorch_matmul/), [linex](../../../linex/examples/03_pytorch_matmul/), and [nexus](../../../nexus/examples/07_pytorch_matmul/). For a **HIP** full pipeline, see [Example 01](../01_extract_and_replay/README.md).

## What it does

1. Allocates random square matrices on `cuda`, computes `A @ B`, synchronizes, prints `sum(A@B)`.

## Run it

From `kerncap/examples/03_pytorch_matmul/`:

```bash
python3 tensor_matmul.py
python3 tensor_matmul.py --size 1024
```

Profile with kerncap:

```bash
python3 profile_with_kerncap.py
```

Optional full pipeline after choosing a kernel substring from the profile:

```bash
python3 profile_with_kerncap.py --kernel SUBSTRING --output ./my_reproducer
```

See [Example 02 README](../02_pytorch_tensor_add/README.md) for caveats on PyTorch kernel names and source finding.

## Requirements

Same as Example 02: ROCm, `rocprofv3`, ROCm PyTorch, `pip install -e ./kerncap`.

See the [examples index](../README.md).
