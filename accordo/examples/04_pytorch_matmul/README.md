# Example 04: PyTorch — matrix multiply (Accordo)

Demonstrates **Accordo** on two small HIP binaries (N=32) that share the kernel name `matmul_nn`: a plain inner `k` loop vs a 2-unrolled `k` loop. Same layout as [Example 03: element-wise add](../03_pytorch_tensor_add/) and the narrative style of [Example 01: reduction](../01_reduction/).

**`tensor_matmul.py`** matches Metrix / Linex / Nexus for the PyTorch story. **`validate.py`** drives Accordo on the HIP pair.

## What it does

1. **`validate.py`**: temp HIP sources, `hipcc -g`, `Accordo` snapshot + compare for `matmul_nn`.
2. **`tensor_matmul.py`**: optional PyTorch `A @ B` + checksum.

## Run it

```bash
python3 validate.py
```

PyTorch only:

```bash
python3 tensor_matmul.py
python3 tensor_matmul.py --size 1024
```

## Requirements

- ROCm and `hipcc`
- Accordo: `pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=accordo"` or `pip install -e ./accordo`
- For `tensor_matmul.py`: GPU PyTorch

## Expected output

Steps 1–5, then `VALIDATION PASSED` and timing, analogous to Example 01.

## Note

GEMM validation uses slightly looser `atol`/`rtol` than Example 03 because of accumulation order in the unrolled loop.

See [accordo documentation](../../README.md).
