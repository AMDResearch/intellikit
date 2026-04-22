# Example 02: PyTorch — add two tensors

Demonstrates a minimal **PyTorch** GPU workload you can run alone or under **Metrix**, same pattern as [Example 01: basic profiling](../01_basic_profiling/).

## What it does

1. Allocates two random `(N, N)` tensors on `cuda` and adds them element-wise.
2. Calls `torch.cuda.synchronize()` so the work finishes before printing.
3. Prints a scalar checksum (`sum(a+b)`) so you can confirm the run.
4. Optionally: from this directory, wrap the same command with `metrix --time-only` to time GPU dispatches.

## Run it

Workload only (from `metrix/examples/02_pytorch_tensor_add/`):

```bash
python3 tensor_add.py
python3 tensor_add.py --size 2048
```

Profile with Metrix (path must resolve to `tensor_add.py`):

```bash
metrix --time-only -n 1 "python3 $(pwd)/tensor_add.py"
```

From the `metrix` package root, use an absolute or repo-relative path to `tensor_add.py` instead of `$(pwd)`.

## Requirements

- PyTorch with GPU support (ROCm wheel on AMD hardware)
- ROCm with `rocprofv3` for Metrix
- Metrix installed: `pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=metrix"` (or `pip install -e .` from the `metrix/` tree)

### Automated tests (optional)

From the Metrix package root: `pytest tests/integration/test_pytorch_tensor_add_example.py -m integration` (skips without GPU / torch / `metrix` on `PATH`).

See the [Metrix package README](../../README.md) and the [examples index](../README.md).
