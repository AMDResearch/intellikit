# Example 06: PyTorch — add two tensors (Nexus)

Demonstrates **Nexus** on a **PyTorch** workload: element-wise add on GPU, same narrative shape as [Example 01: Kernel Tracing](../01_trace_kernel/) (one driver script + clear steps).

## What it does

1. `tensor_add.py` allocates tensors on `cuda`, runs `a + b`, synchronizes, prints `sum(a+b)`.
2. `trace_pytorch.py` runs `Nexus().run([python, tensor_add.py])` and prints each kernel’s name, assembly line count, and a short assembly sample.

## Run it

Workload only (from `nexus/examples/06_pytorch_tensor_add/`):

```bash
python3 tensor_add.py
python3 tensor_add.py --size 2048
```

Trace with Nexus:

```bash
python3 trace_pytorch.py
```

## Requirements

- **Nexus** installed: `pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=nexus"` or `pip install -e .` from the `nexus/` tree (see repo README for `cmake`, `libdwarf`, etc.)
- **ROCm** on a supported AMD GPU
- **PyTorch** with a visible GPU (ROCm wheel on AMD hardware)

### Notes

- Kernel names and HIP/Triton line capture depend on PyTorch, Inductor, and ROCm version.

### Automated tests (optional)

From the Nexus package root: `pytest tests/test_pytorch_tensor_add_example.py` (skips without GPU / torch / built `libnexus.so`).

See the [Nexus package README](../../README.md) and the [examples index](../README.md).
