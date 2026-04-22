# Metrix examples

Small programs you can run locally or under `metrix profile` / `metrix --time-only`.

| Directory | What it is |
|-----------|------------|
| [**01_basic_profiling**](01_basic_profiling/) | Writes a HIP vector-add kernel, compiles with `hipcc`, profiles with Metrix (see that folder’s README). |
| [**02_pytorch_tensor_add**](02_pytorch_tensor_add/) | PyTorch GPU workload: element-wise add of two tensors (`tensor_add.py`). |
| [**03_pytorch_matmul**](03_pytorch_matmul/) | PyTorch GPU workload: matrix multiply `A @ B` (`tensor_matmul.py`). |

## Requirements (shared)

- **Metrix**: `pip install -e .` from the `metrix/` package root, or the install URL in each example README.
- **GPU profiling**: ROCm with `rocprofv3` on a supported AMD GPU.

PyTorch examples additionally need a **ROCm (or CUDA) PyTorch** build with a visible GPU.

See the [Metrix package README](../README.md) for CLI options and metric profiles.
