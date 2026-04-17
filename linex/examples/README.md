# Linex examples

| Directory | What it is |
|-----------|------------|
| [**01_simple_sqtt**](01_simple_sqtt/) | Compile a small HIP kernel with `-g`, profile with Linex, print source-line / ISA style output (`example.py`). |
| [**02_pytorch_tensor_add**](02_pytorch_tensor_add/) | PyTorch GPU **element-wise add** (`tensor_add.py`) plus `profile_with_linex.py` to run Linex on that command. |
| [**03_pytorch_matmul**](03_pytorch_matmul/) | PyTorch GPU **matrix multiply** (`tensor_matmul.py`) plus `profile_with_linex.py`. |

## Requirements (shared)

- **Linex** installed from the `linex/` package root
- **ROCm 7.0+** with `rocprofv3` on a supported GPU

The PyTorch example also needs **PyTorch with GPU** (ROCm build on AMD hardware).

See the [Linex README](../README.md) for API details, `-g` vs assembly-only behavior, and MCP usage.
