# Nexus examples

Examples live under `nexus/examples/`. Run scripts from the **`nexus/` package root** (or `cd` into each example folder and adjust paths).

| Directory | Script | What it does |
|-----------|--------|----------------|
| [01_trace_kernel](01_trace_kernel/) | `trace.py` | Writes a HIP vector-add kernel, compiles with `hipcc -g`, traces the binary with Nexus. |
| [02_simple_hip](02_simple_hip/) | `simple_hip.py` | Single HIP kernel trace (basic). |
| [03_multiple_hip](03_multiple_hip/) | `multiple_hip.py` | Two HIP kernels in one run. |
| [04_simple_triton](04_simple_triton/) | `simple_triton.py` | Single Triton kernel. |
| [05_multiple_triton](05_multiple_triton/) | `multiple_triton.py` | Two Triton kernels in one run. |
| [06_pytorch_tensor_add](06_pytorch_tensor_add/) | `tensor_add.py`, `trace_pytorch.py` | PyTorch GPU element-wise add; `trace_pytorch.py` runs `Nexus().run([python, tensor_add.py])` and prints kernels + assembly sample. |
| [07_pytorch_matmul](07_pytorch_matmul/) | `tensor_matmul.py`, `trace_pytorch.py` | PyTorch GPU `A @ B`; same trace helper pattern as tensor add. |

## Usage pattern

```python
from nexus import Nexus

nexus = Nexus(log_level=1)
trace = nexus.run(["python3", "path/to/script.py"])

for kernel in trace:
    print(kernel.name, len(kernel.assembly))
```

## Prerequisites (shared)

- ROCm and Nexus installed (`pip install -e ./nexus` from the IntelliKit clone, or the Git URL in each example README)
- HIP examples: `hipcc` in `PATH`
- Triton examples: `triton` installed
- **PyTorch examples** (06–07): PyTorch with GPU (ROCm on AMD)

See the [Nexus README](../README.md) for installation, API, and limitations.
