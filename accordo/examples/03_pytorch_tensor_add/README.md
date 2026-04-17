# Example 03: PyTorch — element-wise add (Accordo)

Demonstrates **Accordo** on a pair of HIP binaries that share the kernel name `elemwise_add` (baseline vs grid-stride optimized), in the same narrative style as [Example 01: reduction](../01_reduction/).

The optional **`tensor_add.py`** script matches other IntelliKit packages (Metrix / Linex / Nexus): same PyTorch workload for cross-reading. **Accordo** itself compares the HIP programs built by **`validate.py`**, not the PyTorch runtime.

## What it does

1. **`validate.py`**: writes two HIP sources to a temp directory, compiles with `hipcc -g`, runs `Accordo.capture_snapshot` on each binary, then `compare_snapshots`.
2. **`tensor_add.py`**: optional PyTorch `(N,N)` tensor add + checksum (requires GPU PyTorch).

## Run it

HIP validation (from `accordo/examples/03_pytorch_tensor_add/`):

```bash
python3 validate.py
```

PyTorch workload only (GPU + PyTorch required):

```bash
python3 tensor_add.py
python3 tensor_add.py --size 2048
```

## Requirements

- ROCm and `hipcc` installed
- Accordo installed: `pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=accordo"` (or `pip install -e ./accordo`)
- For `tensor_add.py`: PyTorch with a visible GPU (ROCm wheel on AMD hardware)

## Expected output

After `validate.py`, you should see steps 1–5, then `VALIDATION PASSED` and timing lines, similar to Example 01.

## Note

Kernel argument capture depends on KernelDB / debug symbols (`-g`). If validation fails on your stack, check Accordo logs and ROCm version compatibility.

See [accordo documentation](../../README.md) for CLI and API details.
