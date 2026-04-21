# Example 02: PyTorch — add two tensors

Minimal **PyTorch** GPU workload aligned with [metrix](../../../metrix/examples/02_pytorch_tensor_add/), [linex](../../../linex/examples/02_pytorch_tensor_add/), and [nexus](../../../nexus/examples/06_pytorch_tensor_add/). Use **kerncap** to rank GPU kernels for the process, then optionally extract one by name substring. For a **HIP** full pipeline (profile → extract → replay), see [Example 01](../01_extract_and_replay/README.md).

## What it does

1. Allocates two random `(N, N)` tensors on `cuda` and adds them element-wise.
2. Calls `torch.cuda.synchronize()` before printing a checksum (`sum(a+b)`).

## Run it

From `kerncap/examples/02_pytorch_tensor_add/` (or pass absolute paths from the `kerncap/` package root):

```bash
python3 tensor_add.py
python3 tensor_add.py --size 2048
```

Profile with kerncap (lists kernels; no extraction):

```bash
python3 profile_with_kerncap.py
```

After inspecting the profile, extract/replay/validate a kernel whose name contains `SUBSTRING`:

```bash
python3 profile_with_kerncap.py --kernel SUBSTRING
python3 profile_with_kerncap.py --kernel SUBSTRING --output ./my_reproducer --iterations 20
```

PyTorch issues many kernels; names depend on ROCm/PyTorch build. **Pick a substring from the printed list.** Source recovery into this directory may be limited (`has_source` may be false); replay/validate still succeed for many HIP captures.

## Requirements

- PyTorch with GPU (ROCm wheel on AMD hardware)
- ROCm with `hipcc`, `rocprofv3` on `PATH`
- kerncap: `pip install -e ./kerncap` from the IntelliKit clone

### Automated tests (optional)

From the `kerncap/` package root:

```bash
pytest tests/integration/test_pytorch_tensor_add_example.py -v
```

Skips without GPU, `rocprofv3`, or PyTorch CUDA device. The test only runs **profile** (not extract), for stability across environments.

See the [examples index](../README.md) and the [Kerncap README](../../README.md).
