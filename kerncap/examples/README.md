# Kerncap examples

Small workloads and drivers you can run locally. Layout matches other IntelliKit tools: **numbered directories** under `kerncap/examples/` (see [Metrix](../../metrix/examples/README.md), [Linex](../../linex/examples/README.md), [Nexus](../../nexus/examples/README.md)).

Run scripts from the **`kerncap/` package root** (so imports resolve), or `cd` into an example folder and run the paths shown in each README.

| Directory | What it is |
|-----------|------------|
| [**01_extract_and_replay**](01_extract_and_replay/) | HIP app with five kernels (`mini_pipeline.hip`); `extract_and_replay.py` runs profile → extract → replay → validate. |
| [**02_pytorch_tensor_add**](02_pytorch_tensor_add/) | PyTorch element-wise add (`tensor_add.py`); `profile_with_kerncap.py` profiles via `Kerncap` and optionally extracts by `--kernel` substring. |
| [**03_pytorch_matmul**](03_pytorch_matmul/) | PyTorch matrix multiply (`tensor_matmul.py`); same driver pattern as 02. |

## Requirements (shared)

- **kerncap**: `pip install -e .` from the `kerncap/` package root (see [Kerncap README](../README.md))
- **HIP / full pipeline (01)**: ROCm with `hipcc`, `rocprofv3`, visible AMD GPU
- **PyTorch examples (02–03)**: ROCm (or CUDA) PyTorch with a visible GPU, plus ROCm tooling for profiling

## Slurm (cluster GPU node)

From the IntelliKit repo root, submit [`kerncap/slurm/test_pytorch_examples_mi300_1x.sbatch`](../slurm/test_pytorch_examples_mi300_1x.sbatch) (tune `#SBATCH` for your site). The job runs the PyTorch example scripts, both `profile_with_kerncap.py` drivers (profile-only), and `pytest tests/integration/test_pytorch_tensor_add_example.py`.

Optional: set `KERNCAP_SLURM_PYTORCH_KERNEL` to a kernel name substring (from a prior profile) to also run extract → replay → validate on the tensor-add example.
