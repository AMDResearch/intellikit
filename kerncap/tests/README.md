# Tests

## Unit tests (no GPU required)

Tests the Python logic — CSV parsing, source finding, template rendering, output comparison, and metadata generation. Runs anywhere with Python 3.10+.

```bash
pip install pytest numpy jinja2 click
pytest tests/unit/ -v
```

The reproducer tests include pointer-chasing kernel fixtures that verify the template generates correct indirect buffer allocation, relocation patching, and cleanup for three patterns:

- **Batched vector scale** (`float**`) — classic batched BLAS with 8 indirect buffers and 8 relocations
- **Scatter-gather** (`const float* const*`) — pointer array where only the double-pointer arg gets relocations
- **Struct with pointer** (`TensorDesc*` containing `float* data`) — heuristic scan edge case with a single indirect buffer

## Integration tests

### Local (requires ROCm + AMD GPU)

The `test_vector_add.py` test compiles a trivial HIP kernel with `hipcc`, then runs the full `profile → capture → source find → reproducer → validate` pipeline. No Docker needed.

```bash
pytest tests/integration/test_vector_add.py -v
```

### Docker-based (requires Docker + AMD GPU)

These run the full pipeline inside official ROCm containers against real-world kernels. They pull the container, install kerncap from the local source tree, and exercise profiling.

```bash
# Flash Attention Triton backend (rocm/pytorch container)
pytest tests/integration/test_flash_attn.py -v

# Composable Kernel GEMM XDL (rocm/composable_kernel container)
pytest tests/integration/test_ck_gemm.py -v

# All Docker tests
pytest tests/integration/ -m docker -v

# Skip Docker tests (local only)
pytest tests/integration/ -m "not docker" -v
```

Docker tests are marked with `@pytest.mark.docker` and automatically skipped if Docker or `/dev/kfd` is unavailable.

To watch container output (pip installs, builds, etc.) in real time while a test runs, use pytest’s no-capture flag:

```bash
pytest tests/integration/test_flash_attn.py -v -s
```
