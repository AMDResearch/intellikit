"""Integration test: --triton-backend hsa vs --triton-backend python.

Runs a tiny Triton workload under both backends and asserts that they
agree on the things the on-disk schema guarantees.

Specifically, the HSA backend produces a HIP-style VA-faithful capture
(``dispatch.json`` + ``kernarg.bin`` + ``memory_regions.json``) that
``kerncap-replay`` can consume directly, *plus* a ``metadata.json``
that includes the parsed ``kernarg_slots`` table and the
``triton_user_name`` resolved via SHA-256 lookup against
``name_map.json``.

Marked ``@pytest.mark.gpu`` because the workload actually launches a
Triton kernel.
"""

from __future__ import annotations

import json
import os
import shutil
import textwrap
from pathlib import Path

import pytest

from .conftest import skip_no_gpu


WORKLOAD = textwrap.dedent("""\
    import torch
    import triton
    import triton.language as tl


    @triton.jit
    def triton_poi_fused_relu_0(in_out_ptr, xnumel, XBLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * XBLOCK + tl.arange(0, XBLOCK)
        mask = offs < xnumel
        x = tl.load(in_out_ptr + offs, mask=mask)
        tl.store(in_out_ptr + offs, tl.maximum(x, 0.0), mask=mask)


    if __name__ == "__main__":
        n = 1 << 20
        XBLOCK = 1024
        z = torch.randn(n, device="cuda", dtype=torch.float32) - 0.5
        triton_poi_fused_relu_0[(triton.cdiv(n, XBLOCK),)](z, n, XBLOCK=XBLOCK)
        torch.cuda.synchronize()
""")


@pytest.fixture
def workload_path(tmp_path):
    p = tmp_path / "_relu_workload.py"
    p.write_text(WORKLOAD)
    return str(p)


def _check_imports() -> None:
    try:
        import torch  # noqa: F401
        import triton  # noqa: F401
    except Exception as e:
        pytest.skip(f"torch/triton not available: {e}")


@pytest.mark.gpu
@skip_no_gpu
def test_hsa_backend_produces_va_faithful_capture(workload_path, tmp_path):
    _check_imports()
    from kerncap.triton_capture_hsa import run_triton_capture_hsa

    out = tmp_path / "hsa_capture"
    run_triton_capture_hsa(
        kernel_name="triton_poi_fused_relu_0",
        cmd=["python", workload_path],
        output_dir=str(out),
        dispatch=-1,
        timeout=120,
    )

    # The HSA backend MUST produce the VA-faithful artifact set the
    # HIP backend produces, plus the Triton-specific name_map.json.
    for fname in (
        "dispatch.json",
        "kernarg.bin",
        "kernarg_raw.bin",
        "memory_regions.json",
        "metadata.json",
        "name_map.json",
    ):
        assert (out / fname).is_file(), f"missing {fname} in HSA capture"
    assert (out / "memory").is_dir()
    assert (out / "triton_hsacos").is_dir()


@pytest.mark.gpu
@skip_no_gpu
def test_hsa_backend_kernarg_slots_match_spike(workload_path, tmp_path):
    """The C++ parser must recover the same slot table the spike recorded
    for triton_poi_fused_relu_0 (1 user pointer, 1 by_value scalar, 2
    unused readnone pointer slots; total 32 bytes; no constexprs)."""
    _check_imports()
    from kerncap.triton_capture_hsa import run_triton_capture_hsa

    out = tmp_path / "hsa_capture"
    run_triton_capture_hsa(
        kernel_name="triton_poi_fused_relu_0",
        cmd=["python", workload_path],
        output_dir=str(out),
        dispatch=-1,
        timeout=120,
    )

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["triton_user_name"] == "triton_poi_fused_relu_0"
    assert meta["kernarg_segment_size_meta"] == 32
    assert "constexpr_values" in meta

    slots = meta["kernarg_slots"]
    assert len(slots) == 4
    by_value = [s for s in slots if s["value_kind"] == "by_value"]
    assert len(by_value) == 1, (
        "exactly one by_value slot expected (xnumel); constexpr XBLOCK "
        "must NOT appear in the kernarg layout"
    )
    assert by_value[0]["offset"] == 8 and by_value[0]["size"] == 4

    ptrs = [s for s in slots if s["value_kind"] == "global_buffer"]
    assert len(ptrs) == 3
    assert {s["offset"] for s in ptrs} == {0, 16, 24}


@pytest.mark.gpu
@skip_no_gpu
def test_hsa_and_python_backends_agree_on_constexpr_value(workload_path, tmp_path):
    """Both backends must surface XBLOCK=1024 to downstream metadata,
    even though the HSA backend recovers it via name_map.json (constexpr
    values logged by the compile-hook from Triton's per-kernel JSON
    sidecar) rather than via JITFunction.run kwargs introspection."""
    _check_imports()
    from kerncap.triton_capture import run_triton_capture
    from kerncap.triton_capture_hsa import run_triton_capture_hsa

    py_out = tmp_path / "python_capture"
    hsa_out = tmp_path / "hsa_capture"

    run_triton_capture(
        kernel_name="triton_poi_fused_relu_0",
        cmd=["python", workload_path],
        output_dir=str(py_out),
        dispatch=-1,
        timeout=120,
    )
    run_triton_capture_hsa(
        kernel_name="triton_poi_fused_relu_0",
        cmd=["python", workload_path],
        output_dir=str(hsa_out),
        dispatch=-1,
        timeout=120,
    )

    py_meta = json.loads((py_out / "metadata.json").read_text())
    py_xblock = next(a for a in py_meta["args"] if a.get("name") == "XBLOCK")
    assert py_xblock["value"] == 1024

    hsa_meta = json.loads((hsa_out / "metadata.json").read_text())
    cv = hsa_meta.get("constexpr_values", {})
    # Triton 3.x records constexprs under a "(<arg_index>,)"-string key.
    found = any(int(v) == 1024 for v in cv.values() if isinstance(v, (int, float)))
    assert found, f"XBLOCK=1024 not found in HSA constexpr_values: {cv}"
