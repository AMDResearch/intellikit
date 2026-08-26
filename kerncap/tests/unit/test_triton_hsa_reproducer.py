"""Unit tests for the HSA-backed Triton reproducer generator.

Exercises ``generate_triton_hsa_reproducer`` end-to-end on a synthetic
capture directory without requiring a GPU, ROCm, or a real Triton
runtime.  Verifies that the generated ``reproducer.py`` is valid
Python, references the expected pointer/scalar/constexpr args, and
loads buffer data from the captured memory regions.
"""

from __future__ import annotations

import ast
import json
import os
import struct
from pathlib import Path

import pytest

from kerncap.reproducer import (
    _decode_scalar,
    _find_region_for_pointer,
    _select_name_map_row,
    generate_triton_hsa_reproducer,
)
from kerncap.source_finder import KernelSource


@pytest.fixture
def synthetic_capture(tmp_path: Path) -> Path:
    """Build a minimal HSA-style capture directory.

    Layout:
      capture/
        metadata.json        -- with kernarg_slots (1 ptr + 1 scalar)
        kernarg_raw.bin      -- 16 bytes (8 ptr + 4 scalar + 4 padding)
        memory_regions.json
        memory/region_<base>.bin   -- 64 bytes of arange(uint8)
        name_map.json        -- one row matching kernel "demo_kernel"
        dispatch.json        -- mangled/demangled names + ISA
    """
    cap = tmp_path / "capture"
    (cap / "memory").mkdir(parents=True)

    region_base = 0x7F0000000000
    region_size = 64
    pointer = region_base + 16  # the captured pointer is offset within region

    kernarg = bytearray(16)
    struct.pack_into("<Q", kernarg, 0, pointer)
    struct.pack_into("<i", kernarg, 8, 1234)
    (cap / "kernarg_raw.bin").write_bytes(bytes(kernarg))

    meta = {
        "kernel_name": "demo_kernel",
        "kernel_symbol": "demo_kernel.kd",
        "grid": {"x": 8, "y": 1, "z": 1},
        "block": {"x": 64, "y": 1, "z": 1},
        "lds_size": 0,
        "hsaco_sha256": "deadbeef" * 8,
        "kernarg_segment_size_meta": 16,
        "kernarg_slots": [
            {
                "offset": 0,
                "size": 8,
                "value_kind": "global_buffer",
                "value_type": "f16",
                "address_space": "global",
                "name": "x_ptr",
            },
            {
                "offset": 8,
                "size": 4,
                "value_kind": "by_value",
                "value_type": "i32",
                "address_space": "",
                "name": "n",
            },
        ],
    }
    (cap / "metadata.json").write_text(json.dumps(meta))

    (cap / "memory_regions.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "base": region_base,
                        "size": region_size,
                        "is_pool": True,
                        "is_vmem": False,
                        "contains_kernarg": False,
                    }
                ]
            }
        )
    )

    region_bytes = bytes(range(region_size))
    region_file = cap / "memory" / f"region_{region_base:x}.bin"
    region_file.write_bytes(region_bytes)

    (cap / "name_map.json").write_text(
        json.dumps(
            [
                {
                    "user_name": "demo_kernel",
                    "hsaco_sha256": "deadbeef" * 8,
                    "hsaco_path": "",
                    "constexpr_values": {"BLOCK_SIZE": 64},
                    "signature": {"x_ptr": "*fp16", "n": "i32", "BLOCK_SIZE": "constexpr"},
                    "param_names": ["x_ptr", "n", "BLOCK_SIZE"],
                    "launch": {"num_warps": 4, "num_stages": 2},
                    "source_file": "",
                    "source_snapshot": "",
                    "tensor_layout": [
                        {
                            "index": 0,
                            "name": "x_ptr",
                            "shape": [24],
                            "stride": [1],
                            "storage_offset": 0,
                            "dtype": "torch.float16",
                            "nbytes_storage": 48,
                        }
                    ],
                }
            ]
        )
    )

    (cap / "dispatch.json").write_text(
        json.dumps(
            {
                "mangled_name": "demo_kernel.kd",
                "demangled_name": "demo_kernel.kd",
                "isa_name": "amdgcn-amd-amdhsa--gfx942",
                "wavefront_size": 64,
                "grid": [8, 1, 1],
                "block": [64, 1, 1],
            }
        )
    )

    return cap


@pytest.fixture
def synthetic_kernel_source(tmp_path: Path) -> KernelSource:
    """A standalone Triton source file containing the demo kernel."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    main = src_dir / "demo_kernels.py"
    main.write_text(
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def demo_kernel(x_ptr, n, BLOCK_SIZE: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offs < n\n"
        "    x = tl.load(x_ptr + offs, mask=mask)\n"
        "    tl.store(x_ptr + offs, x + 1.0, mask=mask)\n"
    )
    return KernelSource(
        kernel_name="demo_kernel",
        main_file=str(main),
        source_files=[str(main)],
        kernel_function="demo_kernel",
        language="triton",
    )


def test_decode_scalar_signed_int():
    buf = struct.pack("<i", -42)
    assert _decode_scalar(buf, 0, 4, "i32") == -42


def test_decode_scalar_uint64():
    buf = struct.pack("<Q", 2**63)
    assert _decode_scalar(buf, 0, 8, "u64") == 2**63


def test_decode_scalar_float32():
    buf = struct.pack("<f", 1.5)
    assert _decode_scalar(buf, 0, 4, "fp32") == pytest.approx(1.5)


def test_decode_scalar_unknown_type_falls_back_to_size():
    buf = struct.pack("<i", 7)
    assert _decode_scalar(buf, 0, 4, "weird_type") == 7


def test_find_region_for_pointer_hits_inside_range():
    regions = [{"base": 0x1000, "size": 0x100}]
    assert _find_region_for_pointer(0x1040, regions) == (regions[0], 0x40)


def test_find_region_for_pointer_misses_outside_range():
    regions = [{"base": 0x1000, "size": 0x100}]
    assert _find_region_for_pointer(0x2000, regions) is None


def test_select_name_map_row_prefers_sha_match():
    rows = [
        {"user_name": "demo", "hsaco_sha256": "aaa"},
        {"user_name": "demo", "hsaco_sha256": "bbb"},
    ]
    assert _select_name_map_row(rows, "bbb", "demo") is rows[1]


def test_select_name_map_row_falls_back_to_user_name():
    rows = [
        {"user_name": "other", "hsaco_sha256": "x"},
        {"user_name": "demo_kernel", "hsaco_sha256": "y"},
    ]
    assert _select_name_map_row(rows, "no-match", "demo_kernel") is rows[1]


def test_select_name_map_row_returns_none_when_nothing_matches():
    rows = [{"user_name": "other", "hsaco_sha256": "x"}]
    assert _select_name_map_row(rows, "no-sha", "no-name") is None


def test_generate_triton_hsa_reproducer_creates_expected_files(
    tmp_path, synthetic_capture, synthetic_kernel_source
):
    out_dir = tmp_path / "isolated"
    out_dir.mkdir()
    result = generate_triton_hsa_reproducer(
        capture_dir=str(synthetic_capture),
        kernel_source=synthetic_kernel_source,
        output_dir=str(out_dir),
    )
    assert result == str(out_dir)

    repro = out_dir / "reproducer.py"
    assert repro.is_file()
    assert os.access(repro, os.X_OK)

    assert (out_dir / "kernel_variant.py").is_file() or (out_dir / "demo_kernels.py").is_file()
    assert (out_dir / "capture" / "metadata.json").is_file()
    assert (out_dir / "capture" / "kernarg_raw.bin").is_file()
    assert (out_dir / "reference_output").is_dir()


def test_generated_reproducer_is_valid_python(tmp_path, synthetic_capture, synthetic_kernel_source):
    out_dir = tmp_path / "isolated"
    out_dir.mkdir()
    generate_triton_hsa_reproducer(
        capture_dir=str(synthetic_capture),
        kernel_source=synthetic_kernel_source,
        output_dir=str(out_dir),
    )
    body = (out_dir / "reproducer.py").read_text()
    ast.parse(body)


def test_generated_reproducer_references_args_and_constexprs(
    tmp_path, synthetic_capture, synthetic_kernel_source
):
    out_dir = tmp_path / "isolated"
    out_dir.mkdir()
    generate_triton_hsa_reproducer(
        capture_dir=str(synthetic_capture),
        kernel_source=synthetic_kernel_source,
        output_dir=str(out_dir),
    )
    body = (out_dir / "reproducer.py").read_text()
    # Pointer arg passed as kwarg; scalar arg assigned from decoded bytes.
    assert "x_ptr=x_ptr" in body
    assert "n=n" in body
    assert "n = 1234" in body  # scalar value decoded from kernarg_raw.bin
    assert "BLOCK_SIZE=64" in body  # constexpr from name_map (kwarg in launch)
    assert "num_warps=4" in body
    assert "num_stages=2" in body
    assert 'triton_dtype="*fp16"' in body
    assert "byte_offset=16" in body  # pointer was 16 bytes into the region


def test_generated_reproducer_imports_kernel_function(
    tmp_path, synthetic_capture, synthetic_kernel_source
):
    out_dir = tmp_path / "isolated"
    out_dir.mkdir()
    generate_triton_hsa_reproducer(
        capture_dir=str(synthetic_capture),
        kernel_source=synthetic_kernel_source,
        output_dir=str(out_dir),
    )
    body = (out_dir / "reproducer.py").read_text()
    assert "import demo_kernel" in body or "from kernel_variant import demo_kernel" in body
