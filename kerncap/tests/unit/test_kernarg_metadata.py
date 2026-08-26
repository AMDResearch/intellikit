"""Unit tests for the AMDGPU kernarg metadata parser (kernarg_metadata.cpp).

Drives the C parser via a tiny ctypes shim (``kerncap_parse_kernarg_metadata_json``
exported from ``libkerncap.so``).  The fixture under
``tests/unit/fixtures/hsaco/triton_poi_fused_relu_0.hsaco`` is the relu
HSACO captured during the Phase 1 spike at
``kerncap_paper/spikes/triton_hsa_capture/REPORT.md``.

These tests do not require a GPU or any HSA runtime; they exercise only
the parser and the comgr path.
"""

import ctypes
import json
import os
from pathlib import Path

import pytest


FIXTURE = Path(__file__).parent / "fixtures" / "hsaco" / "triton_poi_fused_relu_0.hsaco"


def _load_parser():
    """Load libkerncap.so and return the ctypes-bound parser entry point."""
    try:
        from kerncap import _get_lib_path
    except Exception as e:
        pytest.skip(f"kerncap not importable: {e}")
    try:
        lib_path = _get_lib_path()
    except FileNotFoundError as e:
        pytest.skip(str(e))

    lib = ctypes.CDLL(lib_path)
    libc = ctypes.CDLL("libc.so.6")

    fn = lib.kerncap_parse_kernarg_metadata_json
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    fn.restype = ctypes.c_void_p  # caller must free
    libc.free.argtypes = [ctypes.c_void_p]
    libc.free.restype = None
    return fn, libc


def _parse(hsaco_bytes: bytes):
    fn, libc = _load_parser()
    buf = (ctypes.c_ubyte * len(hsaco_bytes)).from_buffer_copy(hsaco_bytes)
    raw = fn(ctypes.cast(buf, ctypes.c_void_p), len(hsaco_bytes))
    if not raw:
        raise RuntimeError("kerncap_parse_kernarg_metadata_json returned NULL")
    try:
        s = ctypes.string_at(raw).decode()
    finally:
        libc.free(raw)
    return json.loads(s)


@pytest.fixture(scope="module")
def relu_hsaco_bytes():
    if not FIXTURE.is_file():
        pytest.skip(f"fixture HSACO not present: {FIXTURE}")
    return FIXTURE.read_bytes()


def test_parser_returns_one_kernel(relu_hsaco_bytes):
    kernels = _parse(relu_hsaco_bytes)
    assert isinstance(kernels, list)
    assert len(kernels) == 1
    k = kernels[0]
    assert k["name"] == "triton_poi_fused_relu_0"
    assert k["symbol"] == "triton_poi_fused_relu_0.kd"


def test_relu_kernel_segment_sizes(relu_hsaco_bytes):
    """Matches the spike's recorded values for triton_poi_fused_relu_0."""
    k = _parse(relu_hsaco_bytes)[0]
    assert k["kernarg_segment_size"] == 32
    assert k["group_segment_fixed_size"] == 0
    assert k["private_segment_fixed_size"] == 0
    assert k["sgpr_count"] == 17
    assert k["vgpr_count"] == 5


def test_relu_kernel_arg_layout(relu_hsaco_bytes):
    """The exact slot table we recorded in the spike's REPORT.md."""
    k = _parse(relu_hsaco_bytes)[0]
    assert len(k["args"]) == 4

    expected = [
        # offset, size, value_kind, address_space
        (0, 8, "global_buffer", "global"),
        (8, 4, "by_value", ""),
        (16, 8, "global_buffer", "global"),
        (24, 8, "global_buffer", "global"),
    ]
    for slot, (off, sz, kind, addr) in zip(k["args"], expected):
        assert slot["offset"] == off
        assert slot["size"] == sz
        assert slot["value_kind"] == kind
        assert slot["address_space"] == addr


def test_no_constexpr_in_kernarg_layout(relu_hsaco_bytes):
    """The whole point of the plan: XBLOCK and other constexprs MUST NOT
    appear as kernarg slots, because they're folded into the IR."""
    k = _parse(relu_hsaco_bytes)[0]
    by_value_slots = [a for a in k["args"] if a["value_kind"] == "by_value"]
    # The relu kernel has exactly one runtime scalar (xnumel) -- XBLOCK is
    # a tl.constexpr that does NOT show up here.  This is the structural
    # guarantee that the XBLOCK=None failure mode is impossible at the HSA
    # layer.
    assert len(by_value_slots) == 1
    assert by_value_slots[0]["offset"] == 8
    assert by_value_slots[0]["size"] == 4


def test_no_hidden_slots_for_modern_triton(relu_hsaco_bytes):
    """Triton 3.1+ on ROCm 7.x does not emit hidden_* slots; the parser
    must handle their absence gracefully (it does)."""
    k = _parse(relu_hsaco_bytes)[0]
    hidden = [a for a in k["args"] if a["value_kind"].startswith("hidden_")]
    assert hidden == []


def test_parser_rejects_garbage_input():
    fn, libc = _load_parser()
    # 16 bytes of zeros is not a valid ELF; parser must return an empty
    # list rather than crash.
    junk = b"\0" * 16
    buf = (ctypes.c_ubyte * len(junk)).from_buffer_copy(junk)
    raw = fn(ctypes.cast(buf, ctypes.c_void_p), len(junk))
    try:
        s = ctypes.string_at(raw).decode()
    finally:
        libc.free(raw)
    assert json.loads(s) == []
