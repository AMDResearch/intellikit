# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the MessagePack codec."""

import pytest

from proboscis.msgpack_codec import packb, unpackb


class TestRoundtrip:
    @pytest.mark.parametrize("value", [
        None,
        True,
        False,
        0,
        1,
        127,
        128,
        255,
        256,
        65535,
        65536,
        -1,
        -32,
        -33,
        -128,
        -129,
        0.5,
        3.14,
        "",
        "hello",
        "a" * 32,
        "a" * 256,
        [],
        [1, 2, 3],
        {},
        {"key": "value"},
        {"a": 1, "b": [2, 3], "c": {"d": True}},
    ])
    def test_roundtrip(self, value):
        encoded = packb(value)
        decoded = unpackb(encoded)
        assert decoded == value

    def test_bytes_roundtrip(self):
        value = b"\x00\x01\x02\x03"
        encoded = packb(value)
        decoded = unpackb(encoded)
        assert decoded == value

    def test_nested_structure(self):
        value = {
            "amdhsa.kernels": [
                {
                    ".name": "vec_add",
                    ".kernarg_segment_size": 32,
                    ".args": [
                        {".offset": 0, ".size": 8, ".value_kind": "global_buffer"},
                    ],
                }
            ],
            "amdhsa.target": "amdgcn-amd-amdhsa--gfx942",
        }
        encoded = packb(value)
        decoded = unpackb(encoded)
        assert decoded == value

    def test_trailing_data_raises(self):
        encoded = packb(42) + b"\x00"
        with pytest.raises(ValueError, match="trailing data"):
            unpackb(encoded)
