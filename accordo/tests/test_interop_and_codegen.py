# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for HIP interop guards and codegen type sizing.

Only the validation and error-translation branches are covered. The actual
``hipIpcOpenMemHandle`` / ``hipMemcpy`` calls are left to the GPU tests -- a
stubbed HIP runtime would prove nothing about them.
"""

import ctypes
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from accordo._internal import codegen
from accordo._internal import hip_interop as hip


class TestHipTry:
    def test_zero_is_success_and_returns_none(self):
        assert hip.hip_try(0) is None

    def test_nonzero_raises_with_decoded_message(self):
        runtime = MagicMock()
        runtime.hipGetErrorString.return_value = b"out of memory"
        with patch.object(hip, "hip_runtime", runtime):
            with pytest.raises(RuntimeError, match="HIP error code 2: out of memory"):
                hip.hip_try(2)

    def test_error_string_restype_is_set_before_the_call(self):
        """Without c_char_p the returned pointer would decode as an int."""
        runtime = MagicMock()
        runtime.hipGetErrorString.return_value = b"boom"
        with patch.object(hip, "hip_runtime", runtime):
            with pytest.raises(RuntimeError):
                hip.hip_try(1)
        assert runtime.hipGetErrorString.restype is ctypes.c_char_p


class TestIpcMemHandleStruct:
    def test_reserved_field_is_64_bytes(self):
        assert ctypes.sizeof(hip.hipIpcMemHandle_t) == 64


class TestOpenIpcHandleValidation:
    def test_non_array_rejected(self):
        with pytest.raises(TypeError, match="numpy.ndarray"):
            hip.open_ipc_handle(b"\x00" * 64)

    def test_list_rejected(self):
        with pytest.raises(TypeError, match="numpy.ndarray"):
            hip.open_ipc_handle([0] * 64)

    def test_wrong_dtype_rejected(self):
        with pytest.raises(ValueError, match="64-element uint8"):
            hip.open_ipc_handle(np.zeros(64, dtype=np.float32))

    def test_wrong_size_rejected(self):
        with pytest.raises(ValueError, match="64-element uint8"):
            hip.open_ipc_handle(np.zeros(32, dtype=np.uint8))

    def test_oversized_array_rejected(self):
        with pytest.raises(ValueError, match="64-element uint8"):
            hip.open_ipc_handle(np.zeros(128, dtype=np.uint8))


class TestGetTypeSize:
    @pytest.mark.parametrize(
        ("type_str", "expected"),
        [("float", 4), ("double", 8), ("int", 4)],
    )
    def test_known_scalar_types(self, type_str, expected):
        assert codegen._get_type_size(type_str) == expected

    def test_const_qualifier_stripped(self):
        assert codegen._get_type_size("const float") == codegen._get_type_size("float")

    def test_volatile_qualifier_stripped(self):
        assert codegen._get_type_size("volatile int") == codegen._get_type_size("int")

    def test_surrounding_whitespace_ignored(self):
        assert codegen._get_type_size("  float  ") == codegen._get_type_size("float")

    def test_unknown_type_falls_back_to_eight_bytes(self):
        assert codegen._get_type_size("struct MyOpaqueThing") == 8

    def test_unknown_type_warns(self, caplog):
        with caplog.at_level("WARNING"):
            codegen._get_type_size("some_unknown_t")
        assert "Unknown type size" in caplog.text


class TestPackageVersion:
    def test_version_is_exposed(self):
        import accordo

        assert isinstance(accordo.__version__, str)
        assert accordo.__version__

    def test_falls_back_when_distribution_missing(self):
        """An in-tree checkout with no installed dist must still import."""
        import importlib

        from importlib.metadata import PackageNotFoundError

        import accordo

        with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            reloaded = importlib.reload(accordo)
            assert reloaded.__version__ == "0.4.0"
        importlib.reload(accordo)  # restore the real version for other tests
