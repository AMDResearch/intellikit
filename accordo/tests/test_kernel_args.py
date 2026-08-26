# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for kernel argument extraction.

KernelDB is substituted so these exercise Accordo's own path selection, error
translation and match logic rather than the C++ library.
"""

from unittest.mock import MagicMock, patch

import pytest

from accordo import kernel_args as ka
from accordo.exceptions import AccordoError


def _arg(name, type_name):
    a = MagicMock()
    a.name = name
    a.type_name = type_name
    return a


def _kdb(kernels=("reduce_sum",), args=None):
    inst = MagicMock()
    inst.get_kernels.return_value = list(kernels)
    inst.get_kernel_arguments.return_value = (
        args if args is not None else [_arg("input", "const float*"), _arg("n", "int")]
    )
    return inst


class TestExtractKernelArguments:
    def test_returns_name_type_tuples(self, tmp_path):
        binary = tmp_path / "app"
        binary.touch()
        inst = _kdb()
        with (
            patch.object(ka, "KERNELDB_AVAILABLE", True),
            patch.object(ka, "KernelDB", return_value=inst),
        ):
            out = ka.extract_kernel_arguments(str(binary), "reduce_sum")
        assert out == [("input", "const float*"), ("n", "int")]

    def test_resolve_typedefs_requested(self, tmp_path):
        binary = tmp_path / "app"
        binary.touch()
        inst = _kdb()
        with (
            patch.object(ka, "KERNELDB_AVAILABLE", True),
            patch.object(ka, "KernelDB", return_value=inst),
        ):
            ka.extract_kernel_arguments(str(binary), "reduce_sum")
        assert inst.get_kernel_arguments.call_args.kwargs["resolve_typedefs"] is True

    def test_substring_match_selects_kernel(self, tmp_path):
        binary = tmp_path / "app"
        binary.touch()
        inst = _kdb(kernels=("_Z10reduce_sumPfS_i",))
        with (
            patch.object(ka, "KERNELDB_AVAILABLE", True),
            patch.object(ka, "KernelDB", return_value=inst),
        ):
            ka.extract_kernel_arguments(str(binary), "reduce_sum")
        assert inst.get_kernel_arguments.call_args.args[0] == "_Z10reduce_sumPfS_i"

    def test_first_match_wins_when_ambiguous(self, tmp_path):
        binary = tmp_path / "app"
        binary.touch()
        inst = _kdb(kernels=("reduce_sum_a", "reduce_sum_b"))
        with (
            patch.object(ka, "KERNELDB_AVAILABLE", True),
            patch.object(ka, "KernelDB", return_value=inst),
        ):
            ka.extract_kernel_arguments(str(binary), "reduce_sum")
        assert inst.get_kernel_arguments.call_args.args[0] == "reduce_sum_a"

    def test_relative_path_resolved_against_working_directory(self, tmp_path):
        (tmp_path / "app").touch()
        inst = _kdb()
        with (
            patch.object(ka, "KERNELDB_AVAILABLE", True),
            patch.object(ka, "KernelDB", return_value=inst) as cls,
        ):
            ka.extract_kernel_arguments("app", "reduce_sum", working_directory=str(tmp_path))
        assert cls.call_args.args[0] == str(tmp_path / "app")

    def test_absolute_path_left_alone(self, tmp_path):
        binary = tmp_path / "app"
        binary.touch()
        inst = _kdb()
        with (
            patch.object(ka, "KERNELDB_AVAILABLE", True),
            patch.object(ka, "KernelDB", return_value=inst) as cls,
        ):
            ka.extract_kernel_arguments(str(binary), "reduce_sum", working_directory="/elsewhere")
        assert cls.call_args.args[0] == str(binary)

    def test_missing_kerneldb_raises(self):
        with patch.object(ka, "KERNELDB_AVAILABLE", False):
            with pytest.raises(AccordoError, match="kernelDB not available"):
                ka.extract_kernel_arguments("/nope", "k")

    def test_missing_binary_raises(self, tmp_path):
        with patch.object(ka, "KERNELDB_AVAILABLE", True):
            with pytest.raises(AccordoError, match="Binary not found"):
                ka.extract_kernel_arguments(str(tmp_path / "absent"), "k")

    def test_kernel_not_found_lists_available(self, tmp_path):
        binary = tmp_path / "app"
        binary.touch()
        inst = _kdb(kernels=("alpha", "beta"))
        with (
            patch.object(ka, "KERNELDB_AVAILABLE", True),
            patch.object(ka, "KernelDB", return_value=inst),
        ):
            with pytest.raises(AccordoError, match="alpha, beta"):
                ka.extract_kernel_arguments(str(binary), "gamma")

    def test_no_argument_info_mentions_debug_symbols(self, tmp_path):
        binary = tmp_path / "app"
        binary.touch()
        inst = _kdb(args=[])
        with (
            patch.object(ka, "KERNELDB_AVAILABLE", True),
            patch.object(ka, "KernelDB", return_value=inst),
        ):
            with pytest.raises(AccordoError, match="debug symbols"):
                ka.extract_kernel_arguments(str(binary), "reduce_sum")

    def test_underlying_failure_wrapped(self, tmp_path):
        binary = tmp_path / "app"
        binary.touch()
        with (
            patch.object(ka, "KERNELDB_AVAILABLE", True),
            patch.object(ka, "KernelDB", side_effect=RuntimeError("corrupt elf")),
        ):
            with pytest.raises(AccordoError, match="Failed to extract kernel arguments"):
                ka.extract_kernel_arguments(str(binary), "k")

    def test_accordo_error_not_double_wrapped(self, tmp_path):
        """An AccordoError raised inside the try block must propagate unchanged."""
        binary = tmp_path / "app"
        binary.touch()
        inst = _kdb(kernels=("other",))
        with (
            patch.object(ka, "KERNELDB_AVAILABLE", True),
            patch.object(ka, "KernelDB", return_value=inst),
        ):
            with pytest.raises(AccordoError) as exc:
                ka.extract_kernel_arguments(str(binary), "missing")
        assert "Failed to extract kernel arguments" not in str(exc.value)


class TestListAvailableKernels:
    def test_returns_kernel_names(self, tmp_path):
        binary = tmp_path / "app"
        binary.touch()
        inst = _kdb(kernels=("a", "b"))
        with (
            patch.object(ka, "KERNELDB_AVAILABLE", True),
            patch.object(ka, "KernelDB", return_value=inst),
        ):
            assert ka.list_available_kernels(str(binary)) == ["a", "b"]

    def test_relative_path_resolved(self, tmp_path):
        (tmp_path / "app").touch()
        inst = _kdb()
        with (
            patch.object(ka, "KERNELDB_AVAILABLE", True),
            patch.object(ka, "KernelDB", return_value=inst) as cls,
        ):
            ka.list_available_kernels("app", working_directory=str(tmp_path))
        assert cls.call_args.args[0] == str(tmp_path / "app")

    def test_missing_kerneldb_raises(self):
        with patch.object(ka, "KERNELDB_AVAILABLE", False):
            with pytest.raises(AccordoError, match="kernelDB not available"):
                ka.list_available_kernels("/nope")

    def test_missing_binary_raises(self, tmp_path):
        with patch.object(ka, "KERNELDB_AVAILABLE", True):
            with pytest.raises(AccordoError, match="Binary not found"):
                ka.list_available_kernels(str(tmp_path / "absent"))

    def test_failure_wrapped(self, tmp_path):
        binary = tmp_path / "app"
        binary.touch()
        with (
            patch.object(ka, "KERNELDB_AVAILABLE", True),
            patch.object(ka, "KernelDB", side_effect=RuntimeError("bad")),
        ):
            with pytest.raises(AccordoError, match="Failed to list kernels"):
                ka.list_available_kernels(str(binary))
