# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the result types and the module-level validator helpers.

``Accordo`` itself drives real GPU processes and is exercised by
``test_reduction_validation.py``. Covered here are the pieces that carry logic
but no hardware: the result dataclasses, the tolerance comparison, the timeout
signal handler, and the build wrapper's error translation.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from accordo import validator as V
from accordo.exceptions import AccordoBuildError
from accordo.result import ArrayMismatch, ValidationResult


def _mismatch(**over):
    base = dict(
        arg_index=0,
        arg_name="output",
        arg_type="float*",
        max_difference=1.5e-3,
        mean_difference=2.5e-4,
        reference_sample=np.zeros(3),
        optimized_sample=np.ones(3),
    )
    base.update(over)
    return ArrayMismatch(**base)


class TestArrayMismatch:
    def test_dispatch_index_defaults_to_none(self):
        assert _mismatch().dispatch_index is None

    def test_str_reports_name_type_and_diffs(self):
        s = str(_mismatch())
        assert "arg 'output'" in s
        assert "(float*)" in s
        assert "max_diff=1.50e-03" in s
        assert "mean_diff=2.50e-04" in s

    def test_str_omits_dispatch_prefix_when_absent(self):
        assert "dispatch" not in str(_mismatch())

    def test_str_includes_dispatch_prefix_when_set(self):
        assert str(_mismatch(dispatch_index=4)).startswith("Mismatch in dispatch 4: arg")

    def test_dispatch_index_zero_is_still_shown(self):
        """0 is a valid dispatch index and must not be treated as absent."""
        assert "dispatch 0:" in str(_mismatch(dispatch_index=0))


class TestValidationResultDefaults:
    def test_collections_default_to_empty_not_none(self):
        r = ValidationResult(is_valid=True)
        assert r.mismatches == []
        assert r.matched_arrays == {}
        assert r.execution_time_ms == {}

    def test_supplied_collections_are_kept(self):
        m = [_mismatch()]
        r = ValidationResult(
            is_valid=False, mismatches=m, matched_arrays={"a": {}}, execution_time_ms={"ref": 1.0}
        )
        assert r.mismatches is m
        assert r.matched_arrays == {"a": {}}
        assert r.execution_time_ms == {"ref": 1.0}

    def test_optional_scalars_default_to_none(self):
        r = ValidationResult(is_valid=True)
        assert r.error_message is None
        assert r.timeout_used is None


class TestValidationResultCounts:
    def test_num_arrays_counts_matched_and_mismatched(self):
        r = ValidationResult(
            is_valid=False, mismatches=[_mismatch(), _mismatch()], matched_arrays={"a": {}}
        )
        assert r.num_arrays_validated == 3
        assert r.num_mismatches == 2

    def test_empty_result_counts_zero(self):
        r = ValidationResult(is_valid=True)
        assert r.num_arrays_validated == 0
        assert r.num_mismatches == 0

    def test_success_rate_all_matched(self):
        r = ValidationResult(is_valid=True, matched_arrays={"a": {}, "b": {}})
        assert r.success_rate == 100.0

    def test_success_rate_partial(self):
        r = ValidationResult(is_valid=False, mismatches=[_mismatch()], matched_arrays={"a": {}})
        assert r.success_rate == 50.0

    def test_success_rate_none_matched(self):
        r = ValidationResult(is_valid=False, mismatches=[_mismatch()])
        assert r.success_rate == 0.0

    def test_success_rate_zero_arrays_does_not_divide_by_zero(self):
        assert ValidationResult(is_valid=True).success_rate == 0.0


class TestValidationResultSummary:
    def test_pass_summary_reports_count(self):
        r = ValidationResult(is_valid=True, matched_arrays={"a": {}, "b": {}})
        assert "Validation passed" in r.summary()
        assert "2 arrays" in r.summary()

    def test_fail_summary_includes_error_message(self):
        r = ValidationResult(is_valid=False, error_message="tolerance exceeded")
        assert "Validation failed" in r.summary()
        assert "tolerance exceeded" in r.summary()

    def test_fail_summary_lists_each_mismatch(self):
        r = ValidationResult(
            is_valid=False,
            error_message="bad",
            mismatches=[_mismatch(arg_name="a"), _mismatch(arg_name="b")],
        )
        out = r.summary()
        assert "Mismatched arrays (2)" in out
        assert "arg 'a'" in out
        assert "arg 'b'" in out

    def test_fail_summary_without_mismatches_omits_the_list(self):
        r = ValidationResult(is_valid=False, error_message="crashed before comparison")
        assert "Mismatched arrays" not in r.summary()

    def test_str_delegates_to_summary(self):
        r = ValidationResult(is_valid=True)
        assert str(r) == r.summary()


class TestValidateArrays:
    def test_identical_arrays_match(self):
        a = np.array([1.0, 2.0, 3.0])
        assert V._validate_arrays(a, a.copy(), atol=0.0, rtol=0.0, equal_nan=False) is True

    def test_difference_within_atol_matches(self):
        a = np.array([1.0])
        b = np.array([1.0 + 1e-9])
        assert V._validate_arrays(a, b, atol=1e-8, rtol=0.0, equal_nan=False) is True

    def test_difference_beyond_tolerance_fails(self):
        a = np.array([1.0])
        b = np.array([1.5])
        assert V._validate_arrays(a, b, atol=1e-8, rtol=1e-8, equal_nan=False) is False

    def test_rtol_scales_with_magnitude(self):
        a = np.array([1000.0])
        b = np.array([1000.1])
        assert V._validate_arrays(a, b, atol=0.0, rtol=1e-3, equal_nan=False) is True
        assert V._validate_arrays(a, b, atol=0.0, rtol=1e-9, equal_nan=False) is False

    def test_nan_mismatch_unless_equal_nan(self):
        a = np.array([np.nan])
        b = np.array([np.nan])
        assert V._validate_arrays(a, b, atol=0.0, rtol=0.0, equal_nan=False) is False
        assert V._validate_arrays(a, b, atol=0.0, rtol=0.0, equal_nan=True) is True


class TestTimeoutHandler:
    def test_raises_timeout_exception(self):
        with pytest.raises(V._TimeoutException, match="timed out"):
            V._timeout_handler(14, None)

    def test_timeout_exception_is_an_exception(self):
        assert issubclass(V._TimeoutException, Exception)


class TestBuildAccordo:
    def _run_ok(self):
        r = MagicMock()
        r.stdout = "ok"
        r.stderr = ""
        return r

    def test_returns_library_path_on_success(self, tmp_path):
        lib = tmp_path / "build" / "lib" / "libaccordo.so"
        lib.parent.mkdir(parents=True)
        lib.touch()
        with patch.object(V.subprocess, "run", return_value=self._run_ok()):
            assert V._build_accordo(tmp_path) == lib

    def test_configure_then_build_are_both_invoked(self, tmp_path):
        lib = tmp_path / "build" / "lib" / "libaccordo.so"
        lib.parent.mkdir(parents=True)
        lib.touch()
        with patch.object(V.subprocess, "run", return_value=self._run_ok()) as run:
            V._build_accordo(tmp_path, parallel_jobs=4)
        assert run.call_args_list[0].args[0] == ["cmake", "-B", "build"]
        assert run.call_args_list[1].args[0] == [
            "cmake",
            "--build",
            "build",
            "--parallel",
            "4",
        ]

    def test_missing_library_after_build_raises(self, tmp_path):
        with patch.object(V.subprocess, "run", return_value=self._run_ok()):
            with pytest.raises(AccordoBuildError, match="Library not found"):
                V._build_accordo(tmp_path)

    def test_cmake_failure_is_translated(self, tmp_path):
        err = subprocess.CalledProcessError(1, "cmake", stderr="no CMAKE_CXX_COMPILER")
        with patch.object(V.subprocess, "run", side_effect=err):
            with pytest.raises(AccordoBuildError, match="no CMAKE_CXX_COMPILER"):
                V._build_accordo(tmp_path)

    def test_unexpected_error_is_translated(self, tmp_path):
        with patch.object(V.subprocess, "run", side_effect=OSError("cmake not found")):
            with pytest.raises(AccordoBuildError, match="cmake not found"):
                V._build_accordo(tmp_path)

    def test_build_runs_in_the_given_directory(self, tmp_path):
        lib = tmp_path / "build" / "lib" / "libaccordo.so"
        lib.parent.mkdir(parents=True)
        lib.touch()
        with patch.object(V.subprocess, "run", return_value=self._run_ok()) as run:
            V._build_accordo(Path(tmp_path))
        for call in run.call_args_list:
            assert call.kwargs["cwd"] == Path(tmp_path)
            assert call.kwargs["check"] is True
