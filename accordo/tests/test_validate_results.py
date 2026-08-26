# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for snapshot comparison.

``compare_snapshots`` and ``_validate_results`` are pure array logic -- the only
attribute either reads from the instance is ``kernel_args``. They were reachable
before only through a live GPU capture, so the instance is built with
``object.__new__`` to skip the hardware setup in ``__init__``. The arrays and the
comparison itself are real.
"""

import numpy as np
import pytest

from accordo.snapshot import Snapshot
from accordo.validator import Accordo

# Two outputs and one input; only the non-const pointers produce arrays.
KERNEL_ARGS = [
    ("input", "const float*"),
    ("output", "float*"),
    ("n", "int"),
    ("acc", "double*"),
]


def _validator(kernel_args=None):
    v = object.__new__(Accordo)
    v.kernel_args = KERNEL_ARGS if kernel_args is None else kernel_args
    return v


def _snap(arrays=None, dispatch_arrays=None, exec_ms=1.0):
    return Snapshot(
        arrays=arrays if arrays is not None else [],
        execution_time_ms=exec_ms,
        binary=["./app"],
        working_directory=".",
        dispatch_arrays=dispatch_arrays,
    )


def _run(ref, opt, kernel_args=None, **kw):
    return _validator(kernel_args).compare_snapshots(_snap(ref), _snap(opt), **kw)


class TestMatching:
    def test_identical_arrays_pass(self):
        a = [np.ones(4), np.zeros(3)]
        r = _run(a, [x.copy() for x in a])
        assert r.is_valid is True
        assert r.num_mismatches == 0
        assert r.error_message is None

    def test_matched_arrays_keyed_by_dispatch_and_argname(self):
        a = [np.ones(4), np.zeros(3)]
        r = _run(a, [x.copy() for x in a])
        assert set(r.matched_arrays) == {"dispatch_0:output", "dispatch_0:acc"}

    def test_matched_entry_carries_kernel_arg_index_not_array_index(self):
        """Array 1 is kernel arg 3 -- the const input is skipped in the mapping."""
        a = [np.ones(4), np.zeros(3)]
        r = _run(a, [x.copy() for x in a])
        assert r.matched_arrays["dispatch_0:output"]["index"] == 1
        assert r.matched_arrays["dispatch_0:acc"]["index"] == 3

    def test_matched_entry_records_type_size_and_dispatch(self):
        a = [np.ones(4), np.zeros(3)]
        entry = _run(a, [x.copy() for x in a]).matched_arrays["dispatch_0:output"]
        assert entry["type"] == "float*"
        assert entry["size"] == 4
        assert entry["dispatch"] == 0
        assert entry["arg_name"] == "output"

    def test_execution_times_propagated(self):
        v = _validator()
        r = v.compare_snapshots(_snap([np.ones(2)], exec_ms=5.0), _snap([np.ones(2)], exec_ms=7.0))
        assert r.execution_time_ms == {"reference": 5.0, "optimized": 7.0}


class TestMismatches:
    def test_differing_arrays_fail(self):
        r = _run([np.zeros(4)], [np.ones(4)])
        assert r.is_valid is False
        assert r.num_mismatches == 1

    def test_mismatch_carries_kernel_arg_metadata(self):
        m = _run([np.zeros(4)], [np.ones(4)]).mismatches[0]
        assert m.arg_index == 1
        assert m.arg_name == "output"
        assert m.arg_type == "float*"
        assert m.dispatch_index == 0

    def test_max_and_mean_difference_computed(self):
        ref = np.array([0.0, 0.0, 0.0, 0.0])
        opt = np.array([1.0, 3.0, 0.0, 0.0])
        m = _run([ref], [opt]).mismatches[0]
        assert m.max_difference == 3.0
        assert m.mean_difference == 1.0

    def test_all_nan_difference_falls_back_to_zero(self):
        """Every diff is NaN, so there is no finite value to reduce over."""
        ref = np.array([np.nan, np.nan])
        opt = np.array([1.0, 2.0])
        m = _run([ref], [opt]).mismatches[0]
        assert m.max_difference == 0.0
        assert m.mean_difference == 0.0

    def test_samples_truncated_at_ten_elements(self):
        ref = np.zeros(50)
        opt = np.ones(50)
        m = _run([ref], [opt]).mismatches[0]
        assert len(m.reference_sample) == 10
        assert len(m.optimized_sample) == 10

    def test_short_arrays_are_not_truncated(self):
        m = _run([np.zeros(3)], [np.ones(3)]).mismatches[0]
        assert len(m.reference_sample) == 3

    def test_error_message_lists_every_mismatch(self):
        r = _run([np.zeros(2), np.zeros(2)], [np.ones(2), np.ones(2)])
        assert "2 array(s) mismatched" in r.error_message
        assert r.error_message.count("  - ") == 2

    def test_partial_failure_reports_both_sides(self):
        same = np.ones(3)
        r = _run([same, np.zeros(2)], [same.copy(), np.ones(2)])
        assert r.is_valid is False
        assert r.num_mismatches == 1
        assert set(r.matched_arrays) == {"dispatch_0:output"}


class TestTolerance:
    def test_within_atol_passes(self):
        r = _run([np.array([1.0])], [np.array([1.0 + 1e-9])], atol=1e-8, rtol=0.0)
        assert r.is_valid is True

    def test_beyond_tolerance_fails(self):
        r = _run([np.array([1.0])], [np.array([1.5])], atol=1e-8, rtol=1e-8)
        assert r.is_valid is False

    def test_tolerance_alias_overrides_atol(self):
        """`tolerance` is the legacy name and must win when both are given."""
        ref, opt = [np.array([1.0])], [np.array([1.1])]
        assert _run(ref, opt, atol=1e-9).is_valid is False
        assert _run(ref, opt, tolerance=1.0, atol=1e-9).is_valid is True

    def test_equal_nan_toggle(self):
        ref, opt = [np.array([np.nan])], [np.array([np.nan])]
        assert _run(ref, opt).is_valid is False
        assert _run(ref, opt, equal_nan=True).is_valid is True


class TestDispatches:
    def test_dispatch_arrays_preferred_over_arrays(self):
        v = _validator()
        ref = Snapshot(
            arrays=[np.zeros(2)],
            execution_time_ms=1.0,
            binary=["./a"],
            working_directory=".",
            dispatch_arrays=[[np.ones(2)], [np.ones(2)]],
        )
        opt = Snapshot(
            arrays=[np.zeros(2)],
            execution_time_ms=1.0,
            binary=["./a"],
            working_directory=".",
            dispatch_arrays=[[np.ones(2)], [np.ones(2)]],
        )
        r = v.compare_snapshots(ref, opt)
        assert r.is_valid is True
        assert set(r.matched_arrays) == {"dispatch_0:output", "dispatch_1:output"}

    def test_second_dispatch_mismatch_detected(self):
        v = _validator()
        ref = _snap(dispatch_arrays=[[np.ones(2)], [np.ones(2)]])
        opt = _snap(dispatch_arrays=[[np.ones(2)], [np.zeros(2)]])
        r = v.compare_snapshots(ref, opt)
        assert r.is_valid is False
        assert r.mismatches[0].dispatch_index == 1

    def test_dispatch_count_mismatch_reported(self):
        v = _validator()
        ref = _snap(dispatch_arrays=[[np.ones(2)], [np.ones(2)]])
        opt = _snap(dispatch_arrays=[[np.ones(2)]])
        r = v.compare_snapshots(ref, opt)
        assert r.is_valid is False
        assert "Dispatch count mismatch: 2 vs 1" in r.error_message
        assert r.mismatches == []

    def test_array_count_mismatch_reported(self):
        r = _run([np.ones(2), np.ones(2)], [np.ones(2)])
        assert r.is_valid is False
        assert "Array count mismatch at dispatch 0: 2 vs 1" in r.error_message

    def test_array_count_mismatch_names_the_dispatch(self):
        v = _validator()
        ref = _snap(dispatch_arrays=[[np.ones(2)], [np.ones(2), np.ones(2)]])
        opt = _snap(dispatch_arrays=[[np.ones(2)], [np.ones(2)]])
        r = v.compare_snapshots(ref, opt)
        assert "at dispatch 1" in r.error_message


class TestOutputArgMapping:
    def test_const_pointers_excluded_from_mapping(self):
        args = [("in", "const float*"), ("out", "float*")]
        r = _run([np.ones(2)], [np.ones(2)], kernel_args=args)
        assert r.matched_arrays["dispatch_0:out"]["index"] == 1

    def test_scalars_excluded_from_mapping(self):
        args = [("n", "int"), ("out", "float*")]
        r = _run([np.ones(2)], [np.ones(2)], kernel_args=args)
        assert "dispatch_0:out" in r.matched_arrays

    def test_multiple_outputs_map_in_declaration_order(self):
        args = [("a", "float*"), ("n", "int"), ("b", "double*")]
        r = _run([np.ones(2), np.zeros(2)], [np.ones(2), np.zeros(2)], kernel_args=args)
        assert r.matched_arrays["dispatch_0:a"]["index"] == 0
        assert r.matched_arrays["dispatch_0:b"]["index"] == 2

    def test_no_arrays_is_vacuously_valid(self):
        r = _run([], [])
        assert r.is_valid is True
        assert r.matched_arrays == {}


@pytest.mark.parametrize("n", [1, 10, 11, 100])
def test_sample_length_never_exceeds_ten(n):
    m = _run([np.zeros(n)], [np.ones(n)]).mismatches[0]
    assert len(m.reference_sample) == min(n, 10)
