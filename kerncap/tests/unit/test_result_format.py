"""Unit tests for the canonical ``Result: ...`` line shape.

Locks in the contract from the kerncap output-consistency plan:

* every validation outcome (smoke / byte-exact / numeric) renders to a
  single line of the form ``Result: PASS  (...)`` or ``Result: FAIL  (...)``;
* the ``replay`` subcommand uses the same prefix and parenthetical
  schema, just with a "replayed N iterations OK" head;
* swapping a HIP capture for a Triton-HSA capture changes only the
  contents of the parenthetical, never the line shape.
"""

from __future__ import annotations

import math

from kerncap.validator import (
    ValidationResult,
    format_replay_result,
    format_result,
)


# ---------------------------------------------------------------------------
# Smoke (HIP-VA-faithful baseline + Triton-HSA baseline both land here)
# ---------------------------------------------------------------------------


def test_format_result_smoke_pass():
    result = ValidationResult(passed=True, details=[], mode="smoke")
    assert format_result(result) == "Result: PASS  (smoke)"


def test_format_result_smoke_fail():
    result = ValidationResult(passed=False, details=[], mode="smoke")
    assert format_result(result) == "Result: FAIL  (smoke)"


# ---------------------------------------------------------------------------
# Byte-exact (HIP variant via --hsaco, Triton-HSA variant via candidate.hsaco)
# ---------------------------------------------------------------------------


def test_format_result_byte_exact_all_match():
    result = ValidationResult(
        passed=True,
        details=[],
        mode="byte-exact",
        regions_total=200,
        regions_identical=200,
    )
    assert format_result(result) == ("Result: PASS  (byte-exact, 200 of 200 regions identical)")


def test_format_result_byte_exact_some_differ():
    result = ValidationResult(
        passed=False,
        details=[],
        mode="byte-exact",
        regions_total=200,
        regions_identical=197,
    )
    assert format_result(result) == (
        "Result: FAIL  (byte-exact, 197 of 200 regions identical, 3 differ)"
    )


# ---------------------------------------------------------------------------
# Numeric (Triton-Python reproducer; HIP-with-source reproducer)
# ---------------------------------------------------------------------------


def test_format_result_numeric_pass_with_max_error():
    result = ValidationResult(
        passed=True,
        details=[],
        max_error=1.23e-08,
        mode="numeric",
        atol=1e-6,
        outputs_total=3,
        outputs_passed=3,
    )
    line = format_result(result)
    assert line.startswith("Result: PASS  (numeric, ")
    assert "max_error = 1.23e-08" in line
    assert "atol = 1e-06" in line


def test_format_result_numeric_fail_reports_failing_count():
    result = ValidationResult(
        passed=False,
        details=[],
        max_error=12.5,
        mode="numeric",
        atol=1e-6,
        outputs_total=3,
        outputs_passed=2,
    )
    line = format_result(result)
    assert line.startswith("Result: FAIL  (numeric, ")
    assert "1 of 3 outputs differ" in line
    assert "max_error = 1.25e+01" in line


def test_format_result_numeric_handles_nan_max_error():
    result = ValidationResult(
        passed=False,
        details=[],
        max_error=float("nan"),
        mode="numeric",
        atol=1e-6,
        outputs_total=1,
        outputs_passed=0,
    )
    line = format_result(result)
    assert "max_error = nan" in line


# ---------------------------------------------------------------------------
# Schema invariants (the "looks the same" guarantee)
# ---------------------------------------------------------------------------


def test_every_result_line_starts_with_canonical_prefix():
    """No matter the mode/outcome, the line begins with ``Result: PASS  (``
    or ``Result: FAIL  (``.  Lets a tail-grep audit work uniformly."""
    cases = [
        ValidationResult(passed=True, details=[], mode="smoke"),
        ValidationResult(passed=False, details=[], mode="smoke"),
        ValidationResult(
            passed=True,
            details=[],
            mode="byte-exact",
            regions_total=1,
            regions_identical=1,
        ),
        ValidationResult(
            passed=False,
            details=[],
            mode="byte-exact",
            regions_total=2,
            regions_identical=1,
        ),
        ValidationResult(
            passed=True,
            details=[],
            max_error=0.0,
            mode="numeric",
            atol=1e-6,
            outputs_total=1,
            outputs_passed=1,
        ),
        ValidationResult(
            passed=False,
            details=[],
            max_error=1.0,
            mode="numeric",
            atol=1e-6,
            outputs_total=1,
            outputs_passed=0,
        ),
    ]
    for r in cases:
        line = format_result(r)
        prefix = "Result: PASS  (" if r.passed else "Result: FAIL  ("
        assert line.startswith(prefix), f"bad prefix for {r!r}: {line!r}"
        assert line.endswith(")"), f"missing closing paren: {line!r}"


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_format_replay_result_pass_with_timing():
    line = format_replay_result(
        returncode=0,
        iterations=5,
        avg_us=7.30,
        min_us=4.49,
        max_us=8.17,
    )
    assert line.startswith("Result: PASS  ")
    assert "replayed 5 iteration(s) OK" in line
    assert "avg = 7.30 us" in line
    assert "min = 4.49" in line
    assert "max = 8.17" in line


def test_format_replay_result_pass_without_timing():
    """Even with no timing data, the line shape is preserved."""
    line = format_replay_result(
        returncode=0,
        iterations=1,
        avg_us=None,
        min_us=None,
        max_us=None,
    )
    assert line.startswith("Result: PASS  ")
    assert "replayed 1 iteration(s) OK" in line


def test_format_replay_result_fail_includes_returncode():
    line = format_replay_result(
        returncode=137,
        iterations=10,
        avg_us=None,
        min_us=None,
        max_us=None,
    )
    assert line.startswith("Result: FAIL  ")
    assert "137" in line
