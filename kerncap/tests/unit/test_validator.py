"""Unit tests for kerncap.validator — output comparison logic."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from kerncap.validator import (
    _compare_outputs,
    _compare_replay_outputs,
    _infer_numpy_dtype,
    _validate_replay,
    ValidationResult,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_reproducer(tmp_path):
    """Create a mock reproducer directory with capture data and outputs."""
    repro = tmp_path / "repro"
    capture = repro / "capture"
    ref_out = repro / "reference_output"
    capture.mkdir(parents=True)
    ref_out.mkdir(parents=True)

    n = 1024
    meta = {
        "kernel_name": "test_kernel",
        "grid": {"x": 4, "y": 1, "z": 1},
        "block": {"x": 256, "y": 1, "z": 1},
        "lds_size": 0,
        "args": [
            {
                "index": 0, "name": "input", "type": "const float*",
                "size": 8, "offset": 0, "is_pointer": True,
                "is_const": True, "is_output": False,
                "buffer_size": n * 4, "file": "arg_0.bin",
            },
            {
                "index": 1, "name": "output", "type": "float*",
                "size": 8, "offset": 8, "is_pointer": True,
                "is_const": False, "is_output": True,
                "buffer_size": n * 4, "file": "arg_1.bin",
            },
        ],
    }
    (capture / "metadata.json").write_text(json.dumps(meta))

    # Reference data (what the original kernel produced)
    ref_data = np.random.randn(n).astype(np.float32)
    ref_data.tofile(str(capture / "arg_1.bin"))

    return repro, ref_data, meta


class TestCompareOutputs:
    """Tests for output comparison logic."""

    def test_identical_outputs_pass(self, mock_reproducer):
        repro, ref_data, meta = mock_reproducer
        # Write identical output
        ref_data.tofile(str(repro / "reference_output" / "output_1.bin"))

        result = _compare_outputs(str(repro), meta, atol=1e-6, rtol=1e-5, details=[])
        assert result.passed

    def test_small_difference_passes(self, mock_reproducer):
        repro, ref_data, meta = mock_reproducer
        # Add tiny noise within tolerance
        noisy = ref_data + np.random.randn(len(ref_data)).astype(np.float32) * 1e-8
        noisy.tofile(str(repro / "reference_output" / "output_1.bin"))

        result = _compare_outputs(str(repro), meta, atol=1e-6, rtol=1e-5, details=[])
        assert result.passed

    def test_large_difference_fails(self, mock_reproducer):
        repro, ref_data, meta = mock_reproducer
        # Add large noise
        wrong = ref_data + 1.0
        wrong.tofile(str(repro / "reference_output" / "output_1.bin"))

        result = _compare_outputs(str(repro), meta, atol=1e-6, rtol=1e-5, details=[])
        assert not result.passed
        assert result.max_error > 0.5

    def test_missing_output_fails(self, mock_reproducer):
        repro, _, meta = mock_reproducer
        # Don't write any output file
        result = _compare_outputs(str(repro), meta, atol=1e-6, rtol=1e-5, details=[])
        assert not result.passed

    def test_no_output_args_passes(self, tmp_path):
        """Kernel with no output pointers should pass trivially."""
        repro = tmp_path / "repro"
        capture = repro / "capture"
        capture.mkdir(parents=True)

        meta = {
            "args": [
                {"index": 0, "type": "const float*", "is_pointer": True,
                 "is_const": True, "is_output": False, "file": "arg_0.bin"},
            ],
        }
        (capture / "metadata.json").write_text(json.dumps(meta))

        result = _compare_outputs(str(repro), meta, atol=1e-6, rtol=1e-5, details=[])
        assert result.passed


class TestNaNHandling:
    """Tests for data containing NaN values."""

    def test_nan_in_reference_fails_with_note(self, tmp_path):
        repro = tmp_path / "repro"
        capture = repro / "capture"
        ref_out = repro / "reference_output"
        capture.mkdir(parents=True)
        ref_out.mkdir(parents=True)

        n = 256
        meta = {
            "args": [
                {
                    "index": 0, "name": "out", "type": "float*",
                    "size": 8, "offset": 0, "is_pointer": True,
                    "is_const": False, "is_output": True,
                    "buffer_size": n * 4, "file": "arg_0.bin",
                },
            ],
        }
        (capture / "metadata.json").write_text(json.dumps(meta))

        ref = np.ones(n, dtype=np.float32)
        ref[10] = np.nan
        ref.tofile(str(capture / "arg_0.bin"))

        out = np.ones(n, dtype=np.float32)
        out.tofile(str(ref_out / "output_0.bin"))

        result = _compare_outputs(str(repro), meta, atol=1e-6, rtol=1e-5, details=[])
        assert not result.passed
        assert any("NaN" in d for d in result.details)
        assert any("Note:" in d for d in result.details)
        assert result.max_error != result.max_error  # NaN != NaN

    def test_nan_in_both_ref_and_out_fails(self, tmp_path):
        repro = tmp_path / "repro"
        capture = repro / "capture"
        ref_out = repro / "reference_output"
        capture.mkdir(parents=True)
        ref_out.mkdir(parents=True)

        n = 128
        meta = {
            "args": [
                {
                    "index": 0, "name": "out", "type": "float*",
                    "size": 8, "offset": 0, "is_pointer": True,
                    "is_const": False, "is_output": True,
                    "buffer_size": n * 4, "file": "arg_0.bin",
                },
            ],
        }
        (capture / "metadata.json").write_text(json.dumps(meta))

        data = np.full(n, np.nan, dtype=np.float32)
        data.tofile(str(capture / "arg_0.bin"))
        data.tofile(str(ref_out / "output_0.bin"))

        result = _compare_outputs(str(repro), meta, atol=1e-6, rtol=1e-5, details=[])
        assert not result.passed
        assert any("NaN" in d for d in result.details)

    def test_nan_does_not_mask_real_error(self, tmp_path):
        """max_error should be NaN when any arg has NaN, even if other args have finite errors."""
        repro = tmp_path / "repro"
        capture = repro / "capture"
        ref_out = repro / "reference_output"
        capture.mkdir(parents=True)
        ref_out.mkdir(parents=True)

        n = 64
        meta = {
            "args": [
                {
                    "index": 0, "name": "a", "type": "float*",
                    "size": 8, "offset": 0, "is_pointer": True,
                    "is_const": False, "is_output": True,
                    "buffer_size": n * 4, "file": "arg_0.bin",
                },
                {
                    "index": 1, "name": "b", "type": "float*",
                    "size": 8, "offset": 8, "is_pointer": True,
                    "is_const": False, "is_output": True,
                    "buffer_size": n * 4, "file": "arg_1.bin",
                },
            ],
        }
        (capture / "metadata.json").write_text(json.dumps(meta))

        clean = np.ones(n, dtype=np.float32)
        clean.tofile(str(capture / "arg_0.bin"))
        clean.tofile(str(ref_out / "output_0.bin"))

        nan_data = np.ones(n, dtype=np.float32)
        nan_data[0] = np.nan
        nan_data.tofile(str(capture / "arg_1.bin"))
        np.ones(n, dtype=np.float32).tofile(str(ref_out / "output_1.bin"))

        result = _compare_outputs(str(repro), meta, atol=1e-6, rtol=1e-5, details=[])
        assert not result.passed
        import math
        assert math.isnan(result.max_error)


class TestInferDtype:
    """Tests for numpy dtype inference."""

    def test_float(self):
        assert _infer_numpy_dtype("float*") == np.float32

    def test_double(self):
        assert _infer_numpy_dtype("const double*") == np.float64

    def test_int(self):
        assert _infer_numpy_dtype("int*") == np.int32

    def test_half(self):
        assert _infer_numpy_dtype("__half*") == np.float16

    def test_unknown_defaults_to_float32(self):
        assert _infer_numpy_dtype("weird_type*") == np.float32


def _make_va_capture(tmp_path):
    """Create a minimal VA-faithful capture directory (dispatch.json format)."""
    repro = tmp_path / "repro"
    capture = repro / "capture"
    memory = capture / "memory"
    capture.mkdir(parents=True)
    memory.mkdir()

    (capture / "dispatch.json").write_text(json.dumps({
        "mangled_name": "_Z11test_kernelPfS_i",
        "demangled_name": "test_kernel",
        "grid": [256, 1, 1],
        "block": [256, 1, 1],
    }))
    (capture / "memory_regions.json").write_text(json.dumps({
        "regions": [{"base": 0x1000, "size": 4096}],
    }))

    data = np.arange(1024, dtype=np.uint8)
    data.tofile(str(memory / "region_1000.bin"))

    return repro, capture


def _mock_replay_success(stdout="", stderr=""):
    """Return a CompletedProcess mimicking a successful replay."""
    return MagicMock(
        returncode=0,
        stdout=stdout or "Average GPU time: 1.23 us\n",
        stderr=stderr,
    )


def _mock_replay_failure(stderr="segfault"):
    return MagicMock(returncode=1, stdout="", stderr=stderr)


class TestValidateReplayBaseline:
    """Baseline validation (no --hsaco) should be a smoke test."""

    @patch("kerncap._get_replay_path", return_value="/usr/bin/kerncap-replay")
    @patch("subprocess.run")
    def test_baseline_smoke_test_passes(self, mock_run, _mock_path, tmp_path):
        repro, _ = _make_va_capture(tmp_path)
        mock_run.return_value = _mock_replay_success()

        result = _validate_replay(str(repro), atol=1e-6, rtol=1e-5)
        assert result.passed
        assert any("smoke test" in d for d in result.details)

    @patch("kerncap._get_replay_path", return_value="/usr/bin/kerncap-replay")
    @patch("subprocess.run")
    def test_baseline_replay_failure(self, mock_run, _mock_path, tmp_path):
        repro, _ = _make_va_capture(tmp_path)
        mock_run.return_value = _mock_replay_failure()

        result = _validate_replay(str(repro), atol=1e-6, rtol=1e-5)
        assert not result.passed
        assert any("Replay failed" in d for d in result.details)


class TestValidateReplayVariant:
    """Variant validation (--hsaco) should compare two replay outputs."""

    @patch("kerncap._get_replay_path", return_value="/usr/bin/kerncap-replay")
    @patch("subprocess.run")
    def test_variant_identical_passes(self, mock_run, _mock_path, tmp_path):
        repro, capture = _make_va_capture(tmp_path)
        output_dir = capture / "output"
        region_data = np.arange(1024, dtype=np.uint8)

        call_count = [0]
        def side_effect(cmd, **kwargs):
            call_count[0] += 1
            output_dir.mkdir(exist_ok=True)
            region_data.tofile(str(output_dir / "region_1000.bin"))
            return _mock_replay_success()

        mock_run.side_effect = side_effect

        result = _validate_replay(
            str(repro), atol=1e-6, rtol=1e-5, hsaco="/tmp/variant.hsaco",
        )
        assert result.passed
        assert call_count[0] == 2
        assert any("identical" in d for d in result.details)

    @patch("kerncap._get_replay_path", return_value="/usr/bin/kerncap-replay")
    @patch("subprocess.run")
    def test_variant_different_fails(self, mock_run, _mock_path, tmp_path):
        repro, capture = _make_va_capture(tmp_path)
        output_dir = capture / "output"

        call_count = [0]
        def side_effect(cmd, **kwargs):
            call_count[0] += 1
            output_dir.mkdir(exist_ok=True)
            if call_count[0] == 1:
                np.zeros(1024, dtype=np.uint8).tofile(
                    str(output_dir / "region_1000.bin"))
            else:
                np.ones(1024, dtype=np.uint8).tofile(
                    str(output_dir / "region_1000.bin"))
            return _mock_replay_success()

        mock_run.side_effect = side_effect

        result = _validate_replay(
            str(repro), atol=1e-6, rtol=1e-5, hsaco="/tmp/variant.hsaco",
        )
        assert not result.passed
        assert any("FAIL" in d for d in result.details)
        assert any("bytes differ" in d for d in result.details)

    @patch("kerncap._get_replay_path", return_value="/usr/bin/kerncap-replay")
    @patch("subprocess.run")
    def test_variant_size_mismatch_fails(self, mock_run, _mock_path, tmp_path):
        repro, capture = _make_va_capture(tmp_path)
        output_dir = capture / "output"

        call_count = [0]
        def side_effect(cmd, **kwargs):
            call_count[0] += 1
            output_dir.mkdir(exist_ok=True)
            if call_count[0] == 1:
                np.zeros(1024, dtype=np.uint8).tofile(
                    str(output_dir / "region_1000.bin"))
            else:
                np.zeros(512, dtype=np.uint8).tofile(
                    str(output_dir / "region_1000.bin"))
            return _mock_replay_success()

        mock_run.side_effect = side_effect

        result = _validate_replay(
            str(repro), atol=1e-6, rtol=1e-5, hsaco="/tmp/variant.hsaco",
        )
        assert not result.passed
        assert any("SIZE MISMATCH" in d for d in result.details)

    @patch("kerncap._get_replay_path", return_value="/usr/bin/kerncap-replay")
    @patch("subprocess.run")
    def test_variant_baseline_replay_fails(self, mock_run, _mock_path, tmp_path):
        """If the baseline replay fails, variant validation should fail."""
        repro, _ = _make_va_capture(tmp_path)
        mock_run.return_value = _mock_replay_failure()

        result = _validate_replay(
            str(repro), atol=1e-6, rtol=1e-5, hsaco="/tmp/variant.hsaco",
        )
        assert not result.passed
        assert any("Replay failed" in d for d in result.details)


class TestCompareReplayOutputs:
    """Direct tests for _compare_replay_outputs."""

    def test_identical_regions(self, tmp_path):
        base_dir = tmp_path / "baseline"
        var_dir = tmp_path / "variant"
        base_dir.mkdir()
        var_dir.mkdir()

        data = np.arange(256, dtype=np.uint8)
        data.tofile(str(base_dir / "region_a000.bin"))
        data.tofile(str(var_dir / "region_a000.bin"))

        result = _compare_replay_outputs(str(base_dir), str(var_dir), [])
        assert result.passed

    def test_missing_region_in_variant(self, tmp_path):
        base_dir = tmp_path / "baseline"
        var_dir = tmp_path / "variant"
        base_dir.mkdir()
        var_dir.mkdir()

        np.zeros(256, dtype=np.uint8).tofile(str(base_dir / "region_a000.bin"))

        result = _compare_replay_outputs(str(base_dir), str(var_dir), [])
        assert not result.passed
        assert any("MISSING in variant" in d for d in result.details)

    def test_no_output_files(self, tmp_path):
        base_dir = tmp_path / "baseline"
        var_dir = tmp_path / "variant"
        base_dir.mkdir()
        var_dir.mkdir()

        result = _compare_replay_outputs(str(base_dir), str(var_dir), [])
        assert result.passed
