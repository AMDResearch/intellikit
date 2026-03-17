"""Unit tests for the Kerncap Python API class."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from kerncap import Kerncap, ExtractResult, KernelStat, ReplayResult, ValidationResult


class TestKerncapProfile:
    """Tests for Kerncap.profile()."""

    @patch("kerncap.profiler.run_profile")
    def test_profile_delegates_to_run_profile(self, mock_run_profile):
        stats = [
            KernelStat(
                name="my_kernel",
                calls=10,
                total_duration_ns=5000000,
                avg_duration_ns=500000,
                percentage=80.0,
            ),
        ]
        mock_run_profile.return_value = stats

        kc = Kerncap()
        result = kc.profile(["./app", "--flag"])

        mock_run_profile.assert_called_once_with(
            ["./app", "--flag"],
            output_path=None,
        )
        assert result == stats
        assert result[0].name == "my_kernel"

    @patch("kerncap.profiler.run_profile")
    def test_profile_passes_output_path(self, mock_run_profile):
        mock_run_profile.return_value = []

        kc = Kerncap()
        kc.profile(["./app"], output_path="/tmp/profile.json")

        mock_run_profile.assert_called_once_with(
            ["./app"],
            output_path="/tmp/profile.json",
        )

    @patch("kerncap.profiler.run_profile")
    def test_profile_returns_empty_list(self, mock_run_profile):
        mock_run_profile.return_value = []

        kc = Kerncap()
        result = kc.profile(["./app"])
        assert result == []

    @patch("kerncap.profiler.run_profile")
    def test_profile_propagates_exceptions(self, mock_run_profile):
        mock_run_profile.side_effect = FileNotFoundError("rocprofv3 not found")

        kc = Kerncap()
        with pytest.raises(FileNotFoundError, match="rocprofv3"):
            kc.profile(["./app"])


class TestKerncapExtract:
    """Tests for Kerncap.extract()."""

    @patch("kerncap.extract.run_extract")
    def test_extract_delegates_to_run_extract(self, mock_run_extract):
        expected = ExtractResult(
            output_dir="./isolated/my_kernel",
            capture_dir="./isolated/my_kernel/capture",
            language="hip",
            has_source=True,
            generated_files=["capture/", "Makefile"],
        )
        mock_run_extract.return_value = expected

        kc = Kerncap()
        result = kc.extract(
            "my_kernel",
            cmd="./app --flag",
            source_dir="./src",
            output="./isolated/my_kernel",
        )

        mock_run_extract.assert_called_once_with(
            kernel_name="my_kernel",
            cmd="./app --flag",
            source_dir="./src",
            output="./isolated/my_kernel",
            language=None,
            dispatch=-1,
            defines=None,
            timeout=300,
        )
        assert result.output_dir == "./isolated/my_kernel"
        assert result.has_source is True

    @patch("kerncap.extract.run_extract")
    def test_extract_with_all_options(self, mock_run_extract):
        mock_run_extract.return_value = ExtractResult(
            output_dir="/tmp/out",
            capture_dir="/tmp/out/capture",
        )

        kc = Kerncap()
        kc.extract(
            "my_kernel",
            cmd=["./app", "--flag"],
            source_dir="./src",
            output="/tmp/out",
            language="triton",
            dispatch=2,
            defines=["-D FOO"],
            timeout=60,
        )

        mock_run_extract.assert_called_once_with(
            kernel_name="my_kernel",
            cmd=["./app", "--flag"],
            source_dir="./src",
            output="/tmp/out",
            language="triton",
            dispatch=2,
            defines=["-D FOO"],
            timeout=60,
        )


class TestKerncapValidate:
    """Tests for Kerncap.validate()."""

    @patch("kerncap.validator.validate_reproducer")
    def test_validate_delegates_to_validate_reproducer(self, mock_validate):
        expected = ValidationResult(
            passed=True,
            details=["Replay: OK", "Baseline replay OK (smoke test)"],
            max_error=0.0,
        )
        mock_validate.return_value = expected

        kc = Kerncap()
        result = kc.validate("./isolated/my_kernel")

        mock_validate.assert_called_once_with(
            reproducer_dir="./isolated/my_kernel",
            tolerance=1e-6,
            rtol=1e-5,
            hsaco=None,
        )
        assert result.passed is True

    @patch("kerncap.validator.validate_reproducer")
    def test_validate_with_hsaco(self, mock_validate):
        mock_validate.return_value = ValidationResult(
            passed=True,
            details=[],
            max_error=0.0,
        )

        kc = Kerncap()
        kc.validate(
            "./isolated/my_kernel",
            hsaco="optimized.hsaco",
            tolerance=1e-4,
            rtol=1e-3,
        )

        mock_validate.assert_called_once_with(
            reproducer_dir="./isolated/my_kernel",
            tolerance=1e-4,
            rtol=1e-3,
            hsaco="optimized.hsaco",
        )

    @patch("kerncap.validator.validate_reproducer")
    def test_validate_failure_result(self, mock_validate):
        mock_validate.return_value = ValidationResult(
            passed=False,
            details=["FAIL (max_error=1.5)"],
            max_error=1.5,
        )

        kc = Kerncap()
        result = kc.validate("./isolated/my_kernel")
        assert result.passed is False
        assert result.max_error == 1.5


class TestKerncapReplay:
    """Tests for Kerncap.replay()."""

    @patch("kerncap._get_replay_path")
    @patch("subprocess.run")
    def test_replay_basic(self, mock_subprocess, mock_replay_path, tmp_path):
        mock_replay_path.return_value = "/usr/bin/kerncap-replay"
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Kernel dispatched successfully\nAverage GPU time: 42.5 us\n",
            stderr="",
        )

        capture_dir = tmp_path / "repro" / "capture"
        capture_dir.mkdir(parents=True)

        kc = Kerncap()
        result = kc.replay(str(tmp_path / "repro"))

        assert result.returncode == 0
        assert result.timing_us == pytest.approx(42.5)
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert call_args[0] == "/usr/bin/kerncap-replay"

    @patch("kerncap._get_replay_path")
    @patch("subprocess.run")
    def test_replay_with_options(self, mock_subprocess, mock_replay_path, tmp_path):
        mock_replay_path.return_value = "/usr/bin/kerncap-replay"
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        )

        capture_dir = tmp_path / "repro" / "capture"
        capture_dir.mkdir(parents=True)

        kc = Kerncap()
        kc.replay(
            str(tmp_path / "repro"),
            hsaco="opt.hsaco",
            iterations=100,
            dump_output=True,
            hip_launch=True,
        )

        call_args = mock_subprocess.call_args[0][0]
        assert "--hsaco" in call_args
        assert "opt.hsaco" in call_args
        assert "--iterations" in call_args
        assert "100" in call_args
        assert "--dump-output" in call_args
        assert "--hip-launch" in call_args

    @patch("kerncap._get_replay_path")
    @patch("subprocess.run")
    def test_replay_no_timing(self, mock_subprocess, mock_replay_path, tmp_path):
        mock_replay_path.return_value = "/usr/bin/kerncap-replay"
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Kernel dispatched successfully\n",
            stderr="",
        )

        capture_dir = tmp_path / "repro" / "capture"
        capture_dir.mkdir(parents=True)

        kc = Kerncap()
        result = kc.replay(str(tmp_path / "repro"))
        assert result.timing_us is None

    @patch("kerncap._get_replay_path")
    @patch("subprocess.run")
    def test_replay_failure(self, mock_subprocess, mock_replay_path, tmp_path):
        mock_replay_path.return_value = "/usr/bin/kerncap-replay"
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="HSA error: invalid argument",
        )

        capture_dir = tmp_path / "repro" / "capture"
        capture_dir.mkdir(parents=True)

        kc = Kerncap()
        result = kc.replay(str(tmp_path / "repro"))
        assert result.returncode == 1
        assert "HSA error" in result.stderr


class TestReplayResult:
    """Tests for the ReplayResult dataclass."""

    def test_default_timing(self):
        r = ReplayResult(returncode=0, stdout="", stderr="")
        assert r.timing_us is None

    def test_with_timing(self):
        r = ReplayResult(returncode=0, stdout="", stderr="", timing_us=123.4)
        assert r.timing_us == 123.4
