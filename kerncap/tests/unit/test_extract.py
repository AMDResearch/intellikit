"""Unit tests for kerncap.extract — the refactored extract pipeline."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from kerncap.extract import ExtractResult, run_extract, _generate_reproducer


class TestRunExtract:
    """Tests for the top-level run_extract function."""

    @patch("kerncap.extract._generate_reproducer")
    @patch("kerncap.extract.run_capture")
    @patch("kerncap.extract.detect_language")
    def test_basic_hip_extract(self, mock_detect, mock_capture, mock_generate):
        mock_detect.return_value = "hip"
        mock_generate.return_value = ExtractResult(
            output_dir="./isolated/my_kernel",
            capture_dir="./isolated/my_kernel/capture",
            language="hip",
            has_source=False,
        )

        result = run_extract(
            kernel_name="my_kernel",
            cmd="./app --flag",
            source_dir="./src",
        )

        mock_capture.assert_called_once()
        assert mock_capture.call_args.kwargs["kernel_name"] == "my_kernel"
        assert mock_capture.call_args.kwargs["cmd"] == ["./app", "--flag"]
        assert result.language == "hip"

    @patch("kerncap.extract._generate_reproducer")
    @patch("kerncap.extract.run_capture")
    def test_cmd_as_list(self, mock_capture, mock_generate):
        mock_generate.return_value = ExtractResult(
            output_dir="/tmp/out",
            capture_dir="/tmp/out/capture",
        )

        run_extract(
            kernel_name="kern",
            cmd=["./app", "--flag"],
        )

        assert mock_capture.call_args.kwargs["cmd"] == ["./app", "--flag"]

    @patch("kerncap.extract._generate_reproducer")
    @patch("kerncap.extract.run_capture")
    def test_default_output_dir(self, mock_capture, mock_generate):
        mock_generate.return_value = ExtractResult(
            output_dir="./isolated/kern",
            capture_dir="./isolated/kern/capture",
        )

        run_extract(kernel_name="kern", cmd=["./app"])

        capture_call = mock_capture.call_args.kwargs
        assert capture_call["output_dir"] == "./isolated/kern/capture"

    @patch("kerncap.extract._generate_reproducer")
    @patch("kerncap.extract.run_capture")
    def test_custom_output_dir(self, mock_capture, mock_generate):
        mock_generate.return_value = ExtractResult(
            output_dir="/tmp/custom",
            capture_dir="/tmp/custom/capture",
        )

        run_extract(kernel_name="kern", cmd=["./app"], output="/tmp/custom")

        assert mock_capture.call_args.kwargs["output_dir"] == "/tmp/custom/capture"

    @patch("kerncap.extract._generate_reproducer")
    @patch("kerncap.extract.run_capture")
    def test_dispatch_forwarded(self, mock_capture, mock_generate):
        mock_generate.return_value = ExtractResult(
            output_dir="/tmp/out",
            capture_dir="/tmp/out/capture",
        )

        run_extract(kernel_name="kern", cmd=["./app"], dispatch=5)

        assert mock_capture.call_args.kwargs["dispatch"] == 5

    @patch("kerncap.extract._generate_reproducer")
    @patch("kerncap.extract.run_capture")
    @patch("kerncap.extract.detect_language")
    def test_unknown_language_falls_back_to_none(
        self,
        mock_detect,
        mock_capture,
        mock_generate,
    ):
        mock_detect.return_value = "unknown"
        mock_generate.return_value = ExtractResult(
            output_dir="/tmp/out",
            capture_dir="/tmp/out/capture",
        )

        run_extract(kernel_name="kern", cmd=["./app"], source_dir="./src")

        assert mock_capture.call_args.kwargs["language"] is None

    @patch("kerncap.extract._generate_reproducer")
    @patch("kerncap.extract.run_capture")
    def test_explicit_language_skips_detection(self, mock_capture, mock_generate):
        mock_generate.return_value = ExtractResult(
            output_dir="/tmp/out",
            capture_dir="/tmp/out/capture",
        )

        run_extract(
            kernel_name="kern",
            cmd=["./app"],
            source_dir="./src",
            language="triton",
        )

        assert mock_capture.call_args.kwargs["language"] == "triton"

    @patch("kerncap.extract.run_capture")
    def test_capture_failure_propagates(self, mock_capture):
        mock_capture.side_effect = RuntimeError("Capture failed")

        with pytest.raises(RuntimeError, match="Capture failed"):
            run_extract(kernel_name="kern", cmd=["./app"])


class TestGenerateReproducer:
    """Tests for the _generate_reproducer routing logic."""

    def test_missing_metadata_raises(self, tmp_path):
        capture_dir = str(tmp_path / "capture")
        os.makedirs(capture_dir)

        with pytest.raises(FileNotFoundError, match="No dispatch.json"):
            _generate_reproducer(
                "kern",
                capture_dir,
                str(tmp_path),
                None,
                None,
                [],
            )

    @patch("kerncap.extract._generate_triton")
    def test_routes_to_triton_from_metadata(self, mock_triton, tmp_path):
        capture_dir = tmp_path / "capture"
        capture_dir.mkdir()
        (capture_dir / "dispatch.json").write_text(
            json.dumps({"language": "triton", "kernel_name": "kern"})
        )

        expected = ExtractResult(
            output_dir=str(tmp_path),
            capture_dir=str(capture_dir),
            language="triton",
        )
        mock_triton.return_value = expected

        result = _generate_reproducer(
            "kern",
            str(capture_dir),
            str(tmp_path),
            "./src",
            None,
            [],
        )

        mock_triton.assert_called_once()
        assert result.language == "triton"

    @patch("kerncap.extract._generate_hsaco")
    def test_routes_to_hsaco_by_default(self, mock_hsaco, tmp_path):
        capture_dir = tmp_path / "capture"
        capture_dir.mkdir()
        (capture_dir / "dispatch.json").write_text(json.dumps({"kernel_name": "kern"}))

        expected = ExtractResult(
            output_dir=str(tmp_path),
            capture_dir=str(capture_dir),
            language="hip",
        )
        mock_hsaco.return_value = expected

        result = _generate_reproducer(
            "kern",
            str(capture_dir),
            str(tmp_path),
            None,
            None,
            [],
        )

        mock_hsaco.assert_called_once()

    @patch("kerncap.extract._generate_triton")
    def test_explicit_language_overrides_metadata(self, mock_triton, tmp_path):
        capture_dir = tmp_path / "capture"
        capture_dir.mkdir()
        (capture_dir / "dispatch.json").write_text(
            json.dumps({"language": "hip", "kernel_name": "kern"})
        )

        mock_triton.return_value = ExtractResult(
            output_dir=str(tmp_path),
            capture_dir=str(capture_dir),
            language="triton",
        )

        _generate_reproducer(
            "kern",
            str(capture_dir),
            str(tmp_path),
            "./src",
            "triton",
            [],
        )

        mock_triton.assert_called_once()

    @patch("kerncap.extract._generate_hsaco")
    def test_reads_metadata_json_fallback(self, mock_hsaco, tmp_path):
        capture_dir = tmp_path / "capture"
        capture_dir.mkdir()
        (capture_dir / "metadata.json").write_text(json.dumps({"kernel_name": "kern"}))

        mock_hsaco.return_value = ExtractResult(
            output_dir=str(tmp_path),
            capture_dir=str(capture_dir),
        )

        _generate_reproducer(
            "kern",
            str(capture_dir),
            str(tmp_path),
            None,
            None,
            [],
        )

        mock_hsaco.assert_called_once()


class TestExtractResult:
    """Tests for the ExtractResult dataclass."""

    def test_defaults(self):
        r = ExtractResult(output_dir="/tmp/out", capture_dir="/tmp/out/capture")
        assert r.language is None
        assert r.has_source is False
        assert r.generated_files == []

    def test_with_values(self):
        r = ExtractResult(
            output_dir="/tmp/out",
            capture_dir="/tmp/out/capture",
            language="triton",
            has_source=True,
            generated_files=["reproducer.py", "capture/"],
        )
        assert r.language == "triton"
        assert r.has_source is True
        assert len(r.generated_files) == 2
