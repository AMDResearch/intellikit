"""Unit tests for kerncap.capturer — environment variable setup."""

import os
from unittest.mock import patch, MagicMock

import pytest

from kerncap.capturer import run_capture


class TestLdPreload:
    """Verify that run_capture sets LD_PRELOAD (not HSA_TOOLS_LIB)."""

    @patch("kerncap.capturer.subprocess.run")
    @patch("kerncap._get_lib_path", return_value="/fake/libkerncap.so")
    def test_sets_ld_preload(self, _mock_lib, mock_run, tmp_path):
        dispatch = tmp_path / "dispatch.json"
        dispatch.write_text("{}")

        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

        run_capture(
            kernel_name="kern",
            cmd=["./app"],
            output_dir=str(tmp_path),
        )

        env = mock_run.call_args.kwargs["env"]
        assert "LD_PRELOAD" in env
        assert env["LD_PRELOAD"] == "/fake/libkerncap.so"
        assert "HSA_TOOLS_LIB" not in env

    @patch("kerncap.capturer.subprocess.run")
    @patch("kerncap._get_lib_path", return_value="/fake/libkerncap.so")
    def test_prepends_to_existing_ld_preload(self, _mock_lib, mock_run, tmp_path):
        dispatch = tmp_path / "dispatch.json"
        dispatch.write_text("{}")

        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

        with patch.dict(os.environ, {"LD_PRELOAD": "/other/lib.so"}):
            run_capture(
                kernel_name="kern",
                cmd=["./app"],
                output_dir=str(tmp_path),
            )

        env = mock_run.call_args.kwargs["env"]
        assert env["LD_PRELOAD"] == "/fake/libkerncap.so:/other/lib.so"

    @patch("kerncap.capturer.subprocess.run")
    @patch("kerncap._get_lib_path", return_value="/fake/libkerncap.so")
    def test_strips_hsa_tools_lib(self, _mock_lib, mock_run, tmp_path):
        dispatch = tmp_path / "dispatch.json"
        dispatch.write_text("{}")

        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

        with patch.dict(os.environ, {"HSA_TOOLS_LIB": "/some/other_tool.so"}):
            run_capture(
                kernel_name="kern",
                cmd=["./app"],
                output_dir=str(tmp_path),
            )

        env = mock_run.call_args.kwargs["env"]
        assert "HSA_TOOLS_LIB" not in env
        assert "LD_PRELOAD" in env

    @patch("kerncap.capturer.subprocess.run")
    @patch("kerncap._get_lib_path", return_value="/fake/libkerncap.so")
    def test_sets_kerncap_env_vars(self, _mock_lib, mock_run, tmp_path):
        dispatch = tmp_path / "dispatch.json"
        dispatch.write_text("{}")

        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

        run_capture(
            kernel_name="my_kernel",
            cmd=["./app"],
            output_dir=str(tmp_path),
            dispatch=3,
        )

        env = mock_run.call_args.kwargs["env"]
        assert env["KERNCAP_KERNEL"] == "my_kernel"
        assert env["KERNCAP_OUTPUT"] == str(tmp_path)
        assert env["KERNCAP_DISPATCH"] == "3"

    @patch("kerncap.capturer.subprocess.run")
    @patch("kerncap._get_lib_path", return_value="/fake/libkerncap.so")
    def test_dispatch_not_set_when_negative(self, _mock_lib, mock_run, tmp_path):
        dispatch = tmp_path / "dispatch.json"
        dispatch.write_text("{}")

        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

        run_capture(
            kernel_name="kern",
            cmd=["./app"],
            output_dir=str(tmp_path),
            dispatch=-1,
        )

        env = mock_run.call_args.kwargs["env"]
        assert "KERNCAP_DISPATCH" not in env

    @patch("kerncap.capturer.subprocess.run")
    @patch("kerncap._get_lib_path", return_value="/fake/libkerncap.so")
    def test_sets_capture_child_env(self, _mock_lib, mock_run, tmp_path):
        dispatch = tmp_path / "dispatch.json"
        dispatch.write_text("{}")

        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

        run_capture(
            kernel_name="kern",
            cmd=["./app"],
            output_dir=str(tmp_path),
        )

        env = mock_run.call_args.kwargs["env"]
        assert env.get("KERNCAP_CAPTURE_CHILD") == "1"

    @patch("kerncap.capturer.subprocess.run")
    @patch("kerncap._get_lib_path", return_value="/fake/libkerncap.so")
    def test_triton_delegates_to_triton_capture(self, _mock_lib, mock_run, tmp_path):
        with patch("kerncap.triton_capture.run_triton_capture") as mock_triton:
            mock_triton.return_value = str(tmp_path)

            run_capture(
                kernel_name="kern",
                cmd=["python", "train.py"],
                output_dir=str(tmp_path),
                language="triton",
            )

            mock_triton.assert_called_once()
            mock_run.assert_not_called()
