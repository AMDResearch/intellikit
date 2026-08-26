# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the UProfProfiler wrapper around the AMD uProf CLI.

AMD uProf is never invoked: ``subprocess.run`` is patched and fed the real
``subprocess.CompletedProcess``/``CalledProcessError``/``TimeoutExpired``
objects the wrapper claims to handle, so a change in how those are consumed
breaks these tests instead of silently passing.  Every path out of
``find_hotspots`` is covered -- success, a missing executable, a missing
uProf CLI, a timeout, a non-zero exit, and both of the "uProf printed
something we could not parse" ``RuntimeError`` paths.

All paths come from ``tmp_path`` so the tests do not depend on the working
directory pytest happens to be invoked from, and an autouse fixture clears
``INTELLIKIT_UPROF_CLI`` so a developer machine that happens to have uProf
configured cannot change the outcome.
"""

import logging
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from uprof_mcp.uprof_profiler import UProfProfiler, UProfProfilerResult

SAMPLE_STDOUT = """AMDuProfCLI: collection started
Generated data files path: /scratch/AMDuProf-app-datafiles
Generated report file: /scratch/AMDuProf-app.csv
"""


@pytest.fixture(autouse=True)
def _clean_env():  # noqa: ANN202
    """Run each test with INTELLIKIT_UPROF_CLI unset unless it sets it."""
    with patch.dict("os.environ"):
        os.environ.pop("INTELLIKIT_UPROF_CLI", None)
        yield


def _executable(tmp_path: Path) -> Path:
    """Create a real file to stand in for the profiled executable."""
    exe = tmp_path / "app"
    exe.write_text("#!/bin/sh\nexit 0\n")
    return exe


def _completed(stdout: str = SAMPLE_STDOUT) -> subprocess.CompletedProcess:
    """Build a real CompletedProcess as subprocess.run would return."""
    return subprocess.CompletedProcess(args=["AMDuProfCLI"], returncode=0, stdout=stdout, stderr="")


class TestInit:
    """Cover how the profiler decides which uProf CLI to run."""

    def test_default_executable_path(self) -> None:
        """With no argument and no environment override the default is used."""
        assert UProfProfiler().profiler_exe == UProfProfiler.DEFAULT_EXE_PATH

    def test_explicit_uprof_argument(self) -> None:
        """An explicit uprof path wins over the default."""
        profiler = UProfProfiler(uprof="/opt/AMDuProf_6.0/bin/AMDuProfCLI")
        assert profiler.profiler_exe == "/opt/AMDuProf_6.0/bin/AMDuProfCLI"

    def test_explicit_uprof_pathlike_is_stringified(self, tmp_path: Path) -> None:
        """A PathLike uprof argument is stored as a string."""
        profiler = UProfProfiler(uprof=tmp_path / "AMDuProfCLI")
        assert profiler.profiler_exe == str(tmp_path / "AMDuProfCLI")

    def test_environment_variable_override(self) -> None:
        """INTELLIKIT_UPROF_CLI overrides the default path."""
        env_path = "/custom/path/to/AMDuProfCLI"
        with patch.dict("os.environ", {"INTELLIKIT_UPROF_CLI": env_path}):
            assert UProfProfiler().profiler_exe == env_path

    def test_default_logger_is_created(self) -> None:
        """Omitting the logger yields one named after the class."""
        assert UProfProfiler().logger.name == "UProfProfiler"

    def test_supplied_logger_is_kept(self) -> None:
        """A supplied logger is used as-is."""
        logger = logging.getLogger("test-uprof")
        assert UProfProfiler(logger=logger).logger is logger


class TestFindHotspots:
    """Cover every path out of find_hotspots."""

    def test_success_returns_parsed_paths(self, tmp_path: Path) -> None:
        """Both paths are lifted out of the uProf stdout into a real result."""
        profiler = UProfProfiler()
        with patch("subprocess.run", return_value=_completed()) as run:
            result = profiler.find_hotspots(tmp_path / "out", _executable(tmp_path), ["--fast"])
        assert isinstance(result, UProfProfilerResult)
        assert result.results_path == Path("/scratch/AMDuProf-app-datafiles")
        assert result.report_path == Path("/scratch/AMDuProf-app.csv")
        assert run.call_count == 1

    def test_command_line_is_assembled(self, tmp_path: Path) -> None:
        """The uProf command names the config, both output paths and the args."""
        out_dir = tmp_path / "out"
        exe = _executable(tmp_path)
        profiler = UProfProfiler(uprof="/opt/uprof/AMDuProfCLI")
        with patch("subprocess.run", return_value=_completed()) as run:
            profiler.find_hotspots(out_dir, exe, ["--size", "1024"])
        assert run.call_args.args[0] == [
            "/opt/uprof/AMDuProfCLI",
            "profile",
            "--config",
            "hotspots",
            "--output-dir",
            str(out_dir),
            "--log-path",
            str(out_dir),
            str(exe),
            "--size",
            "1024",
        ]

    def test_subprocess_options(self, tmp_path: Path) -> None:
        """Output is captured, decoded, checked, and bounded by the timeout."""
        profiler = UProfProfiler()
        with patch("subprocess.run", return_value=_completed()) as run:
            profiler.find_hotspots(tmp_path / "out", _executable(tmp_path), None)
        assert run.call_args.kwargs == {
            "timeout": UProfProfiler.DEFAULT_TIMEOUT,
            "capture_output": True,
            "check": True,
            "text": True,
        }

    def test_none_arguments_append_nothing(self, tmp_path: Path) -> None:
        """executable_args=None runs the executable with no trailing arguments."""
        exe = _executable(tmp_path)
        profiler = UProfProfiler()
        with patch("subprocess.run", return_value=_completed()) as run:
            profiler.find_hotspots(tmp_path / "out", exe, None)
        assert run.call_args.args[0][-1] == str(exe)

    def test_output_directory_is_created(self, tmp_path: Path) -> None:
        """A missing output directory, including parents, is created."""
        out_dir = tmp_path / "nested" / "results"
        profiler = UProfProfiler()
        with patch("subprocess.run", return_value=_completed()):
            profiler.find_hotspots(out_dir, _executable(tmp_path), None)
        assert out_dir.is_dir()

    def test_existing_output_directory_is_reused(self, tmp_path: Path) -> None:
        """An output directory that already exists is not an error."""
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        profiler = UProfProfiler()
        with patch("subprocess.run", return_value=_completed()):
            profiler.find_hotspots(out_dir, _executable(tmp_path), None)
        assert out_dir.is_dir()

    def test_parsed_paths_are_stripped(self, tmp_path: Path) -> None:
        """Trailing whitespace on the uProf output lines is removed."""
        stdout = (
            "Generated data files path: /scratch/data   \r\n"
            "Generated report file: /scratch/r.csv \n"
        )
        profiler = UProfProfiler()
        with patch("subprocess.run", return_value=_completed(stdout)):
            result = profiler.find_hotspots(tmp_path / "out", _executable(tmp_path), None)
        assert result.results_path == Path("/scratch/data")
        assert result.report_path == Path("/scratch/r.csv")

    def test_missing_executable(self, tmp_path: Path) -> None:
        """A non-existent executable fails before uProf is ever launched."""
        profiler = UProfProfiler()
        with patch("subprocess.run") as run:
            with pytest.raises(FileNotFoundError, match=r"not found\."):
                profiler.find_hotspots(tmp_path / "out", tmp_path / "missing", None)
            assert run.call_count == 0

    def test_directory_is_not_an_executable(self, tmp_path: Path) -> None:
        """A directory passed as the executable is rejected the same way."""
        a_dir = tmp_path / "a_dir"
        a_dir.mkdir()
        profiler = UProfProfiler()
        with pytest.raises(FileNotFoundError, match=r"not found\."):
            profiler.find_hotspots(tmp_path / "out", a_dir, None)

    def test_missing_uprof_cli(self, tmp_path: Path) -> None:
        """A missing uProf CLI is re-raised naming the path that was tried."""
        profiler = UProfProfiler(uprof="/nowhere/AMDuProfCLI")
        with (
            patch("subprocess.run", side_effect=FileNotFoundError(2, "No such file")),
            pytest.raises(FileNotFoundError, match="uprof executable not found at"),
        ):
            profiler.find_hotspots(tmp_path / "out", _executable(tmp_path), None)

    def test_timeout_propagates(self, tmp_path: Path) -> None:
        """A profiling run that overruns the timeout propagates TimeoutExpired."""
        timeout = subprocess.TimeoutExpired(cmd=["AMDuProfCLI"], timeout=180)
        profiler = UProfProfiler()
        with (
            patch("subprocess.run", side_effect=timeout),
            pytest.raises(subprocess.TimeoutExpired),
        ):
            profiler.find_hotspots(tmp_path / "out", _executable(tmp_path), None)

    def test_non_zero_exit_propagates(self, tmp_path: Path) -> None:
        """A non-zero uProf exit propagates CalledProcessError."""
        failure = subprocess.CalledProcessError(1, ["AMDuProfCLI"], stderr="profiling failed")
        profiler = UProfProfiler()
        with (
            patch("subprocess.run", side_effect=failure),
            pytest.raises(subprocess.CalledProcessError),
        ):
            profiler.find_hotspots(tmp_path / "out", _executable(tmp_path), None)

    def test_missing_results_path_in_output(self, tmp_path: Path) -> None:
        """Output without the data files line is a RuntimeError, not a crash."""
        stdout = "Generated report file: /scratch/AMDuProf-app.csv\n"
        profiler = UProfProfiler()
        with (
            patch("subprocess.run", return_value=_completed(stdout)),
            pytest.raises(RuntimeError, match="Profiling results path not found"),
        ):
            profiler.find_hotspots(tmp_path / "out", _executable(tmp_path), None)

    def test_missing_report_path_in_output(self, tmp_path: Path) -> None:
        """Output without the report line is a RuntimeError, not a crash."""
        stdout = "Generated data files path: /scratch/AMDuProf-app-datafiles\n"
        profiler = UProfProfiler()
        with (
            patch("subprocess.run", return_value=_completed(stdout)),
            pytest.raises(RuntimeError, match="Profiling report not found"),
        ):
            profiler.find_hotspots(tmp_path / "out", _executable(tmp_path), None)

    def test_empty_output(self, tmp_path: Path) -> None:
        """Completely empty uProf output trips the results path check first."""
        profiler = UProfProfiler()
        with (
            patch("subprocess.run", return_value=_completed("")),
            pytest.raises(RuntimeError, match="Profiling results path not found"),
        ):
            profiler.find_hotspots(tmp_path / "out", _executable(tmp_path), None)
