# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from os import PathLike
from pathlib import Path


@dataclass(frozen=True)
class ProfilerResult:
    """Class for the results of a profiling session.

    Attributes:
        results_path (Path): Path to the profiling results directory.
        report_path (Path): Path to the generated profiling report file.
    """

    results_path: Path
    report_path: Path


class UProfProfiler:
    """Profiler based on AMD uProf for profiling x86 executables.

    This class provides a Python interface to AMD uProf, a performance analysis
    tool-suite for x86-based applications. It supports profiling applications to
    identify performance hotspots and generate detailed reports.

    Example:
        Basic profiling to find hotspots::

            from omnikit import UProfProfiler
            from pathlib import Path

            # Initialize profiler
            profiler = UProfProfiler()

            # Profile an application
            result = profiler.find_hotspots(
                output_dir="./profiling_results",
                executable="./my_app",
                executable_args=["arg1", "arg2"],
            )

            print(f"Profiling results: {result.results_path}")
            print(f"Report file: {result.report_path}")

            # Read and display the report
            with open(result.report_path) as f:
                print(f.read())

        Profiling without arguments::

            # Profile an executable that doesn't need arguments
            result = profiler.find_hotspots(
                output_dir="./profiling_results", executable="./my_app", executable_args=None
            )

        Using a custom uProf CLI path::

            # Specify custom uProf path directly
            profiler = UProfProfiler(uprof="/opt/AMDuProf_6.0/bin/AMDuProfCLI")

            # Or use environment variable
            import os

            os.environ["OMNIKIT_UPROF_CLI"] = "/opt/AMDuProf_6.0/bin/AMDuProfCLI"
            profiler = UProfProfiler()

        Handling profiling errors::

            try:
                result = profiler.find_hotspots(
                    output_dir="./profiling_results", executable="./my_app", executable_args=None
                )
            except FileNotFoundError as e:
                print(f"Executable or uProf not found: {e}")
            except subprocess.TimeoutExpired:
                print("Profiling timed out")
            except subprocess.CalledProcessError as e:
                print(f"Profiling failed: {e}")

    Attributes:
        DEFAULT_EXE_PATH (str): Default executable path for uProf CLI.
        DEFAULT_TIMEOUT (int): Default timeout for profiling operations in seconds.
        logger (logging.Logger): Logger instance for logging messages.
        profiler_exe (str): Path to the profiler executable.
    """

    DEFAULT_EXE_PATH: str = "/opt/AMDuProf_5.1-701/bin/AMDuProfCLI"
    DEFAULT_TIMEOUT: int = 180  # seconds

    logger: logging.Logger
    profiler_exe: str

    def __init__(self, logger: logging.Logger | None, uprof: str | PathLike | None = None) -> None:
        """Initializes the UProfProfiler instance.

        Args:
            logger (logging.Logger): Logger instance for logging. If None, a default logger is
                created.
            uprof (str | PathLike | None): Optional path to the uProf executable. If None, uses the
                environment variable 'OMNIKIT_UPROF_CLI' or the default path DEFAULT_EXE_PATH.
        """
        self.logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self.profiler_exe = os.getenv(
            "OMNIKIT_UPROF_CLI", str(uprof) if uprof is not None else UProfProfiler.DEFAULT_EXE_PATH
        )
        self.logger.info("Initialized profiler.")

    def find_hotspots(
        self,
        output_dir: str | PathLike,
        executable: str | PathLike,
        executable_args: list[str] | None,
    ) -> ProfilerResult:
        """Finds hotspots via profiling.

        Parameters:
            output_dir (str | PathLike): Directory to save profiling results.
            executable (str | PathLike): Path to the executable to profile.
            executable_args (list[str] | None): Arguments to pass to the executable.

        Returns:
            UProfProfileResult: Object containing paths to results and report.

        Raises:
            FileNotFoundError: If the executable or uProf CLI is not found.
            subprocess.TimeoutExpired: If the profiling process times out.
            subprocess.CalledProcessError: If the profiling process fails.
            RuntimeError: If results or report paths cannot be found in the output.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        executable = Path(executable)
        if not executable.is_file():
            msg = f"Executable '{executable}' not found."
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        if executable_args is None:
            executable_args = []

        cmd = [
            self.profiler_exe,
            "profile",
            "--config",
            "hotspots",
            "--output-dir",
            str(output_dir),
            "--log-path",
            str(output_dir),
            str(executable),
            *executable_args,
        ]

        try:
            result = subprocess.run(
                cmd,
                timeout=UProfProfiler.DEFAULT_TIMEOUT,
                capture_output=True,
                check=True,
                text=True,
            )
            self.logger.info("Profiling %s successful.\nOutput:\n%s", cmd, result.stdout)
        except FileNotFoundError as e:
            msg = f"uprof executable not found at '{self.profiler_exe}'"
            self.logger.exception(msg)
            raise FileNotFoundError(msg) from e
        except subprocess.TimeoutExpired as e:
            self.logger.exception("Profiling %s timed out: %s", cmd, str(e))
            raise
        except subprocess.CalledProcessError as e:
            self.logger.exception("Profiling %s failed: %s", cmd, str(e))
            raise

        results_path = re.search("Generated data files path: (.*)", result.stdout)
        if not results_path:
            msg = f"Profiling results path not found for {cmd}."
            raise RuntimeError(msg)
        results_path = results_path.group(1).strip()

        report_path = re.search("Generated report file: (.*)", result.stdout)
        if not report_path:
            msg = f"Profiling report not found for {cmd}."
            raise RuntimeError(msg)
        report_path = report_path.group(1).strip()

        self.logger.info(
            "Profiling %s complete. Results in '%s', report in '%s'", cmd, results_path, report_path
        )
        return ProfilerResult(Path(results_path), Path(report_path))
