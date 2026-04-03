"""kerncap — Kernel extraction and isolation tool for HIP and Triton on AMD GPUs."""

from importlib.metadata import PackageNotFoundError, version as _get_version

try:
    __version__ = _get_version("kerncap")
except PackageNotFoundError:
    __version__ = "0.1.0"

import logging
import os
import pathlib
import re
import shutil
import site
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from kerncap.extract import ExtractResult
from kerncap.profiler import KernelStat
from kerncap.validator import ValidationResult

__all__ = [
    "Kerncap",
    "ExtractResult",
    "KernelStat",
    "ReplayResult",
    "ValidationResult",
]

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    """Result of replaying a captured kernel."""

    returncode: int
    stdout: str
    stderr: str
    timing_us: Optional[float] = None


class Kerncap:
    """Python API for kerncap kernel extraction and isolation.

    Example
    -------
    >>> from kerncap import Kerncap
    >>> kc = Kerncap()
    >>> profile = kc.profile(["./my_app", "--args"])
    >>> for kernel in profile:
    ...     print(f"{kernel.name}: {kernel.total_duration_ns / 1e6:.1f} ms")
    >>> result = kc.extract("kernel_name", cmd="./my_app --args", source_dir="./src")
    >>> validation = kc.validate(result.output_dir)
    >>> print(validation.passed)
    """

    def profile(
        self,
        cmd: list[str],
        output_path: Optional[str] = None,
    ) -> List[KernelStat]:
        """Profile an application and rank kernels by execution time.

        Parameters
        ----------
        cmd : list[str]
            Application command to profile.
        output_path : str, optional
            Path to write the profile JSON summary.

        Returns
        -------
        list[KernelStat]
            Kernel statistics sorted by total duration (descending).
        """
        from kerncap.profiler import run_profile

        return run_profile(cmd, output_path=output_path)

    def extract(
        self,
        kernel_name: str,
        cmd: str | list[str],
        source_dir: Optional[str] = None,
        output: Optional[str] = None,
        language: Optional[str] = None,
        dispatch: int = -1,
        defines: Optional[List[str]] = None,
        timeout: int = 300,
    ) -> ExtractResult:
        """Extract a kernel into a standalone reproducer.

        Runs the full pipeline: capture -> find source -> generate reproducer.

        Parameters
        ----------
        kernel_name : str
            Kernel name (or substring) to capture.
        cmd : str or list[str]
            Application command to run for capture.
        source_dir : str, optional
            Source directory to search for kernel source.
        output : str, optional
            Output directory for reproducer.
        language : str, optional
            Kernel language ("hip" or "triton"). Auto-detected if omitted.
        dispatch : int
            Dispatch index to capture (-1 = first match).
        defines : list[str], optional
            Extra preprocessor defines for reproducer.
        timeout : int
            Maximum seconds to wait for the application.

        Returns
        -------
        ExtractResult
        """
        from kerncap.extract import run_extract

        return run_extract(
            kernel_name=kernel_name,
            cmd=cmd,
            source_dir=source_dir,
            output=output,
            language=language,
            dispatch=dispatch,
            defines=defines,
            timeout=timeout,
        )

    def validate(
        self,
        reproducer_dir: str,
        tolerance: float = 1e-6,
        rtol: float = 1e-5,
        hsaco: Optional[str] = None,
    ) -> ValidationResult:
        """Validate a reproducer by comparing outputs to captured reference.

        Parameters
        ----------
        reproducer_dir : str
            Path to the reproducer project directory.
        tolerance : float
            Absolute tolerance for np.allclose.
        rtol : float
            Relative tolerance for np.allclose.
        hsaco : str, optional
            Path to an alternative HSACO to validate against the captured baseline.

        Returns
        -------
        ValidationResult
        """
        from kerncap.validator import validate_reproducer

        return validate_reproducer(
            reproducer_dir=reproducer_dir,
            tolerance=tolerance,
            rtol=rtol,
            hsaco=hsaco,
        )

    def replay(
        self,
        reproducer_dir: str,
        hsaco: Optional[str] = None,
        iterations: int = 1,
        dump_output: bool = False,
        hip_launch: bool = False,
    ) -> ReplayResult:
        """Replay a captured kernel using VA-faithful HSA dispatch.

        Parameters
        ----------
        reproducer_dir : str
            Path to the reproducer project (containing capture/).
        hsaco : str, optional
            Override HSACO file (use a recompiled variant).
        iterations : int
            Number of kernel iterations.
        dump_output : bool
            Dump post-execution memory regions.
        hip_launch : bool
            Use HIP runtime for kernel launch.

        Returns
        -------
        ReplayResult
        """
        capture_dir = os.path.join(reproducer_dir, "capture")
        if not os.path.isdir(capture_dir):
            capture_dir = reproducer_dir

        replay_bin = _get_replay_path()

        cmd = [replay_bin, capture_dir]
        if hsaco:
            cmd.extend(["--hsaco", hsaco])
        if iterations > 1:
            cmd.extend(["--iterations", str(iterations)])
        if dump_output:
            cmd.append("--dump-output")
        if hip_launch:
            cmd.append("--hip-launch")

        proc = subprocess.run(cmd, capture_output=True, text=True)

        timing_us = None
        match = re.search(r"Average GPU time:\s*([\d.]+)\s*us", proc.stdout)
        if match:
            timing_us = float(match.group(1))

        return ReplayResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            timing_us=timing_us,
        )


def _get_replay_path() -> str:
    """Return the absolute path to the installed kerncap-replay binary.

    Search order mirrors _get_lib_path(): package bin/ dir, site-packages,
    then PATH.
    """
    bin_name = "kerncap-replay"
    pkg_dir = pathlib.Path(__file__).resolve().parent

    candidates = [
        pkg_dir / "bin" / bin_name,
        pkg_dir.parent / "bin" / bin_name,
    ]

    sp_dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    user_sp = getattr(site, "getusersitepackages", lambda: None)()
    if user_sp:
        sp_dirs.append(user_sp)

    for sp in sp_dirs:
        candidates.append(pathlib.Path(sp) / "kerncap" / "bin" / bin_name)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    found = shutil.which(bin_name)
    if found:
        return found

    searched = [str(c) for c in candidates]
    raise FileNotFoundError(
        f"Could not locate {bin_name}. "
        f"Ensure the package was built correctly (pip install .) "
        f"or ensure kerncap-replay is on PATH.\n"
        f"Searched: {searched}"
    )


def _get_lib_path() -> str:
    """Return the absolute path to the installed libkerncap.so.

    The shared library is installed alongside the Python package by
    scikit-build-core.  We search in order:
      1. KERNCAP_LIB_PATH environment variable (explicit override)
      2. Relative to this file (works when importing the installed package)
      3. Installed site-packages (works when the local source tree shadows
         the installed package, e.g. running tests from the repo root)
    """
    env_path = os.environ.get("KERNCAP_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    lib_name = "libkerncap.so"
    pkg_dir = pathlib.Path(__file__).resolve().parent

    candidates = [
        pkg_dir / "lib" / lib_name,
        pkg_dir / lib_name,
    ]

    sp_dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    user_sp = getattr(site, "getusersitepackages", lambda: None)()
    if user_sp:
        sp_dirs.append(user_sp)

    for sp in sp_dirs:
        candidates.append(pathlib.Path(sp) / "kerncap" / "lib" / lib_name)
        candidates.append(pathlib.Path(sp) / "kerncap" / lib_name)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    searched = [str(c) for c in candidates]
    raise FileNotFoundError(
        f"Could not locate {lib_name}. "
        f"Ensure the package was built correctly (pip install .) "
        f"or set KERNCAP_LIB_PATH to the library location.\n"
        f"Searched: {searched}"
    )
