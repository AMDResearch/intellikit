# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Proboscis: Agent-Driven GPU Kernel Instrumentation

Intercepts GPU kernel launches, injects probe instrumentation using the
hidden-argument ABI trick, runs the target program, and returns structured
results. Works on compiled binaries — no source code modification needed.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

__version__ = "0.1.0"


def _find_proboscis_lib() -> Optional[Path]:
    """Find the libproboscis.so library."""
    import site

    possible_paths = [
        Path(__file__).parent / "libproboscis.so",
        Path(__file__).parent.parent / "build" / "lib" / "libproboscis.so",
    ]

    user_site = site.getusersitepackages()
    if user_site:
        possible_paths.append(Path(user_site) / "proboscis" / "libproboscis.so")

    for site_dir in site.getsitepackages():
        possible_paths.append(Path(site_dir) / "proboscis" / "libproboscis.so")

    for path in possible_paths:
        if path.exists():
            return path.resolve()

    return None


class ProbeResult:
    """Container for probe results from a single kernel."""

    def __init__(self, kernel_name: str, probe_type: str, data: Dict[str, Any]):
        self.kernel_name = kernel_name
        self.probe_type = probe_type
        self._data = data

    @property
    def records(self) -> List[Dict[str, Any]]:
        return self._data.get("records", [])

    @property
    def summary(self) -> Dict[str, Any]:
        return self._data.get("summary", {})

    @property
    def record_count(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return f"ProbeResult(kernel={self.kernel_name!r}, type={self.probe_type!r}, records={self.record_count})"


class InstrumentationResult:
    """Container for all instrumentation results from a run."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        self._results = {
            name: ProbeResult(name, data.get("probe_type", "unknown"), info)
            for name, info in data.get("kernels", {}).items()
        }

    @property
    def kernels(self) -> List[ProbeResult]:
        return list(self._results.values())

    @property
    def kernel_names(self) -> List[str]:
        return list(self._results.keys())

    @property
    def probe_type(self) -> str:
        return self._data.get("probe_type", "unknown")

    @property
    def command(self) -> List[str]:
        return self._data.get("command", [])

    def __getitem__(self, kernel_name: str) -> ProbeResult:
        if kernel_name not in self._results:
            raise KeyError(f"Kernel '{kernel_name}' not found in results")
        return self._results[kernel_name]

    def __iter__(self):
        return iter(self._results.values())

    def __len__(self) -> int:
        return len(self._results)

    def save(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self._data, f, indent=2)

    def __repr__(self) -> str:
        return f"InstrumentationResult({len(self)} kernels, probe={self.probe_type})"


class Proboscis:
    """
    Agent-driven GPU kernel instrumentation engine.

    Uses the hidden-argument ABI trick to inject probes into running GPU kernels
    without source code modification. The probe context pointer is added as a
    hidden kernel argument and the kernarg buffer is repacked at dispatch time.

    Example:
        >>> p = Proboscis()
        >>> result = p.instrument(["./vec_add"], "find memory accesses")
        >>> for kernel in result:
        ...     print(f"{kernel.kernel_name}: {kernel.record_count} records")
    """

    def __init__(self, log_level: int = 1):
        self.log_level = log_level
        self._lib_path = _find_proboscis_lib()

    def instrument(
        self,
        command: List[str],
        probe: str,
        target_kernel: Optional[str] = None,
        sample_rate: int = 1,
        max_records: int = 10000,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> InstrumentationResult:
        """
        Run a command with probe instrumentation and return results.

        Args:
            command: Command to run (e.g., ["./vec_add"])
            probe: Natural language probe description (e.g., "find memory accesses")
            target_kernel: Optional kernel name pattern to instrument (None = all)
            sample_rate: Sample 1-in-N dispatches (1 = every dispatch)
            max_records: Maximum probe records to collect
            env: Additional environment variables
            cwd: Working directory

        Returns:
            InstrumentationResult with per-kernel probe data
        """
        from .planner import plan_probe
        from .patcher import PatchConfig

        probe_spec = plan_probe(probe, target_kernel, sample_rate, max_records)

        # Write probe config for the C++ runtime
        config = {
            "probe_spec": probe_spec.to_dict(),
            "log_level": self.log_level,
        }

        fd, config_path = tempfile.mkstemp(suffix=".json", prefix="proboscis_config_")
        os.close(fd)

        fd, results_path = tempfile.mkstemp(suffix=".json", prefix="proboscis_results_")
        os.close(fd)

        try:
            with open(config_path, "w") as f:
                json.dump(config, f)

            run_env = os.environ.copy()
            if self._lib_path:
                run_env["HSA_TOOLS_LIB"] = str(self._lib_path)
            run_env["PROBOSCIS_CONFIG"] = config_path
            run_env["PROBOSCIS_RESULTS"] = results_path
            run_env["PROBOSCIS_LOG_LEVEL"] = str(self.log_level)

            if env:
                run_env.update(env)

            result = subprocess.run(command, env=run_env, cwd=cwd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(
                    f"Command failed with exit code {result.returncode}:\n{result.stderr}"
                )

            try:
                with open(results_path, "r") as f:
                    content = f.read().strip()
                    if not content:
                        return InstrumentationResult({"probe_type": probe_spec.probe_type, "command": command})
                    data = json.loads(content)
                data["command"] = command
                data["probe_type"] = probe_spec.probe_type
                return InstrumentationResult(data)
            except (FileNotFoundError, json.JSONDecodeError):
                return InstrumentationResult({"probe_type": probe_spec.probe_type, "command": command})

        finally:
            for path in (config_path, results_path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    @staticmethod
    def load(results_file: str) -> InstrumentationResult:
        """Load results from a saved JSON file."""
        with open(results_file, "r") as f:
            data = json.load(f)
        return InstrumentationResult(data)


__all__ = [
    "Proboscis",
    "InstrumentationResult",
    "ProbeResult",
    "__version__",
]
