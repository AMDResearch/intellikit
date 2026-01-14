#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Nexus: HSA Packet Source Code Extractor

A custom tool that intercepts Heterogeneous System Architecture (HSA) packets,
extracts the source code from them, and outputs it to a JSON file containing
the assembly and the HIP code.

This package provides Python utilities for using Nexus to analyze ROCm GPU kernels.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator

__version__ = "0.1.0"


def _find_nexus_lib() -> Optional[Path]:
    """Find the libnexus.so library in the package or build directory."""
    import site
    import sys

    possible_paths = [
        # Installed with package (regular install)
        Path(__file__).parent / "libnexus.so",
        # Development build
        Path(__file__).parent.parent / "build" / "lib" / "libnexus.so",
    ]

    # For editable installs, check user site-packages
    user_site = site.getusersitepackages()
    if user_site:
        possible_paths.append(Path(user_site) / "nexus" / "libnexus.so")

    # Also check all site-packages directories
    for site_dir in site.getsitepackages():
        possible_paths.append(Path(site_dir) / "nexus" / "libnexus.so")

    for path in possible_paths:
        if path.exists():
            return path.resolve()

    return None


class Kernel:
    """Represents a single GPU kernel with its assembly and source code."""

    def __init__(self, name: str, data: Dict[str, Any]):
        self.name = name
        self._data = data

    @property
    def assembly(self) -> List[str]:
        """Assembly instructions for this kernel."""
        return self._data.get("assembly", [])

    @property
    def hip(self) -> List[str]:
        """HIP source code lines."""
        return self._data.get("hip", [])

    @property
    def files(self) -> List[str]:
        """Source files referenced by this kernel."""
        return self._data.get("files", [])

    @property
    def lines(self) -> List[int]:
        """Line numbers corresponding to source code."""
        return self._data.get("lines", [])

    @property
    def signature(self) -> str:
        """Function signature of the kernel."""
        return self._data.get("signature", "")

    def __repr__(self) -> str:
        return f"Kernel(name={self.name!r}, assembly={len(self.assembly)} instructions)"


class Trace:
    """
    Container for kernel trace data from Nexus.

    Supports iteration and dict-like access to kernels.

    Example:
        >>> trace = nexus.run(["python", "app.py"])
        >>> for kernel in trace:
        ...     print(f"{kernel.name}: {len(kernel.assembly)} instructions")
        >>> vector_add = trace["vector_add"]
        >>> print(vector_add.assembly)
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        self._kernels = {name: Kernel(name, info) for name, info in data.get("kernels", {}).items()}

    def __iter__(self) -> Iterator[Kernel]:
        """Iterate over all kernels."""
        return iter(self._kernels.values())

    def __getitem__(self, kernel_name: str) -> Kernel:
        """Access kernel by name."""
        if kernel_name not in self._kernels:
            raise KeyError(f"Kernel '{kernel_name}' not found in trace")
        return self._kernels[kernel_name]

    def __len__(self) -> int:
        """Number of kernels in trace."""
        return len(self._kernels)

    def __contains__(self, kernel_name: str) -> bool:
        """Check if kernel exists in trace."""
        return kernel_name in self._kernels

    @property
    def kernels(self) -> List[Kernel]:
        """List of all kernels."""
        return list(self._kernels.values())

    @property
    def kernel_names(self) -> List[str]:
        """List of kernel names."""
        return list(self._kernels.keys())

    def save(self, filepath: str) -> None:
        """
        Save trace data to JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        with open(filepath, "w") as f:
            json.dump(self._data, f, indent=2)

    def __repr__(self) -> str:
        return f"Trace({len(self)} kernels)"


class Nexus:
    """
    Nexus HSA packet tracer for analyzing ROCm GPU kernels.

    Example:
        >>> nexus = Nexus(log_level=3)
        >>> trace = nexus.run(["python", "my_script.py"])
        >>> for kernel in trace:
        ...     print(f"{kernel.name}: {len(kernel.assembly)} instructions")
        >>> # Access specific kernel
        >>> vector_add = trace["vector_add"]
        >>> print(vector_add.hip)
    """

    def __init__(self, log_level: int = 1, extra_search_prefix: Optional[str] = None):
        """
        Initialize Nexus tracer.

        Args:
            log_level: Verbosity level (0=none, 1=info, 2=warning, 3=error, 4=detail)
            extra_search_prefix: Additional search directories for HIP files (colon-separated)
        """
        self.log_level = log_level
        self.extra_search_prefix = extra_search_prefix

        lib_path = _find_nexus_lib()
        if lib_path is None:
            raise RuntimeError(
                "Could not find libnexus.so. Please build the project first:\n"
                "  cmake -B build -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release\n"
                "  cmake --build build --parallel"
            )

        self._lib_path = lib_path

    def run(
        self,
        command: List[str],
        output: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> Trace:
        """
        Run a command with Nexus tracing and return kernel trace.

        Args:
            command: Command to run as a list of strings (e.g., ["python", "script.py"])
            output: Optional path to save JSON trace file (auto-generated if None)
            env: Additional environment variables
            cwd: Working directory for the command

        Returns:
            Trace object with kernel data

        Example:
            >>> nexus = Nexus(log_level=3)
            >>> trace = nexus.run(["python", "gpu_app.py"])
            >>> for kernel in trace:
            ...     print(f"{kernel.name}: {len(kernel.assembly)} asm")
            >>> # Save trace if needed
            >>> trace.save("my_trace.json")
        """
        # Generate output filename if not provided
        if output is None:
            import tempfile

            fd, output = tempfile.mkstemp(suffix=".json", prefix="nexus_trace_")
            os.close(fd)

        # Prepare environment
        run_env = os.environ.copy()
        run_env["HSA_TOOLS_LIB"] = str(self._lib_path)
        run_env["NEXUS_LOG_LEVEL"] = str(self.log_level)
        run_env["NEXUS_OUTPUT_FILE"] = output

        # Enable Triton line info for better source tracking
        run_env["TRITON_DISABLE_LINE_INFO"] = "0"

        if self.extra_search_prefix:
            run_env["NEXUS_EXTRA_SEARCH_PREFIX"] = self.extra_search_prefix

        if env:
            run_env.update(env)

        # Run the command
        result = subprocess.run(command, env=run_env, cwd=cwd, capture_output=True, text=True)

        # Check if command succeeded
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}:\n{result.stderr}")

        # Load and return the trace data
        try:
            with open(output, "r") as f:
                content = f.read().strip()
                if not content:
                    # Empty file - no kernels were executed
                    return Trace({})
                data = json.loads(content)
            return Trace(data)
        except FileNotFoundError:
            # File wasn't created - no kernels were executed
            return Trace({})
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse trace file '{output}': {e}")

    @staticmethod
    def load(trace_file: str) -> Trace:
        """
        Load kernel trace from a saved JSON file.

        Args:
            trace_file: Path to the JSON trace file

        Returns:
            Trace object with kernel data

        Example:
            >>> trace = Nexus.load("trace.json")
            >>> for kernel in trace:
            ...     print(kernel.name)
        """
        with open(trace_file, "r") as f:
            data = json.load(f)
        return Trace(data)


__all__ = [
    "Nexus",
    "Trace",
    "Kernel",
    "__version__",
]
