# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
import subprocess
from dataclasses import dataclass
from os import PathLike
from pathlib import Path


@dataclass(frozen=True)
class HipCompilerResult:
    """Result of code compilation via HipCompiler.

    Attributes:
        success (bool): Whether the compilation was successful.
        errors (str | None): Compilation error messages, if any.
        raw_output (str | None): Raw output from the compiler.
    """

    success: bool
    errors: str | None
    raw_output: str | None


class HipCompiler:
    """Class to handle compilation using ROCm 'hipcc'.

    This class provides a Python interface to compile HIP (Heterogeneous-computing Interface for
    Portability) source code using the hipcc compiler. It supports various compilation options
    including include directories, library paths, libraries to link against, and custom compiler
    flags.

    Example:
        Basic compilation::

            from omnikit import HipCompiler

            # Initialize compiler
            compiler = HipCompiler()

            # Compile a HIP source file
            result = compiler.compile(source_file="my_kernel.hip", output_file="my_kernel.out")

            if result.success:
                print("Compilation successful!")
            else:
                print(f"Compilation failed: {result.errors}")

        Compilation with additional options::

            # Specify include directories and libraries
            result = compiler.compile(
                source_file="my_kernel.hip",
                output_file="my_kernel.out",
                include_dirs=["/usr/local/include", "./include"],
                library_dirs=["/usr/local/lib"],
                libraries=["m", "pthread"],
                extra_flags=["-O3", "-DNDEBUG", "-Wall"],
            )

        Using a custom hipcc path::

            # Specify custom hipcc path directly
            compiler = HipCompiler(hipcc="/opt/rocm-6.0/bin/hipcc")

            # Or use environment variable
            import os

            os.environ["OMNIKIT_HIPCC"] = "/opt/rocm-6.0/bin/hipcc"
            compiler = HipCompiler()

    Attributes:
        logger (logging.Logger): Logger for logging compilation messages.
        hipcc_exe (str): Path to the `hipcc` compiler executable.
    """

    logger: logging.Logger
    hipcc_exe: str

    def __init__(
        self,
        logger: logging.Logger | None = None,
        hipcc: str | PathLike | None = None,
    ) -> None:
        """Initialize the HipCompiler.

        Args:
            logger (logging.Logger | None): Logger for logging messages. If None, a default
                logger is created.
            hipcc (str | PathLike | None): Path to the `hipcc` compiler
                executable. If None, uses the `OMNIKIT_HIPCC` environment variable
                or defaults to 'hipcc' in the system PATH.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.hipcc_exe = os.getenv("OMNIKIT_HIPCC", str(hipcc) if hipcc is not None else "hipcc")
        self.logger.info("Initialized compiler.")

    def compile(
        self,
        source_file: str | PathLike,
        output_file: str | PathLike,
        include_dirs: list[str | PathLike] | None = None,
        library_dirs: list[str | PathLike] | None = None,
        libraries: list[str] | None = None,
        extra_flags: list[str] | None = None,
    ) -> HipCompilerResult:
        """Compile a HIP source file into an executable.

        Args:
            source_file (str | PathLike): Path to the HIP source file.
            output_file (str | PathLike): Path to the output executable file.
            include_dirs (list[str | PathLike] | None): List of include directories.
            library_dirs (list[str | PathLike] | None): List of library directories.
            libraries (list[str] | None): List of libraries to link against.
            extra_flags (list[str] | None): Additional flags to pass to the compiler.

        Returns:
            HipCompilerResult: Result of the compilation process.

        Raises:
            ValueError: If source_file or output_file is not provided.
            FileNotFoundError: If the source_file does not exist or hipcc is not found
                at the specified path.
            subprocess.CalledProcessError: If the compilation process fails.
            RuntimeError: If the compilation process fails.
        """
        if source_file is None:
            msg = "Source file must be provided."
            raise ValueError(msg)
        source_file = Path(source_file)

        if output_file is None:
            msg = "Output file must be provided."
            raise ValueError(msg)
        output_file = Path(output_file)

        if not source_file.exists():
            msg = f"Source file {source_file} does not exist."
            raise FileNotFoundError(msg)

        if output_file.exists():
            self.logger.warning(
                "Output file %s already exists and will be overwritten.", str(output_file)
            )

        flags = [] if extra_flags is None else extra_flags
        includedirs = [] if include_dirs is None else [f"-I{e}" for e in include_dirs]
        libdirs = [] if library_dirs is None else [f"-L{e}" for e in library_dirs]
        libs = [] if libraries is None else [f"-l{e}" for e in libraries]

        cmd = [
            self.hipcc_exe,
            str(source_file),
            "-o",
            str(output_file),
            *flags,
            *includedirs,
            *libdirs,
            *libs,
        ]

        self.logger.info("Compiling %s with command %s", str(source_file), cmd)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            success = result.returncode == 0
            errors = result.stderr if not success else None
            raw_output = result.stdout + "\n" + result.stderr
        except FileNotFoundError as e:
            msg = f"hipcc executable not found at '{self.hipcc_exe}'"
            self.logger.exception(msg)
            raise FileNotFoundError(msg) from e
        except subprocess.CalledProcessError as e:
            msg = f"hipcc execution failed: {e.stderr}"
            self.logger.exception(msg)
            raise RuntimeError(msg) from e

        if success:
            self.logger.info(
                "Compilation of %s succeeded, executable in %s",
                str(source_file),
                str(output_file),
            )
        else:
            self.logger.error("Compilation of %s failed. Error(s):\n%s", str(source_file), errors)

        return HipCompilerResult(success=success, errors=errors, raw_output=raw_output)
