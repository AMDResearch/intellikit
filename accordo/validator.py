# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Accordo: Main validation class for GPU kernel correctness."""

import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from ._internal.codegen import generate_kernel_metadata
from ._internal.ipc.communication import get_kern_arg_data, send_response
from .exceptions import AccordoBuildError, AccordoProcessError, AccordoTimeoutError
from .kernel_args import extract_kernel_arguments
from .result import ArrayMismatch, ValidationResult
from .snapshot import Snapshot


class _TimeoutException(Exception):
    """Internal exception for timeout handling."""

    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise _TimeoutException("Operation timed out")


def _build_accordo(accordo_path: Path, parallel_jobs: int = 16) -> Path:
    """Build Accordo C++ library.

    Args:
            accordo_path: Path to Accordo directory
            parallel_jobs: Number of parallel build jobs

    Returns:
            Path to built library

    Raises:
            AccordoBuildError: If build fails
    """
    try:
        # Configure with CMake
        result = subprocess.run(
            ["cmake", "-B", "build"],
            cwd=accordo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        logging.debug(f"CMake configure output: {result.stdout}")

        # Build
        result = subprocess.run(
            ["cmake", "--build", "build", "--parallel", str(parallel_jobs)],
            cwd=accordo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        logging.debug(f"CMake build output: {result.stdout}")

        lib_path = accordo_path / "build" / "lib" / "libaccordo.so"
        if not lib_path.exists():
            raise AccordoBuildError(f"Library not found at {lib_path}")

        return lib_path

    except subprocess.CalledProcessError as e:
        raise AccordoBuildError(f"Accordo build failed: {e.stderr}")
    except Exception as e:
        raise AccordoBuildError(f"Accordo build failed: {str(e)}")


def _validate_arrays(arr1: np.ndarray, arr2: np.ndarray, tolerance: float) -> bool:
    """Validate two arrays are close within tolerance.

    Args:
            arr1: First array
            arr2: Second array
            tolerance: Absolute tolerance

    Returns:
            True if arrays match within tolerance
    """
    return np.allclose(arr1, arr2, atol=tolerance, rtol=0)


class Accordo:
    """Validator for a specific GPU kernel.

    Each Accordo instance is tied to a specific kernel signature. The library
    is built once in __init__, then reused for all capture_snapshot calls.

    Example:
            >>> validator = Accordo(
            ...     binary="./app_ref",
            ...     kernel_name="reduce_sum"
            ... )
            >>> ref = validator.capture_snapshot(binary=["./app_ref"])
            >>> opt = validator.capture_snapshot(binary=["./app_opt"])
            >>> result = validator.compare_snapshots(ref, opt, tolerance=1e-6)
    """

    def __init__(
        self,
        binary: Union[str, List[str]],
        kernel_name: str,
        kernel_args: Optional[List[Tuple[str, str]]] = None,
        working_directory: str = ".",
        accordo_path: Optional[Path] = None,
        force_rebuild: bool = False,
        parallel_jobs: int = 16,
        log_level: str = "WARNING",
    ):
        """Initialize Accordo validator for a specific kernel.

        This extracts the kernel signature, generates the header, and builds
        libaccordo.so. The built library is then reused for all captures.

        Args:
                binary: Binary to extract kernel signature from (string or command list)
                kernel_name: Name of the kernel to validate
                kernel_args: Optional manual kernel args as [(name, type), ...].
                             If None, auto-extracted from binary.
                working_directory: Working directory for binary execution
                accordo_path: Path to Accordo directory (auto-detected if None)
                force_rebuild: Force rebuild even if library exists
                parallel_jobs: Number of parallel build jobs
                log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")

        Raises:
                AccordoBuildError: If kernel signature extraction or build fails

        Example:
                >>> # Auto-extract kernel args (requires -g flag)
                >>> validator = Accordo(
                ...     binary="./app_ref",
                ...     kernel_name="reduce_sum"
                ... )
                >>>
                >>> # Or specify manually
                >>> validator = Accordo(
                ...     binary="./app_ref",
                ...     kernel_name="reduce_sum",
                ...     kernel_args=[("input", "const float*"), ("output", "float*")]
                ... )
        """
        self.kernel_name = kernel_name
        self.working_directory = working_directory
        self.parallel_jobs = parallel_jobs

        # Set log level
        logging.basicConfig(level=getattr(logging, log_level.upper()))

        # Normalize binary to list
        if isinstance(binary, str):
            binary = [binary]

        # Auto-extract kernel arguments if not provided
        if kernel_args is None:
            logging.info(f"Extracting kernel arguments for '{kernel_name}' from {binary[0]}...")
            try:
                kernel_args = extract_kernel_arguments(
                    binary_path=binary[0], kernel_name=kernel_name, working_directory=working_directory
                )
                logging.info(f"Successfully extracted {len(kernel_args)} kernel argument(s)")
            except Exception as e:
                raise AccordoBuildError(
                    f"Failed to auto-extract kernel arguments: {str(e)}\n"
                    "Please specify kernel_args manually or compile with -g flag."
                )

        self.kernel_args = kernel_args

        # Extract type strings for metadata generation
        arg_types = [arg[1] for arg in kernel_args]

        # Generate kernel metadata (JSON)
        logging.info("Generating kernel metadata...")
        self.metadata_path = generate_kernel_metadata(arg_types)

        # Auto-detect accordo_path if not provided
        if accordo_path is None:
            accordo_dir = Path(__file__).parent
            if (accordo_dir / "build").exists() or (accordo_dir / "CMakeLists.txt").exists():
                accordo_path = accordo_dir
            else:
                raise RuntimeError(
                    f"Could not find Accordo build directory. Expected at {accordo_dir}. "
                    "Please build Accordo first or specify accordo_path explicitly."
                )

        self.accordo_path = Path(accordo_path)
        logging.debug(f"Accordo path: {self.accordo_path}")

        # Build library
        lib_path = self.accordo_path / "build" / "lib" / "libaccordo.so"
        if force_rebuild or not lib_path.exists():
            logging.info(f"Building Accordo library for kernel '{kernel_name}'...")
            self._lib_path = _build_accordo(self.accordo_path, parallel_jobs)
            logging.info(f"Build complete: {self._lib_path}")
        else:
            self._lib_path = lib_path
            logging.info(f"Using existing library: {self._lib_path}")

    def capture_snapshot(
        self,
        binary: Union[str, List[str]],
        timeout_seconds: int = 30,
        dispatch_id: Optional[int] = None,
    ) -> Snapshot:
        """Capture a snapshot of kernel argument data from a binary execution.

        Args:
                binary: Command to run binary (string or list like ["./app", "arg1"])
                timeout_seconds: Timeout for this capture
                dispatch_id: Optional dispatch ID to capture (future feature)

        Returns:
                Snapshot object containing captured arrays and execution metadata

        Raises:
                AccordoProcessError: If instrumented process crashes
                AccordoTimeoutError: If execution exceeds timeout

        Example:
                >>> validator = Accordo(binary="./ref", kernel_name="reduce")
                >>> ref = validator.capture_snapshot(binary="./ref")
                >>> opt = validator.capture_snapshot(binary="./opt_v1")
        """
        # Normalize binary to list
        if isinstance(binary, str):
            binary = [binary]

        # Wrap app run with timeout
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            start_time = time.time()
            # Prepare a metadata file path for exporter to write dispatch dims
            dispatch_metadata_file = f"/tmp/accordo_dispatch_{int(start_time * 1000)}.json"

            extra_env = {"ACCORDO_DISPATCH_METADATA_FILE": dispatch_metadata_file}
            if dispatch_id is not None:
                extra_env["ACCORDO_DISPATCH_ID"] = str(dispatch_id)

            result_arrays = self._run_instrumented_app(
                binary_cmd=binary,
                label="snapshot",
                extra_env=extra_env,
            )
            signal.alarm(0)  # Cancel alarm on success
            execution_time_ms = (time.time() - start_time) * 1000

            # Try to read grid/block metadata if exporter produced it
            grid = None
            block = None
            try:
                if os.path.exists(dispatch_metadata_file):
                    with open(dispatch_metadata_file, "r") as f:
                        meta = json.load(f)
                    if isinstance(meta, dict):
                        grid = meta.get("grid")
                        block = meta.get("block")
            except Exception:
                pass

            return Snapshot(
                arrays=result_arrays,
                execution_time_ms=execution_time_ms,
                binary=binary,
                working_directory=self.working_directory,
                grid_size=grid,
                block_size=block,
            )
        except _TimeoutException:
            signal.alarm(0)
            logging.error(f"Snapshot capture timed out after {timeout_seconds}s")
            raise AccordoTimeoutError(
                f"Snapshot capture timed out after {timeout_seconds}s. This may indicate a GPU crash or hung process.",
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError as e:
            signal.alarm(0)
            raise AccordoTimeoutError(f"Snapshot timeout: {str(e)}", timeout_seconds)
        except RuntimeError as e:
            signal.alarm(0)
            raise AccordoProcessError(f"Process crashed during snapshot: {str(e)}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def compare_snapshots(
        self,
        reference_snapshot: Snapshot,
        optimized_snapshot: Snapshot,
        tolerance: float = 1e-6,
    ) -> ValidationResult:
        """Compare two snapshots and validate their arrays.

        Args:
                reference_snapshot: Snapshot from reference binary
                optimized_snapshot: Snapshot from optimized binary
                tolerance: Absolute tolerance for array comparison

        Returns:
                ValidationResult with validation status and details

        Example:
                >>> result = validator.compare_snapshots(ref, opt, tolerance=1e-4)
                >>> if result.is_valid:
                ...     print(f"✓ PASS: {result.num_arrays_validated} arrays matched")
        """
        results = {
            "reference": reference_snapshot.arrays,
            "optimized": optimized_snapshot.arrays,
        }
        execution_times = {
            "reference": reference_snapshot.execution_time_ms,
            "optimized": optimized_snapshot.execution_time_ms,
        }

        return self._validate_results(
            results=results,
            execution_times=execution_times,
            tolerance=tolerance,
        )

    def _run_instrumented_app(
        self,
        binary_cmd: List[str],
        label: str,
        extra_env: Optional[dict] = None,
    ) -> List[np.ndarray]:
        """Run an instrumented application and collect kernel argument data.

        Args:
                binary_cmd: Binary command with arguments
                label: Label for this run
                extra_env: Optional extra environment variables

        Returns:
                List of numpy arrays with kernel argument data
        """
        timestamp = int(time.time() * 1000)
        pipe_name = f"/tmp/kernel_pipe_{timestamp}_{label}"
        ipc_file_name = f"/tmp/ipc_handle_{timestamp}_{label}.bin"

        # Clean up any existing files
        for file_path in [pipe_name, ipc_file_name]:
            if os.path.exists(file_path):
                os.remove(file_path)

        # Set up environment
        env = os.environ.copy()
        env["HSA_TOOLS_LIB"] = str(self._lib_path)
        env["KERNEL_TO_TRACE"] = self.kernel_name
        env["ACCORDO_METADATA_FILE"] = self.metadata_path  # Pass metadata to C++
        if extra_env:
            env.update(extra_env)

        # Set log level
        debug_level = logging.getLogger().getEffectiveLevel()
        level_map = {
            logging.WARNING: 0,
            logging.INFO: 1,
            logging.DEBUG: 2,
            logging.NOTSET: 3,
        }
        env["ACCORDO_LOG_LEVEL"] = str(level_map.get(debug_level, 0))
        env["ACCORDO_PIPE_NAME"] = pipe_name
        env["ACCORDO_IPC_OUTPUT_FILE"] = ipc_file_name

        # Launch process
        logging.debug(f"Launching {label} process for kernel {self.kernel_name}")
        logging.debug(f"binary_cmd: {binary_cmd}")
        logging.debug(f"working_directory: {self.working_directory}")
        logging.debug(f"ipc_file_name: {ipc_file_name}")

        original_dir = os.getcwd()
        try:
            os.chdir(self.working_directory)
            process_pid = os.posix_spawn(binary_cmd[0], binary_cmd, env)
            logging.debug(f"Launched {label} process with PID: {process_pid}")
        finally:
            os.chdir(original_dir)

        # Get kernel argument data via IPC
        arg_types = [arg[1] for arg in self.kernel_args]
        try:
            result_arrays = get_kern_arg_data(
                pipe_name,
                arg_types,
                ipc_file_name,
                process_pid=process_pid,
                baseline_time_ms=None,
            )
        except TimeoutError:
            # Kill the process if it timed out
            try:
                os.kill(process_pid, 9)
            except (OSError, ProcessLookupError):
                pass
            raise

        # Send completion response
        send_response(pipe_name)

        return result_arrays

    def _validate_results(
        self,
        results: dict,
        execution_times: dict,
        tolerance: float,
    ) -> ValidationResult:
        """Validate results from reference and optimized runs.

        Args:
                results: Dictionary with "reference" and "optimized" array lists
                execution_times: Execution times for each run
                tolerance: Absolute tolerance for comparison

        Returns:
                ValidationResult with validation status
        """
        reference_arrays = results["reference"]
        optimized_arrays = results["optimized"]

        if len(reference_arrays) != len(optimized_arrays):
            return ValidationResult(
                is_valid=False,
                error_message=f"Array count mismatch: {len(reference_arrays)} vs {len(optimized_arrays)}",
                execution_time_ms=execution_times,
            )

        mismatches = []
        matched_arrays = {}

        # Create mapping from array index to kernel arg index (only for output args)
        # Arrays list contains only outputs, but kernel_args contains all arguments
        output_arg_indices = [
            i for i, (arg_name, arg_type) in enumerate(self.kernel_args)
            if "*" in arg_type and "const" not in arg_type
        ]

        for array_idx, (ref_arr, opt_arr) in enumerate(zip(reference_arrays, optimized_arrays)):
            # Map array index to the correct kernel argument index
            kernel_arg_idx = output_arg_indices[array_idx]
            arg_name, arg_type = self.kernel_args[kernel_arg_idx]

            if not _validate_arrays(ref_arr, opt_arr, tolerance):
                # Array mismatch
                diff = np.abs(ref_arr - opt_arr)
                mismatch = ArrayMismatch(
                    arg_index=kernel_arg_idx,  # Use kernel arg index, not array index
                    arg_name=arg_name,
                    arg_type=arg_type,
                    max_difference=float(np.max(diff)),
                    mean_difference=float(np.mean(diff)),
                    reference_sample=ref_arr[:10] if len(ref_arr) > 10 else ref_arr,
                    optimized_sample=opt_arr[:10] if len(opt_arr) > 10 else opt_arr,
                )
                mismatches.append(mismatch)

                logging.debug(f"Output array {array_idx} (kernel arg {kernel_arg_idx} '{arg_name}' {arg_type}) - NOT close")
                logging.debug(f"  Max difference: {mismatch.max_difference}")
                logging.debug(f"  Mean difference: {mismatch.mean_difference}")
            else:
                # Array matched
                matched_arrays[arg_name] = {
                    "index": kernel_arg_idx,  # Use kernel arg index
                    "type": arg_type,
                    "size": len(ref_arr),
                }
                logging.debug(f"Output array {array_idx} (kernel arg {kernel_arg_idx} '{arg_name}' {arg_type}) - MATCH")

        # Determine overall success
        is_valid = len(mismatches) == 0

        if is_valid:
            return ValidationResult(
                is_valid=True,
                matched_arrays=matched_arrays,
                execution_time_ms=execution_times,
            )
        else:
            # Build error message
            error_lines = [f"Validation failed: {len(mismatches)} array(s) mismatched"]
            for m in mismatches:
                error_lines.append(f"  - {m}")
            error_message = "\n".join(error_lines)

            return ValidationResult(
                is_valid=False,
                error_message=error_message,
                mismatches=mismatches,
                matched_arrays=matched_arrays,
                execution_time_ms=execution_times,
            )
