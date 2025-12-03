# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""AccordoValidator: Main validation class for Accordo."""

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ._internal.codegen import generate_kernel_header
from ._internal.ipc.communication import get_kern_arg_data, send_response
from .config import ValidationConfig
from .exceptions import AccordoBuildError, AccordoProcessError, AccordoTimeoutError
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
	"""Validator for GPU kernel correctness using Accordo.

	This class manages the entire validation pipeline:
	- Building the Accordo C++ library
	- Running instrumented processes
	- Collecting kernel argument data via IPC
	- Validating arrays match within tolerance

	Example:
		>>> from accordo import AccordoValidator, ValidationConfig, KernelArg
		>>> config = ValidationConfig(
		...     kernel_name="my_kernel",
		...     kernel_args=[
		...         KernelArg(name="result", type="double*", direction="out"),
		...         KernelArg(name="input", type="const double*", direction="in"),
		...     ],
		...     tolerance=1e-6
		... )
		>>> validator = AccordoValidator(config)
		>>> result = validator.validate(reference_app, optimized_app)
		>>> if result.is_valid:
		...     print("Validation passed!")
	"""

	def __init__(
		self,
		config: ValidationConfig,
		accordo_path: Optional[Path] = None,
		force_rebuild: bool = False,
		parallel_jobs: int = 16,
	):
		"""Initialize AccordoValidator.

		Args:
			config: Validation configuration
			accordo_path: Path to Accordo directory (auto-detected if None)
			force_rebuild: Force rebuild even if library exists
			parallel_jobs: Number of parallel build jobs
		"""
		self.config = config
		self.parallel_jobs = parallel_jobs
		self._built = False
		self._lib_path = None

		# Auto-detect accordo_path if not provided
		if accordo_path is None:
			# Find it relative to this file (accordo package directory)
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

		# Build if forced or library doesn't exist
		lib_path = self.accordo_path / "build" / "lib" / "libaccordo.so"
		if force_rebuild or not lib_path.exists():
			logging.info("Building Accordo C++ library...")
			self._lib_path = _build_accordo(self.accordo_path, parallel_jobs)
			self._built = True
		else:
			self._lib_path = lib_path
			self._built = True

	def capture_snapshot(
		self,
		binary: list[str],
		working_directory: str = ".",
		timeout_seconds: int = 30,
	) -> Snapshot:
		"""Capture a snapshot of kernel argument data from a binary execution.

		Args:
			binary: Command to run binary (e.g., ["./app", "arg1"])
			working_directory: Directory to run binary from
			timeout_seconds: Timeout for this capture

		Returns:
			Snapshot object containing captured arrays and execution metadata

		Raises:
			AccordoBuildError: If Accordo library not built
			AccordoProcessError: If instrumented process crashes
			AccordoTimeoutError: If execution exceeds timeout

		Note:
			Binary must be pre-compiled. Accordo does not build applications.
		"""
		if not self._built:
			raise AccordoBuildError("Accordo library not built")

		# Generate kernel header with additional includes
		arg_types = self.config.get_arg_types()
		generate_kernel_header(arg_types, self.config.additional_includes)

		# Wrap app run with timeout
		old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
		signal.alarm(timeout_seconds)

		try:
			start_time = time.time()
			result_arrays = self._run_instrumented_app(
				binary, working_directory, label="snapshot", baseline_time_ms=None
			)
			signal.alarm(0)  # Cancel alarm on success
			execution_time_ms = (time.time() - start_time) * 1000

			return Snapshot(
				arrays=result_arrays,
				execution_time_ms=execution_time_ms,
				binary=binary,
				working_directory=working_directory,
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
	) -> ValidationResult:
		"""Compare two snapshots and validate their arrays.

		Args:
			reference_snapshot: Snapshot from reference binary (from capture_snapshot)
			optimized_snapshot: Snapshot from optimized binary (from capture_snapshot)

		Returns:
			ValidationResult with validation status and details
		"""
		results = {
			"reference": reference_snapshot.arrays,
			"optimized": optimized_snapshot.arrays,
		}
		execution_times = {
			"reference": reference_snapshot.execution_time_ms,
			"optimized": optimized_snapshot.execution_time_ms,
		}

		return self._validate_results(results, execution_times)

	def validate(
		self,
		reference_binary: list[str],
		optimized_binary: list[str],
		working_directory: str = ".",
		baseline_time_ms: Optional[float] = None,
	) -> ValidationResult:
		"""Validate optimized kernel against reference (convenience method).

		This is a convenience wrapper that captures both snapshots and compares them.
		For better performance when validating multiple optimizations against the same
		reference, use capture_snapshot() and compare_snapshots() directly.

		Args:
			reference_binary: Command to run reference binary (e.g., ["./app", "arg1"])
			optimized_binary: Command to run optimized binary (e.g., ["./app_opt", "arg1"])
			working_directory: Directory to run binaries from
			baseline_time_ms: Baseline execution time for dynamic timeout

		Returns:
			ValidationResult with validation status and details

		Raises:
			AccordoBuildError: If Accordo library not built
			AccordoProcessError: If instrumented process crashes
			AccordoTimeoutError: If execution exceeds timeout

		Note:
			Both binaries must be pre-compiled. Accordo does not build applications.
		"""
		# Calculate timeouts
		ref_timeout = 30  # Default for reference
		if baseline_time_ms:
			opt_timeout = int((baseline_time_ms * self.config.timeout_multiplier / 1000.0) + 30.0)
		else:
			opt_timeout = 30

		# Capture snapshots
		reference_snapshot = self.capture_snapshot(reference_binary, working_directory, ref_timeout)
		optimized_snapshot = self.capture_snapshot(optimized_binary, working_directory, opt_timeout)

		# Compare
		return self.compare_snapshots(reference_snapshot, optimized_snapshot)

	def _run_instrumented_app(
		self, binary_cmd: list[str], working_directory: str, label: str, baseline_time_ms: Optional[float] = None
	) -> list[np.ndarray]:
		"""Run an instrumented application and collect kernel argument data.

		Args:
			binary_cmd: Binary command with arguments (e.g., ["./app", "arg1"])
			working_directory: Directory to run the binary from
			label: Label for this run ("reference" or "optimized")
			baseline_time_ms: Baseline time for dynamic timeout

		Returns:
			List of numpy arrays with kernel argument data
		"""
		timestamp = int(time.time() * 1000)  # Use milliseconds for uniqueness
		pipe_name = f"/tmp/kernel_pipe_{timestamp}_{label}"
		ipc_file_name = f"/tmp/ipc_handle_{timestamp}_{label}.bin"

		# Clean up any existing files
		for file_path in [pipe_name, ipc_file_name]:
			if os.path.exists(file_path):
				os.remove(file_path)

		# Set up environment
		env = os.environ.copy()
		env["HSA_TOOLS_LIB"] = str(self._lib_path)
		env["KERNEL_TO_TRACE"] = self.config.kernel_name

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
		logging.debug(f"Launching {label} process with PID for kernel {self.config.kernel_name}")
		logging.debug(f"binary_cmd: {binary_cmd}")
		logging.debug(f"working_directory: {working_directory}")
		logging.debug(f"kernel_args: {self.config.get_arg_types()}")
		logging.debug(f"ipc_file_name: {ipc_file_name}")

		original_dir = os.getcwd()
		try:
			os.chdir(working_directory)
			process_pid = os.posix_spawn(binary_cmd[0], binary_cmd, env)
			logging.debug(f"Launched {label} process with PID: {process_pid}")
		finally:
			os.chdir(original_dir)

		# Get kernel argument data via IPC
		try:
			result_arrays = get_kern_arg_data(
				pipe_name,
				self.config.get_arg_types(),
				ipc_file_name,
				process_pid=process_pid,
				baseline_time_ms=baseline_time_ms,
			)
		except TimeoutError:
			# Kill the process if it timed out
			try:
				os.kill(process_pid, 9)
			except (OSError, ProcessLookupError):
				pass  # Process already dead
			raise

		# Send completion response
		send_response(pipe_name)

		return result_arrays

	def _validate_results(
		self, results: dict[str, list[np.ndarray]], execution_times: dict[str, float]
	) -> ValidationResult:
		"""Validate results from reference and optimized runs.

		Args:
			results: Dictionary with "reference" and "optimized" array lists
			execution_times: Execution times for each run

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

		for i, (ref_arr, opt_arr) in enumerate(zip(reference_arrays, optimized_arrays)):
			arg = self.config.kernel_args[i]

			if not _validate_arrays(ref_arr, opt_arr, self.config.tolerance):
				# Array mismatch
				diff = np.abs(ref_arr - opt_arr)
				mismatch = ArrayMismatch(
					arg_index=i,
					arg_name=arg.name,
					arg_type=arg.type,
					max_difference=float(np.max(diff)),
					mean_difference=float(np.mean(diff)),
					reference_sample=ref_arr[:10] if len(ref_arr) > 10 else ref_arr,
					optimized_sample=opt_arr[:10] if len(opt_arr) > 10 else opt_arr,
				)
				mismatches.append(mismatch)

				logging.debug(f"Arrays at index {i} for arg '{arg.name}' ({arg.type}) are NOT close.")
				logging.debug(f"  Max difference: {mismatch.max_difference}")
				logging.debug(f"  Mean difference: {mismatch.mean_difference}")
			else:
				# Array matched
				matched_arrays[arg.name] = {
					"index": i,
					"type": arg.type,
					"size": len(ref_arr),
				}
				logging.debug(f"Arrays at index {i} for arg '{arg.name}' ({arg.type}) are close.")

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
