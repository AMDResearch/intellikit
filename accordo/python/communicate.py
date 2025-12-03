################################################################################
# MIT License

# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

import ctypes
import errno
import logging
import math
import os
import stat
import sys
import threading
import time

import ml_dtypes
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hip import memcpy_d2h, open_ipc_handle


def run_with_timeout(func, timeout_seconds, *args, **kwargs):
	"""Cross-platform timeout wrapper using threading.

	Runs func in a thread and raises TimeoutError if it doesn't complete in time.
	"""
	result = [None]
	exception = [None]

	def target():
		try:
			result[0] = func(*args, **kwargs)
		except Exception as e:
			exception[0] = e

	thread = threading.Thread(target=target, daemon=True)
	thread.start()
	thread.join(timeout=timeout_seconds)

	if thread.is_alive():
		# Thread is still running - timeout occurred
		raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

	if exception[0] is not None:
		raise exception[0]

	return result[0]


def read_ipc_handles(args, ipc_file_name):
	count = sum(1 for arg in args if "*" in arg and "const" not in arg)

	handles = []
	sizes = []
	handles_set = set()

	while len(handles) < count:
		if not os.path.exists(ipc_file_name):
			time.sleep(0.1)
			continue

		with open(ipc_file_name, "rb") as file:
			data = file.read()

		messages = data.split(b"BEGIN\n")
		for message in messages:
			if b"END\n" in message:
				content = message.split(b"END\n")[0]

				if len(content) == 72:
					handle_data = content[:64]
					size_data = content[64:72]

					handle_np = np.frombuffer(handle_data, dtype=np.uint8)
					handle_tuple = tuple(handle_np)

					if handle_tuple not in handles_set:
						handles.append(handle_np)
						handles_set.add(handle_tuple)

						size_value = int.from_bytes(size_data, byteorder="little")
						sizes.append(size_value)

						# Verbose IPC handle debugging (only when new handle received)
						if logging.getLogger().isEnabledFor(logging.DEBUG):
							logging.debug("Final IPC Handle (hex):")
							for i in range(0, len(handle_np), 16):
								chunk = handle_np[i : i + 16]
								logging.debug(" ".join(f"{b:02x}" for b in chunk))
							logging.debug(f"Corresponding Pointer Size: {size_value} bytes")

		if len(handles) < count:
			# Don't spam logs in hot loop - removed logging.debug here
			time.sleep(0.1)

	return handles, sizes


def send_response(pipe_name):
	with open(pipe_name, "w") as fifo:
		fifo.write("done\n")


def get_kern_arg_data(pipe_name, args, ipc_file_name, ipc_timeout_seconds=30, process_pid=None, baseline_time_ms=None):
	logging.debug(f"pipe_name: {pipe_name}")
	logging.debug(f"get_kern_arg_data args: {args}")
	logging.debug(f"ipc_file_name: {ipc_file_name}")

	# Calculate dynamic timeout based on baseline performance
	if baseline_time_ms is not None and baseline_time_ms > 0:
		# 2x baseline, rounded up to next second, minimum 3 seconds
		ipc_timeout_seconds = max(3, math.ceil(baseline_time_ms / 1000.0 * 2.0))
		logging.debug(f"Using dynamic timeout: {ipc_timeout_seconds}s (2x baseline of {baseline_time_ms}ms)")
	else:
		logging.debug(f"Using default timeout: {ipc_timeout_seconds}s (no baseline available)")

	def _do_ipc_work():
		"""Inner function that does the actual IPC work - wrapped with timeout"""
		fifo = None
		try:
			if not os.path.exists(pipe_name):
				os.mkfifo(pipe_name)
				os.chmod(pipe_name, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

			# Try to open the pipe, checking if the process is still alive
			while fifo is None:
				# Check if the process is still alive (crash detection, not timeout)
				if process_pid is not None:
					try:
						os.kill(process_pid, 0)  # Signal 0 just checks if process exists
					except OSError:
						raise RuntimeError(
							f"Accordo process (PID {process_pid}) crashed or terminated before opening pipe. Check for segfaults or GPU memory access errors."
						)

				try:
					# Try non-blocking open
					fd = os.open(pipe_name, os.O_RDONLY | os.O_NONBLOCK)
					fifo = os.fdopen(fd, "rb")
					break
				except OSError as e:
					if e.errno == errno.ENXIO:  # ENXIO - no writer connected yet
						time.sleep(0.1)
						continue
					else:
						raise

			# Read IPC handles
			while True:
				# Check if the process is still alive (crash detection, not timeout)
				if process_pid is not None:
					try:
						os.kill(process_pid, 0)
					except OSError:
						raise RuntimeError(
							f"Accordo process (PID {process_pid}) crashed or terminated during execution. Check for segfaults or GPU memory access errors."
						)

				try:
					ipc_handles, ptr_sizes = read_ipc_handles(args, ipc_file_name)
					break
				except Exception:
					# For non-timeout exceptions, retry with a short sleep
					time.sleep(0.1)

			return ipc_handles, ptr_sizes

		finally:
			if fifo:
				fifo.close()

	# Run IPC work with timeout wrapper (cross-platform)
	try:
		ipc_handles, ptr_sizes = run_with_timeout(_do_ipc_work, ipc_timeout_seconds)
	except TimeoutError:
		# Enhance timeout message with context
		timeout_msg = f"Timeout after {ipc_timeout_seconds} seconds during IPC communication"
		if baseline_time_ms is not None:
			timeout_msg += f" (baseline: {baseline_time_ms}ms, 2x timeout: {ipc_timeout_seconds}s). Code may be correct but too slow to be worth profiling."
		raise TimeoutError(timeout_msg)

	type_map = {
		"double*": ctypes.c_double,
		"float*": ctypes.c_float,
		"int*": ctypes.c_int,
		"std::size_t*": ctypes.c_size_t,
		"__half*": np.float16,
		"__hip_bfloat16*": ml_dtypes.bfloat16,
	}
	results = []
	pointer_args = list(filter(lambda arg: "*" in arg and "const" not in arg, args))
	logging.debug(f"pointer_args: {pointer_args}")
	for handle, arg, array_size in zip(ipc_handles, pointer_args, ptr_sizes):
		ptr = open_ipc_handle(handle)
		logging.debug(f"Opened IPC Ptr: {ptr} (0x{ptr:x})")
		arg_type = arg.split()[0]
		logging.debug(f"arg_type: {arg_type}")
		if arg_type in type_map:
			dtype = type_map[arg_type]
			logging.debug(f"dtype: {dtype}")
			# Special handling for FP16 and bfloat16
			if arg_type == "__half*":
				temp_array = memcpy_d2h(ptr, array_size // 2, ctypes.c_uint16)
				host_array = np.frombuffer(temp_array, dtype=np.float16)
			elif arg_type == "__hip_bfloat16*":
				temp_array = memcpy_d2h(ptr, array_size // 2, ctypes.c_uint16)
				host_array = np.frombuffer(temp_array, dtype=ml_dtypes.bfloat16)
			else:
				num_elements = array_size // ctypes.sizeof(dtype)
				host_array = memcpy_d2h(ptr, num_elements, dtype)
		else:
			raise TypeError(f"Unsupported pointer type: {arg_type}")

		logging.debug(f"Received data from IPC ({arg_type}/{len(host_array)}): {host_array}")
		results.append(host_array)
	return results
