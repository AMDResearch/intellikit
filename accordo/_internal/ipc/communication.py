# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""IPC communication for Accordo."""

import ctypes
import logging
import os
import stat
import time

import ml_dtypes
import numpy as np

from ..hip_interop import memcpy_d2h, open_ipc_handle


def read_ipc_handles(args, ipc_file_name):
    """Read IPC handles and sizes from the IPC file.

    Args:
            args: List of argument type strings
            ipc_file_name: Path to the IPC file

    Returns:
            Tuple of (handles, sizes)
    """
    count = sum(1 for arg in args if "*" in arg and "const" not in arg)

    handles = []
    sizes = []
    handles_set = set()

    while len(handles) < count:
        if not os.path.exists(ipc_file_name):
            logging.debug("Waiting for IPC file...")
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

                        logging.debug("Final IPC Handle (hex):")
                        for i in range(0, len(handle_np), 16):
                            chunk = handle_np[i : i + 16]
                            logging.debug(" ".join(f"{b:02x}" for b in chunk))

                        logging.debug(f"Corresponding Pointer Size: {size_value} bytes")

        if len(handles) < count:
            logging.debug(f"Waiting for {count - len(handles)} more IPC handles...")
            time.sleep(0.1)

    return handles, sizes


def send_response(pipe_name):
    """Send completion response through named pipe."""
    with open(pipe_name, "w") as fifo:
        fifo.write("done\n")


def get_kern_arg_data(pipe_name, args, ipc_file_name, ipc_timeout_seconds=30, process_pid=None, baseline_time_ms=None):
    """Get kernel argument data via IPC.

    Args:
            pipe_name: Path to the named pipe
            args: List of argument type strings
            ipc_file_name: Path to the IPC file
            ipc_timeout_seconds: Timeout for IPC operations
            process_pid: Process ID (for error messages)
            baseline_time_ms: Baseline execution time (for dynamic timeout)

    Returns:
            List of NumPy arrays with argument data

    Raises:
            TimeoutError: If IPC operation times out
            TypeError: If unsupported type encountered
    """
    # Calculate dynamic timeout if baseline provided
    if baseline_time_ms is not None:
        # Use 2x baseline or minimum 3 seconds
        dynamic_timeout = max(3.0, (baseline_time_ms / 1000.0) * 2.0)
        ipc_timeout_seconds = dynamic_timeout
        logging.debug(f"Using dynamic timeout: {ipc_timeout_seconds}s (2x baseline of {baseline_time_ms}ms)")

    logging.debug(f"pipe_name: {pipe_name}")
    logging.debug(f"get_kern_arg_data args: {args}")
    logging.debug(f"ipc_file_name: {ipc_file_name}")

    if not os.path.exists(pipe_name):
        os.mkfifo(pipe_name)
        os.chmod(pipe_name, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    start_time = time.time()
    with open(pipe_name, "rb") as fifo:  # noqa: F841
        while True:
            # Check if the process is still alive (crash detection, not timeout)
            if process_pid is not None:
                try:
                    os.kill(process_pid, 0)  # Signal 0 just checks if process exists
                except OSError:
                    raise RuntimeError(
                        f"Accordo process (PID {process_pid}) crashed or terminated during execution. "
                        "Check for segfaults or GPU memory access errors."
                    )

            if time.time() - start_time > ipc_timeout_seconds:
                timeout_msg = f"Timeout after {ipc_timeout_seconds} seconds during IPC communication"
                if baseline_time_ms is not None:
                    timeout_msg += f" (baseline: {baseline_time_ms}ms, 2x timeout: {ipc_timeout_seconds}s). Code may be correct but too slow to be worth profiling."
                raise TimeoutError(timeout_msg)

            try:
                ipc_handles, ptr_sizes = read_ipc_handles(args, ipc_file_name)
                break
            except Exception as e:
                if time.time() - start_time > ipc_timeout_seconds:
                    timeout_msg = f"Timeout after {ipc_timeout_seconds} seconds waiting for IPC data: {str(e)}"
                    if baseline_time_ms is not None:
                        timeout_msg += f" (baseline: {baseline_time_ms}ms, 2x timeout: {ipc_timeout_seconds}s). Code may be correct but too slow to be worth profiling."
                    raise TimeoutError(timeout_msg)
                time.sleep(0.1)

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

        # Strip type qualifiers (restrict, const, volatile) and type specifiers (struct, union, class, enum)
        words_to_strip = ("restrict", "const", "volatile", "struct", "union", "class", "enum")
        arg_type = " ".join(word for word in arg.split() if word not in words_to_strip)
        logging.debug(f"arg_type (after stripping qualifiers and specifiers): {arg_type}")

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
