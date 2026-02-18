# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""HIP interop functions for Accordo."""

import ctypes
import logging

import numpy as np

rt_path = "libamdhip64.so"
hip_runtime = ctypes.cdll.LoadLibrary(rt_path)


def hip_try(err):
    """Check HIP error code and raise exception if error occurred."""
    if err != 0:
        hip_runtime.hipGetErrorString.restype = ctypes.c_char_p
        error_string = hip_runtime.hipGetErrorString(ctypes.c_int(err)).decode("utf-8")
        raise RuntimeError(f"HIP error code {err}: {error_string}")


class hipIpcMemHandle_t(ctypes.Structure):
    """HIP IPC memory handle structure."""

    _fields_ = [("reserved", ctypes.c_char * 64)]


def open_ipc_handle(ipc_handle_data):
    """Open a HIP IPC memory handle.

    Args:
            ipc_handle_data: NumPy array of uint8 with 64 elements

    Returns:
            Device pointer value
    """
    ptr = ctypes.c_void_p()
    hipIpcMemLazyEnablePeerAccess = ctypes.c_uint(1)
    hip_runtime.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        hipIpcMemHandle_t,
        ctypes.c_uint,
    ]

    if isinstance(ipc_handle_data, np.ndarray):
        if ipc_handle_data.dtype != np.uint8 or ipc_handle_data.size != 64:
            logging.debug(f"ipc_handle_data.size: {ipc_handle_data.size}")
            raise ValueError("ipc_handle_data must be a 64-element uint8 numpy array")
        ipc_handle_bytes = ipc_handle_data.tobytes()
        ipc_handle_data = (ctypes.c_char * 64).from_buffer_copy(ipc_handle_bytes)
    else:
        raise TypeError("ipc_handle_data must be a numpy.ndarray of dtype uint8 with 64 elements")

    raw_memory = ctypes.create_string_buffer(64)
    ctypes.memset(raw_memory, 0x00, 64)
    ipc_handle_struct = hipIpcMemHandle_t.from_buffer(raw_memory)
    ipc_handle_data_bytes = bytes(ipc_handle_data)
    ctypes.memmove(raw_memory, ipc_handle_data_bytes, 64)

    logging.debug("[ipc_handle_struct]:")
    for i in range(0, len(ipc_handle_data_bytes), 16):
        chunk = ipc_handle_data_bytes[i : i + 16]
        logging.debug(" ".join(f"{b:02x}" for b in chunk))

    hip_try(
        hip_runtime.hipIpcOpenMemHandle(
            ctypes.byref(ptr),
            ipc_handle_struct,
            hipIpcMemLazyEnablePeerAccess,
        )
    )

    return ptr.value


def memcpy_d2h(ptr, num_elements_to_copy, dtype):
    """Copy data from device to host.

    Args:
            ptr: Device pointer value
            num_elements_to_copy: Number of elements to copy
            dtype: C type of elements

    Returns:
            NumPy array with copied data
    """
    host_array = np.zeros(num_elements_to_copy, dtype=np.dtype(dtype))

    hip_runtime.hipMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    bytes_to_copy = num_elements_to_copy * ctypes.sizeof(dtype)
    logging.debug(
        f"Copying {num_elements_to_copy * ctypes.sizeof(dtype)} bytes from {hex(ptr)} to {hex(host_array.ctypes.data)}"
    )

    hip_try(
        hip_runtime.hipMemcpy(
            ctypes.c_void_p(host_array.ctypes.data),
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(bytes_to_copy),
            2,  # hipMemcpyDeviceToHost
        )
    )
    return host_array
