# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import logging
from dataclasses import dataclass

import amdsmi


@dataclass(frozen=True)
class DriverInformationResult:
    """Result of driver version query.

    Attributes:
        version (str): The driver version string.
        name (str): The driver name (e.g., "amdgpu").
        date (str): The driver date string.
    """

    version: str
    name: str
    date: str


class AmdSmi:
    """Class to query AMD GPU system management information via the amdsmi Python API.

    Example:
        Basic usage to get the GPU driver version::

            from rocm_mcp.sysinfo import AmdSmi

            smi = AmdSmi()
            result = smi.get_driver_information()
            print(f"Driver: {result.name} {result.version}")

    Attributes:
        logger (logging.Logger): Logger for logging messages.
    """

    logger: logging.Logger

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the AmdSmi wrapper.

        Args:
            logger (logging.Logger): Logger instance. If None, a default logger is created.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        amdsmi.amdsmi_init()

        self.logger.info("Initialized amd-smi wrapper.")

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        try:
            amdsmi.amdsmi_shut_down()
            self.logger.info("Shut down amd-smi wrapper.")
        except Exception as e:
            msg = f"Exception during amd-smi shutdown: {e}"
            self.logger.exception(msg)

    def get_driver_information(self) -> DriverInformationResult:
        """Get AMD GPU driver version via the amdsmi Python API.

        Returns:
            DriverInformationResult: The driver version, name, and date.

        Raises:
            RuntimeError: If no GPU processors are found or the query fails.
        """
        handles = amdsmi.amdsmi_get_processor_handles()
        if not handles:
            msg = "No GPU processors found"
            self.logger.error(msg)
            raise RuntimeError(msg)

        try:
            info = amdsmi.amdsmi_get_gpu_driver_info(handles[0])
        except amdsmi.AmdSmiException as e:
            msg = f"amdsmi query failed: {e}"
            self.logger.exception(msg)
            raise RuntimeError(msg) from e

        return DriverInformationResult(
            version=info["driver_version"],
            name=info["driver_name"],
            date=info["driver_date"],
        )
