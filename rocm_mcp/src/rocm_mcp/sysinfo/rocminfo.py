# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from os import PathLike


class DeviceType(Enum):
    """Enum for HSA device types."""

    Unknown = "Unknown"
    CPU = "CPU"
    GPU = "GPU"
    DSP = "DSP"


@dataclass(frozen=True)
class AgentInfo:
    """Information about a single HSA agent from rocminfo.

    Attributes:
        agent_number (int): Agent number/index.
        name (str): Agent name (e.g., "gfx1100" or processor name).
        uuid (str): Agent UUID.
        marketing_name (str): Marketing name of the device.
        vendor_name (str): Vendor name (e.g., "AMD", "CPU").
        device_type (DeviceType): Type of device.
        compute_units (int | None): Number of compute units.
        max_clock_freq (int | None): Maximum clock frequency in MHz.
        profile (str | None): HSA profile (e.g., "FULL_PROFILE", "BASE_PROFILE").
    """

    agent_number: int
    name: str
    uuid: str
    marketing_name: str
    vendor_name: str
    device_type: DeviceType
    compute_units: int | None
    max_clock_freq: int | None
    profile: str | None


@dataclass(frozen=True)
class RocminfoResult:
    """Result of rocminfo execution.

    Attributes:
        agents (list[AgentInfo]): List of all detected HSA agents.
        raw_output (str): Raw output from rocminfo command.
    """

    agents: list[AgentInfo]
    raw_output: str


class Rocminfo:
    """Class to handle execution and parsing of ROCm 'rocminfo' tool.

    This class provides a Python interface to query ROCm device information using
    the `rocminfo` command-line tool. It parses the output and returns structured
    data about all HSA agents (CPUs, GPUs, etc.) available on the system.

    Example:
        Basic usage to get information about all agents::

            from rocm_mcp import Rocminfo

            # Initialize with default rocminfo path
            rocminfo = Rocminfo()

            # Get information about all agents
            result = rocminfo.get_agents()

            # Access agent information
            for agent in result.agents:
                print(f"Agent {agent.agent_number}: {agent.marketing_name}")
                print(f"  Type: {agent.device_type.value}")
                print(f"  Vendor: {agent.vendor_name}")
                if agent.compute_units:
                    print(f"  Compute Units: {agent.compute_units}")
                if agent.max_clock_freq:
                    print(f"  Max Clock Freq: {agent.max_clock_freq} MHz")

        Using a custom rocminfo executable path::

            # Specify custom path directly
            rocminfo = Rocminfo(rocminfo="/opt/rocm/bin/rocminfo")

            # Or use environment variable
            import os

            os.environ["INTELLIKIT_ROCMINFO"] = "/opt/rocm/bin/rocminfo"
            rocminfo = Rocminfo()

        Filtering agents by device type::

            from rocm_mcp import DeviceType

            result = rocminfo.get_agents()
            gpus = [agent for agent in result.agents if agent.device_type == DeviceType.GPU]
            cpus = [agent for agent in result.agents if agent.device_type == DeviceType.CPU]

            print(f"Found {len(gpus)} GPU(s) and {len(cpus)} CPU(s)")

    Attributes:
        logger (logging.Logger): Logger for logging messages.
        rocminfo_exe (str): Path to the `rocminfo` executable.
    """

    logger: logging.Logger
    rocminfo_exe: str

    def __init__(
        self,
        logger: logging.Logger | None = None,
        rocminfo: str | PathLike | None = None,
    ) -> None:
        """Initialize the Rocminfo wrapper.

        Args:
            logger (logging.Logger): Logger instance for logging. If None, a default logger is
                created.
            rocminfo (str | PathLike | None): Path to the `rocminfo`
                executable. If None, uses the `INTELLIKIT_ROCMINFO` environment variable
                or defaults to 'rocminfo' in the system PATH.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.rocminfo_exe = os.getenv(
            "INTELLIKIT_ROCMINFO", str(rocminfo) if rocminfo is not None else "rocminfo"
        )
        self.logger.info("Initialized rocminfo wrapper.")

    def get_agents(self) -> RocminfoResult:
        """Execute rocminfo and parse the output to extract agent information.

        Returns:
            RocminfoResult: Parsed information about all HSA agents.

        Raises:
            FileNotFoundError: If rocminfo executable is not found.
            RuntimeError: If rocminfo execution fails.
        """
        cmd = [self.rocminfo_exe]

        self.logger.info("Executing rocminfo with command: %s", cmd)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError as e:
            msg = f"rocminfo executable not found at '{self.rocminfo_exe}'"
            self.logger.exception(msg)
            raise FileNotFoundError(msg) from e
        except subprocess.CalledProcessError as e:
            msg = f"rocminfo execution failed: {e.stderr}"
            self.logger.exception(msg)
            raise RuntimeError(msg) from e

        raw_output = result.stdout
        self.logger.debug("rocminfo raw output:\n%s", raw_output)

        agents = self._parse_agents(raw_output)
        self.logger.info("Found %d agents", len(agents))

        return RocminfoResult(agents=agents, raw_output=raw_output)

    def _parse_agents(self, output: str) -> list[AgentInfo]:
        """Parse rocminfo output to extract agent information.

        Args:
            output (str): Raw output from rocminfo command.

        Returns:
            list[AgentInfo]: List of parsed agent information.
        """
        agents = []

        # Split the output into agent sections
        # Each agent section starts with "*******" followed by "Agent N"
        # Uses a lookahead to find where each section ends (either at the next agent
        # or at "*** Done ***" marker)
        agent_pattern = re.compile(
            r"\*+\s*\n\s*Agent\s+(\d+)\s*\n\s*\*+\s*\n(.*?)"
            r"(?=\*+\s*\n\s*Agent\s+\d+|\*+\s*Done)",
            re.DOTALL,
        )

        for match in agent_pattern.finditer(output):
            agent_number = int(match.group(1))
            agent_text = match.group(2)

            # Extract agent attributes
            name = self._extract_field(agent_text, r"Name:\s*(.+)")
            uuid = self._extract_field(agent_text, r"Uuid:\s*(.+)")
            marketing_name = self._extract_field(agent_text, r"Marketing Name:\s*(.+)")
            vendor_name = self._extract_field(agent_text, r"Vendor Name:\s*(.+)")
            device_type = self._extract_field(agent_text, r"Device Type:\s*(.+)")
            profile = self._extract_field(agent_text, r"Profile:\s*(.+)")

            # Extract numeric fields
            compute_units_str = self._extract_field(agent_text, r"Compute Unit:\s*(\d+)")
            compute_units = int(compute_units_str) if compute_units_str else None

            max_clock_str = self._extract_field(agent_text, r"Max Clock Freq\. \(MHz\):\s*(\d+)")
            max_clock_freq = int(max_clock_str) if max_clock_str else None

            agents.append(
                AgentInfo(
                    agent_number=agent_number,
                    name=name or "",
                    uuid=uuid or "",
                    marketing_name=marketing_name or "",
                    vendor_name=vendor_name or "",
                    device_type=getattr(DeviceType, str(device_type), DeviceType.Unknown)
                    if device_type
                    else DeviceType.Unknown,
                    compute_units=compute_units,
                    max_clock_freq=max_clock_freq,
                    profile=profile,
                )
            )

        return agents

    def _extract_field(self, text: str, pattern: str) -> str | None:
        """Extract a field from text using a regex pattern.

        Args:
            text (str): Text to search in.
            pattern (str): Regex pattern with one capture group.

        Returns:
            str | None: Extracted field value or None if not found.
        """
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return None
