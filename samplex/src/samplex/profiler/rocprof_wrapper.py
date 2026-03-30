# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
rocprofv3 PC sampling wrapper.

Supports two sampling methods:
  - stochastic: Hardware-based, cycle-accurate, zero skid. Provides stall
    reasons, instruction types, and wave counts. Requires MI300+ (gfx942+).
  - host_trap: Software-based, time-based (nanoseconds). Broader GPU support
    (MI200+). No stall reasons or instruction types.
"""

import csv
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..logger import logger


VALID_METHODS = ("stochastic", "host_trap")


@dataclass
class PCSample:
    """A single PC sample from rocprofv3."""

    timestamp: int
    exec_mask: int
    dispatch_id: int
    instruction: str
    instruction_comment: str
    correlation_id: int
    wave_issued: bool = False
    instruction_type: str = ""
    stall_reason: str = ""
    wave_count: int = 0


@dataclass
class KernelDispatch:
    """A kernel dispatch from the kernel trace."""

    dispatch_id: int
    kernel_name: str
    agent_id: str
    start_timestamp: int
    end_timestamp: int
    workgroup_size: tuple = (0, 0, 0)
    grid_size: tuple = (0, 0, 0)
    vgpr_count: int = 0
    sgpr_count: int = 0

    @property
    def duration_ns(self) -> int:
        return self.end_timestamp - self.start_timestamp

    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1000.0


@dataclass
class PCSamplingResult:
    """Complete result from a PC sampling run."""

    command: str
    interval: int
    method: str = "stochastic"
    samples: List[PCSample] = field(default_factory=list)
    dispatches: List[KernelDispatch] = field(default_factory=list)


class PCSamplingWrapper:
    """
    Wrapper around rocprofv3 PC sampling.

    Supports two methods:
      - stochastic: Hardware-based, cycle-accurate, zero skid. Provides stall
        reasons, instruction types, wave counts. Requires MI300+ (gfx942+).
      - host_trap: Software-based, time-based (nanoseconds). Broader support
        (MI200+). No stall reasons or instruction types.
    """

    def __init__(self, timeout_seconds: Optional[int] = 0):
        self.timeout = None if timeout_seconds == 0 or timeout_seconds is None else timeout_seconds
        self._check_rocprofv3()

    def _check_rocprofv3(self):
        """Verify rocprofv3 is available and supports PC sampling."""
        try:
            result = subprocess.run(
                ["rocprofv3", "--help"], capture_output=True, timeout=5, text=True
            )
            if result.returncode != 0:
                raise RuntimeError("rocprofv3 not working correctly")
            if "pc-sampling" not in result.stdout:
                raise RuntimeError("rocprofv3 does not support PC sampling (upgrade ROCm?)")
        except FileNotFoundError:
            raise RuntimeError("rocprofv3 not found. Is ROCm installed?")

    def list_configs(self) -> str:
        """List available PC sampling configurations for all GPUs."""
        result = subprocess.run(["rocprofv3", "-L"], capture_output=True, timeout=10, text=True)
        return result.stdout + result.stderr

    def sample(
        self,
        command: str,
        interval: int = 65536,
        method: str = "stochastic",
        output_dir: Optional[Path] = None,
        cwd: Optional[str] = None,
    ) -> PCSamplingResult:
        """
        Run PC sampling on a command.

        Args:
            command: Command to profile (e.g., "./my_app" or "python3 bench.py")
            interval: Sampling interval. For stochastic: cycles, power of 2,
                      default 65536, minimum 256. For host_trap: nanoseconds,
                      default 65536. Lower = more samples but higher overhead.
            method: Sampling method - "stochastic" (MI300+, richer data) or
                    "host_trap" (MI200+, broader support)
            output_dir: Output directory (temp dir if None)
            cwd: Working directory for command execution

        Returns:
            PCSamplingResult with samples and kernel dispatches
        """
        if method not in VALID_METHODS:
            raise ValueError(f"Invalid method '{method}', must be one of {VALID_METHODS}")

        unit = "cycles" if method == "stochastic" else "time"

        if method == "stochastic":
            if interval < 256:
                logger.warning(f"Minimum interval is 256 cycles, adjusting from {interval}")
                interval = 256
            # Validate power of 2
            if interval & (interval - 1) != 0:
                next_pow2 = 1 << (interval - 1).bit_length()
                logger.warning(f"Interval must be power of 2, adjusting {interval} -> {next_pow2}")
                interval = next_pow2

        # Create output directory
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="samplex_")
            output_dir = Path(temp_dir)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            prof_cmd = [
                "rocprofv3",
                "--pc-sampling-beta-enabled",
                "--pc-sampling-method",
                method,
                "--pc-sampling-unit",
                unit,
                "--pc-sampling-interval",
                str(interval),
                "--kernel-trace",
                "--output-format",
                "csv",
                "-d",
                str(output_dir),
                "-o",
                "out",
            ]

            # NOTE: --kernel-include-regex only affects counter-collection and
            # thread-trace, NOT PC sampling. Kernel filtering is applied
            # post-collection in the API layer (api.py).

            prof_cmd.append("--")
            prof_cmd.extend(command.split())

            unit_label = "cycles" if method == "stochastic" else "ns"
            logger.info(
                f"Running {method} PC sampling (interval={interval} {unit_label}): {command}"
            )
            logger.debug(f"Command: {' '.join(prof_cmd)}")

            result = subprocess.run(
                prof_cmd, capture_output=True, timeout=self.timeout, text=True, cwd=cwd
            )

            if result.returncode != 0:
                logger.error(f"rocprofv3 failed (exit {result.returncode})")
                logger.error(f"stderr: {result.stderr}")
                raise RuntimeError(
                    f"rocprofv3 PC sampling failed (exit {result.returncode})\n"
                    f"stderr: {result.stderr}"
                )

            logger.debug(f"rocprofv3 completed, parsing output from {output_dir}")

            samples = self._parse_samples(output_dir, method)
            dispatches = self._parse_kernel_trace(output_dir)

            logger.info(f"Collected {len(samples)} samples across {len(dispatches)} dispatches")

            return PCSamplingResult(
                command=command,
                interval=interval,
                method=method,
                samples=samples,
                dispatches=dispatches,
            )

        except subprocess.TimeoutExpired:
            logger.error(f"PC sampling timed out after {self.timeout}s")
            raise

        finally:
            if output_dir.name.startswith("samplex_"):
                import shutil

                shutil.rmtree(output_dir, ignore_errors=True)

    def _parse_samples(self, output_dir: Path, method: str = "stochastic") -> List[PCSample]:
        """Parse PC sampling CSV output."""
        files = list(output_dir.glob(f"*pc_sampling_{method}*.csv"))
        if not files:
            files = list(output_dir.glob("*pc_sampling*.csv"))
        if not files:
            logger.warning(f"No PC sampling output file found in {output_dir}")
            return []

        csv_file = files[0]
        samples = []

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    sample = PCSample(
                        timestamp=int(row["Sample_Timestamp"]),
                        exec_mask=int(row["Exec_Mask"]),
                        dispatch_id=int(row["Dispatch_Id"]),
                        instruction=row["Instruction"],
                        instruction_comment=row.get("Instruction_Comment", ""),
                        correlation_id=int(row["Correlation_Id"]),
                        wave_issued=row.get("Wave_Issued_Instruction", "0") == "1",
                        instruction_type=row.get("Instruction_Type", ""),
                        stall_reason=row.get("Stall_Reason", ""),
                        wave_count=int(row.get("Wave_Count", 0)),
                    )
                    samples.append(sample)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse sample row: {e}")
                    continue

        return samples

    def _parse_kernel_trace(self, output_dir: Path) -> List[KernelDispatch]:
        """Parse kernel trace CSV output."""
        files = list(output_dir.glob("*kernel_trace*.csv"))
        if not files:
            return []

        csv_file = files[0]
        dispatches = []

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    dispatch = KernelDispatch(
                        dispatch_id=int(row["Dispatch_Id"]),
                        kernel_name=row["Kernel_Name"],
                        agent_id=row.get("Agent_Id", ""),
                        start_timestamp=int(row["Start_Timestamp"]),
                        end_timestamp=int(row["End_Timestamp"]),
                        workgroup_size=(
                            int(row.get("Workgroup_Size_X", 0)),
                            int(row.get("Workgroup_Size_Y", 0)),
                            int(row.get("Workgroup_Size_Z", 0)),
                        ),
                        grid_size=(
                            int(row.get("Grid_Size_X", 0)),
                            int(row.get("Grid_Size_Y", 0)),
                            int(row.get("Grid_Size_Z", 0)),
                        ),
                        vgpr_count=int(row.get("VGPR_Count", 0)),
                        sgpr_count=int(row.get("SGPR_Count", 0)),
                    )
                    dispatches.append(dispatch)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse kernel trace row: {e}")
                    continue

        return dispatches
