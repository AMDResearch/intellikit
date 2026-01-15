"""
ROCProfiler V3 wrapper
Clean, robust interface - NO REGEX, proper CSV parsing!
"""

import subprocess
import tempfile
import csv
import os
from pathlib import Path
from typing import List, Dict, Optional

# Import ProfileResult from backends to avoid duplication
from ..backends.base import ProfileResult
from ..logger import logger


class ROCProfV3Wrapper:
    """
    Clean wrapper around rocprofv3
    - No timeout by default (configurable via timeout_seconds parameter)
    - Robust CSV parsing (using csv module, NOT regex)
    - Proper error handling
    - Multi-pass profiling when counter limit is exceeded
    """

    # Maximum counters per pass (conservative limit for gfx942/MI300)
    MAX_COUNTERS_PER_PASS = 14

    def __init__(self, timeout_seconds: Optional[int] = 0):
        """
        Args:
            timeout_seconds: Timeout in seconds for profiling (0 or None for no timeout)
        """
        # Convert 0 to None for "no timeout" (subprocess.run treats 0 as immediate timeout)
        self.timeout = None if timeout_seconds == 0 or timeout_seconds is None else timeout_seconds
        self._check_rocprofv3()

    def _check_rocprofv3(self):
        """Verify rocprofv3 is available"""
        try:
            # Always use a fixed 5s timeout for the --help check, regardless of profiling timeout
            result = subprocess.run(["rocprofv3", "--help"], capture_output=True, timeout=5, text=True)
            if result.returncode != 0:
                raise RuntimeError("rocprofv3 not working correctly")
        except FileNotFoundError:
            raise RuntimeError("rocprofv3 not found. Is ROCm installed?")
        except subprocess.TimeoutExpired:
            raise RuntimeError("rocprofv3 --help timed out after 5 seconds")

    def profile(
        self,
        command: str,
        counters: List[str],
        output_dir: Optional[Path] = None,
        kernel_filter: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> List[ProfileResult]:
        """
        Profile a command with specified counters (single pass).

        Note: This wrapper only handles single-pass profiling. Multi-pass profiling
        is handled by the backend base class.

        Args:
            command: Command to profile (e.g., "./benchmark")
            counters: List of counter names to collect
            output_dir: Output directory (temp dir if None)
            kernel_filter: Optional kernel name filter
            cwd: Optional working directory

        Returns:
            List of ProfileResult objects, one per dispatch

        Raises:
            subprocess.TimeoutExpired: If profiling exceeds timeout
            RuntimeError: If profiling fails
        """

        # Create temp directory if needed
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="metrix_")
            output_dir = Path(temp_dir)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Build rocprofv3 command
            prof_cmd = ["rocprofv3"]

            if counters:
                # Counter collection mode
                prof_cmd.extend(["--pmc", ",".join(counters)])
            else:
                # Timing-only mode: use kernel tracing
                prof_cmd.append("--kernel-trace")

            prof_cmd.extend(["-d", str(output_dir), "--output-format", "csv"])

            # Add kernel filter if specified (simple substring match)
            if kernel_filter:
                prof_cmd.extend(["--kernel-include", kernel_filter])

            # Add target command
            prof_cmd.append("--")
            prof_cmd.extend(command.split())

            logger.debug(f"rocprofv3 command: {' '.join(prof_cmd)}")
            logger.info(f"Starting rocprofv3 with {len(counters)} counters")
            logger.debug(f"Python process cwd: {os.getcwd()}")
            logger.debug(f"Output directory: {output_dir}")
            logger.info(f"subprocess will run from: {cwd if cwd else os.getcwd()}")

            # Run profiling with timeout
            logger.info("Calling subprocess.run...")
            logger.debug(f"Full command: {' '.join(prof_cmd)}")
            logger.debug(f"Timeout: {self.timeout}")
            logger.debug(f"CWD: {cwd}")

            result = subprocess.run(prof_cmd, capture_output=True, timeout=self.timeout, text=True, cwd=cwd)
            logger.info("subprocess.run returned!")

            logger.info("subprocess.run returned successfully")
            logger.debug(f"Exit code: {result.returncode}")
            logger.debug(f"stdout length: {len(result.stdout)} chars")
            logger.debug(f"stderr length: {len(result.stderr)} chars")

            logger.info(f"rocprofv3 completed (exit code: {result.returncode})")

            if result.returncode != 0:
                logger.error(f"rocprofv3 failed with exit code {result.returncode}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                raise RuntimeError(
                    f"rocprofv3 failed with exit code {result.returncode}\\n"
                    f"stdout: {result.stdout}\\n"
                    f"stderr: {result.stderr}"
                )

            # Log stdout/stderr even on success for debugging
            if result.stdout:
                logger.debug(f"rocprofv3 stdout: {result.stdout[:500]}")  # First 500 chars
            if result.stderr:
                logger.debug(f"rocprofv3 stderr: {result.stderr[:500]}")

            # Parse output CSV files
            logger.debug(f"Parsing CSV files from {output_dir}")
            results = self._parse_output(output_dir)
            logger.info(f"Successfully parsed {len(results)} kernel dispatch(es)")

            return results

        except subprocess.TimeoutExpired:
            # This exception can only occur if self.timeout was set (not None)
            logger.error(f"Profiling timed out after {self.timeout} seconds")
            raise subprocess.TimeoutExpired(cmd=command, timeout=self.timeout)

        finally:
            # Cleanup temp directory if we created it
            if output_dir.name.startswith("metrix_"):
                import shutil

                shutil.rmtree(output_dir, ignore_errors=True)

    def _create_input_file(self, counters: List[str], output_dir: Path) -> Path:
        """
        Create rocprofv3 input file
        Format: Simple text file with counter names
        """
        input_file = output_dir / "input.txt"

        # rocprofv3 input format: one counter per line, or comma-separated
        # Let's use comma-separated on one line
        content = "pmc: " + " ".join(counters) + "\\n"

        with open(input_file, "w") as f:
            f.write(content)

        return input_file

    def _parse_output(self, output_dir: Path) -> List[ProfileResult]:
        """
        Parse rocprofv3 CSV output
        Uses csv module - NO REGEX!
        """

        # Try counter collection first
        counter_files = list(output_dir.glob("*/*_counter_collection.csv"))
        if not counter_files:
            counter_files = list(output_dir.glob("*_counter_collection.csv"))

        # If no counter file, try kernel trace (for timing-only mode)
        if not counter_files:
            trace_files = list(output_dir.glob("*/*_kernel_trace.csv"))
            if not trace_files:
                trace_files = list(output_dir.glob("*_kernel_trace.csv"))

            if trace_files:
                return self._parse_kernel_trace(trace_files[0])

            raise RuntimeError(f"No output CSV found in {output_dir}")

        csv_file = counter_files[0]

        # rocprofv3 format: each counter is a separate row
        # We need to group by dispatch_id to collect all counters

        dispatches = {}  # dispatch_id -> {kernel info, counters dict}

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    dispatch_id = int(row["Dispatch_Id"])

                    # Initialize dispatch entry if first time seeing it
                    if dispatch_id not in dispatches:
                        dispatches[dispatch_id] = {
                            "kernel_name": row["Kernel_Name"],
                            "agent_id": row["Agent_Id"],
                            "start_ts": int(row["Start_Timestamp"]),
                            "end_ts": int(row["End_Timestamp"]),
                            "grid_size": int(row["Grid_Size"]),
                            "workgroup_size": int(row["Workgroup_Size"]),
                            "lds": int(row.get("LDS_Block_Size", 0)),
                            "vgpr": int(row.get("VGPR_Count", 0)),
                            "accum_vgpr": int(row.get("Accum_VGPR_Count", 0)),
                            "sgpr": int(row.get("SGPR_Count", 0)),
                            "counters": {},
                        }

                    # Add counter value
                    counter_name = row["Counter_Name"]
                    counter_value = float(row["Counter_Value"])
                    dispatches[dispatch_id]["counters"][counter_name] = counter_value

                except (KeyError, ValueError) as e:
                    print(f"Warning: Failed to parse row: {e}: {row}")
                    continue

        # Convert to ProfileResult objects
        results = []
        for dispatch_id, dispatch_data in dispatches.items():
            # Convert grid/workgroup size to tuple format (x, 1, 1)
            # rocprofv3 reports total threads, we'll put it in x dimension
            grid_size = (dispatch_data["grid_size"], 1, 1)
            workgroup_size = (dispatch_data["workgroup_size"], 1, 1)

            duration_ns = dispatch_data["end_ts"] - dispatch_data["start_ts"]

            result = ProfileResult(
                dispatch_id=dispatch_id,
                kernel_name=dispatch_data["kernel_name"],
                gpu_id=dispatch_data["agent_id"],
                duration_ns=duration_ns,
                grid_size=grid_size,
                workgroup_size=workgroup_size,
                counters=dispatch_data["counters"],
                lds_per_workgroup=dispatch_data["lds"],
                arch_vgpr=dispatch_data["vgpr"],
                accum_vgpr=dispatch_data["accum_vgpr"],
                sgpr=dispatch_data["sgpr"],
            )
            results.append(result)

        return results

    def _parse_kernel_trace(self, csv_file: Path) -> List[ProfileResult]:
        """
        Parse kernel trace CSV (timing-only mode)
        Format: Kernel_Name,Start_Timestamp,End_Timestamp,Grid_Size,Workgroup_Size,...
        """
        dispatches = {}

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Extract basic info
                    kernel_name = row["Kernel_Name"]
                    start_ts = int(row["Start_Timestamp"])
                    end_ts = int(row["End_Timestamp"])
                    grid_size_val = int(row.get("Grid_Size", 0))
                    workgroup_size_val = int(row.get("Workgroup_Size", 256))

                    # Create unique dispatch ID
                    dispatch_id = len(dispatches)

                    # Create result with no counters (timing only)
                    result = ProfileResult(
                        dispatch_id=dispatch_id,
                        kernel_name=kernel_name,
                        gpu_id=0,
                        duration_ns=end_ts - start_ts,
                        grid_size=(grid_size_val, 1, 1),
                        workgroup_size=(workgroup_size_val, 1, 1),
                        counters={},  # No counters in timing-only mode
                    )
                    dispatches[dispatch_id] = result

                except (KeyError, ValueError) as e:
                    print(f"Warning: Failed to parse trace row: {e}")
                    continue

        return list(dispatches.values())

    def _parse_csv_row(self, row: Dict[str, str]) -> ProfileResult:
        """
        Parse single CSV row into ProfileResult
        Clean, explicit parsing - NO REGEX!
        """

        # Extract basic info
        dispatch_id = int(row["Dispatch_ID"])
        kernel_name = row["Kernel_Name"]
        gpu_id = int(row["GPU_ID"])

        # Parse timing (start/end timestamps)
        start_ts = int(row["Start_Timestamp"])
        end_ts = int(row["End_Timestamp"])
        duration_ns = end_ts - start_ts

        # Parse grid/workgroup sizes
        # Format: "x,y,z" or "x y z" - handle both
        grid_str = row["Grid_Size"].replace(",", " ")
        grid_parts = grid_str.split()
        grid_size = tuple(int(x) for x in grid_parts[:3])

        wg_str = row["Workgroup_Size"].replace(",", " ")
        wg_parts = wg_str.split()
        workgroup_size = tuple(int(x) for x in wg_parts[:3])

        # Parse kernel resources
        lds_per_wg = int(row.get("LDS_Per_Workgroup", 0))
        arch_vgpr = int(row.get("Arch_VGPR", 0))
        accum_vgpr = int(row.get("Accum_VGPR", 0))
        sgpr = int(row.get("SGPR", 0))

        # Extract all counter values
        # Skip known metadata columns
        metadata_cols = {
            "Dispatch_ID",
            "Kernel_Name",
            "GPU_ID",
            "Grid_Size",
            "Workgroup_Size",
            "LDS_Per_Workgroup",
            "Scratch_Per_Workitem",
            "Arch_VGPR",
            "Accum_VGPR",
            "SGPR",
            "wave_size",
            "obj",
            "Start_Timestamp",
            "End_Timestamp",
        }

        counters = {}
        for key, value in row.items():
            if key not in metadata_cols:
                # Try to parse as float
                try:
                    counters[key] = float(value)
                except ValueError:
                    # Keep as string if not numeric
                    counters[key] = value

        return ProfileResult(
            dispatch_id=dispatch_id,
            kernel_name=kernel_name,
            gpu_id=gpu_id,
            duration_ns=duration_ns,
            grid_size=grid_size,
            workgroup_size=workgroup_size,
            counters=counters,
            lds_per_workgroup=lds_per_wg,
            arch_vgpr=arch_vgpr,
            accum_vgpr=accum_vgpr,
            sgpr=sgpr,
        )
