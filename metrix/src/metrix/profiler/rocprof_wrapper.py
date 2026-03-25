"""
ROCProfiler V3 wrapper
Clean, robust interface - regex-free CSV parsing (uses the csv module).
"""

import re
import subprocess
import tempfile
import textwrap
import csv
import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Sequence

# Import ProfileResult from backends to avoid duplication
from ..backends.base import ProfileResult
from ..logger import logger
from ..utils.distributed import detect_distributed_context, normalize_command_argv


# Python wrapper script that torchrun launches per-worker.  Each worker
# reads its RANK / LOCAL_RANK from the environment (set by torchrun),
# creates a rank-specific output directory, then runs rocprofv3 on the
# actual user command.
#
# Argv layout:
#   wrapper.py <base_output_dir> [rocprof_args...] -- <user_command...>
_RANK_WRAPPER_SCRIPT = textwrap.dedent(
    """\
    import os, sys, subprocess, shlex, yaml, copy

    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))

    try:
        sep_idx = sys.argv.index("--")
        base_output_dir = sys.argv[1]
        rocprof_extra = sys.argv[2:sep_idx]
        user_cmd = sys.argv[sep_idx + 1:]
    except (ValueError, IndexError):
        print(f"Usage: {sys.argv[0]} <output_dir> [rocprof_args...] -- <command...>", file=sys.stderr)
        sys.exit(1)

    # Handle single-element user_cmd containing spaces (torchrun passes
    # quoted commands as one argv element via argparse.REMAINDER).
    if len(user_cmd) == 1 and " " in user_cmd[0]:
        user_cmd = shlex.split(user_cmd[0])

    rank_output_dir = os.path.join(base_output_dir, f"rank_{rank}")
    os.makedirs(rank_output_dir, exist_ok=True)

    # Rewrite the --input YAML so output_directory points to the rank-specific
    # dir.  This avoids the "conflicting value for output_directory" error when
    # both --input YAML and -d are provided to rocprofv3.
    rewritten_args = []
    for i, arg in enumerate(rocprof_extra):
        if arg == "--input" and i + 1 < len(rocprof_extra):
            orig_yaml = rocprof_extra[i + 1]
            rank_yaml = os.path.join(rank_output_dir, "rocprof_input.yaml")
            with open(orig_yaml) as f:
                cfg = yaml.safe_load(f)
            for job in cfg.get("jobs", []):
                job["output_directory"] = rank_output_dir
            with open(rank_yaml, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            rewritten_args.extend(["--input", rank_yaml])
        elif i > 0 and rocprof_extra[i - 1] == "--input":
            continue  # already handled
        else:
            rewritten_args.append(arg)

    cmd = ["rocprofv3"] + rewritten_args + ["--"] + user_cmd
    print(f"[Rank {rank}/{world_size}] {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, env=os.environ)
    sys.exit(result.returncode)
    """
)


class ROCProfV3Wrapper:
    """
    Clean wrapper around rocprofv3
    - No timeout by default (configurable via timeout_seconds parameter)
    - Robust CSV parsing (using csv module, NOT regex)
    - Proper error handling
    - Multi-pass profiling when counter limit is exceeded
    - Uses --input with YAML
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
            result = subprocess.run(
                ["rocprofv3", "--help"], capture_output=True, timeout=5, text=True
            )
            if result.returncode != 0:
                raise RuntimeError("rocprofv3 not working correctly")
        except FileNotFoundError:
            raise RuntimeError("rocprofv3 not found. Is ROCm installed?")
        except subprocess.TimeoutExpired:
            raise RuntimeError("rocprofv3 --help timed out after 5 seconds")

    @staticmethod
    def _needs_extra_counters(counter_defs_file: Path) -> bool:
        """Check if counter_defs defines hardware-level counters (block+event)
        that require --extra-counters for rocprofv3 to recognize them."""
        try:
            with open(counter_defs_file, "r") as f:
                data = yaml.safe_load(f)
            for counter in data.get("rocprofiler-sdk", {}).get("counters", []):
                for defn in counter.get("definitions", []):
                    if "block" in defn and "event" in defn:
                        return True
            return False
        except Exception:
            return False

    def profile(
        self,
        command: str | Sequence[str],
        counters: List[str],
        output_dir: Optional[Path] = None,
        kernel_filter: Optional[str] = None,
        cwd: Optional[str] = None,
        kernel_iteration_range: Optional[str] = None,
        extra_counters_path: Optional[Path] = None,
        arch: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        launcher: Optional[str | Sequence[str]] = None,
    ) -> List[ProfileResult]:
        """
        Profile a command with specified counters (single pass).

        Note: This wrapper only handles single-pass profiling. Multi-pass profiling
        is handled by the backend base class.

        Args:
            command: Command to profile (e.g., "./benchmark")
            counters: List of counter names to collect
            output_dir: Output directory (temp dir if None)
            kernel_filter: Optional regular expression to filter kernels by name.
                Only kernels whose names match the pattern will be included in
                profiling results. All other kernel dispatches will be ignored.

                Examples:
                  ``"^gemm.*"``        - kernels whose names start with "gemm"
                  ``".*attention.*"``   - kernels whose names contain "attention"
                  ``"gemm|attention"``  - kernels matching either pattern
            cwd: Optional working directory
            kernel_iteration_range: Optional iteration range (e.g., "[1,5]" to profile iterations 1-5)
            extra_counters_path: Path to YAML with custom counter definitions (rocprofiler-sdk: section)
            arch: GPU architecture (e.g., "gfx1201") to filter counter definitions

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
            # Find or use provided custom counter definitions
            counter_defs_file = extra_counters_path
            if counter_defs_file is None:
                backends_dir = Path(__file__).resolve().parent.parent / "backends"
                if backends_dir.exists():
                    counter_defs_files = list(backends_dir.glob("counter_defs*.yaml"))
                else:
                    counter_defs_files = []

                if counter_defs_files:
                    counter_defs_file = counter_defs_files[0]
                    logger.debug(f"Using custom counter definitions: {counter_defs_file.name}")

            # Create rocprofv3 input YAML file (jobs section + rocprofiler-sdk section)
            input_yaml = self._create_input_yaml(
                counters,
                output_dir,
                kernel_filter,
                kernel_iteration_range,
                counter_defs_file,
                arch=arch,
            )

            # Build rocprofv3 arguments (shared between direct and wrapper modes)
            rocprof_args = []

            if not counters:
                rocprof_args.append("--kernel-trace")

            rocprof_args.extend(["--input", str(input_yaml)])

            if counter_defs_file and self._needs_extra_counters(counter_defs_file):
                rocprof_args.extend(["--extra-counters", str(counter_defs_file)])
                logger.debug(f"Using --extra-counters: {counter_defs_file.name}")

            if kernel_filter:
                rocprof_args.extend(["--kernel-include-regex", kernel_filter])

            command_argv = normalize_command_argv(command)

            run_env = os.environ.copy()
            if env:
                run_env.update(env)

            use_wrapper = launcher is not None

            if use_wrapper:
                # Per-rank wrapper mode: the launcher (e.g. torchrun) spawns
                # one Python wrapper per worker.  Each wrapper reads RANK from
                # its environment and runs its own rocprofv3 with a rank-
                # specific output directory.
                wrapper_path = output_dir / "_metrix_rank_wrapper.py"
                wrapper_path.write_text(_RANK_WRAPPER_SCRIPT)

                launcher_argv = normalize_command_argv(launcher)

                # Build: launcher... wrapper.py <output_dir> [rocprof_args...] -- command...
                # Note: no explicit "python3" — torchrun and similar launchers
                # already invoke the script with the Python interpreter.
                prof_cmd = (
                    launcher_argv
                    + [str(wrapper_path), str(output_dir)]
                    + rocprof_args
                    + ["--"]
                    + command_argv
                )
            else:
                # Direct mode: rocprofv3 [options] -- command...
                prof_cmd = ["rocprofv3"] + rocprof_args + ["-d", str(output_dir)]
                prof_cmd.append("--")
                prof_cmd.extend(command_argv)

            logger.debug(f"Profile command: {' '.join(prof_cmd)}")
            logger.info(f"Starting profiling with {len(counters)} counters (wrapper={use_wrapper})")
            logger.debug(f"Output directory: {output_dir}")
            logger.info(f"subprocess will run from: {cwd if cwd else os.getcwd()}")

            # Run profiling
            result = subprocess.run(
                prof_cmd,
                capture_output=True,
                timeout=self.timeout,
                text=True,
                cwd=cwd,
                env=run_env,
            )

            logger.info(f"Profiling completed (exit code: {result.returncode})")
            logger.debug(f"stdout length: {len(result.stdout)} chars")
            logger.debug(f"stderr length: {len(result.stderr)} chars")

            if result.returncode != 0:
                logger.error(f"Profiling failed with exit code {result.returncode}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                raise RuntimeError(
                    f"Profiling failed with exit code {result.returncode}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            if result.stdout:
                logger.debug(f"stdout: {result.stdout[:500]}")
            if result.stderr:
                logger.debug(f"stderr: {result.stderr[:500]}")

            # Parse output — either per-rank subdirectories or single output dir
            results: List[ProfileResult] = []

            if use_wrapper:
                # Collect per-rank results from rank_N/ subdirectories
                rank_dirs = sorted(output_dir.glob("rank_*"))
                if not rank_dirs:
                    raise RuntimeError(
                        f"No rank_* subdirectories found in {output_dir}. "
                        "The launcher wrapper may not have executed."
                    )

                logger.info(f"Found {len(rank_dirs)} rank output directories")
                for rank_dir in rank_dirs:
                    rank_num = int(rank_dir.name.split("_")[1])
                    try:
                        rank_results = self._parse_output(rank_dir)
                    except RuntimeError:
                        logger.warning(f"No output CSV in {rank_dir}, skipping")
                        continue

                    for pr in rank_results:
                        pr.global_rank = rank_num
                        pr.local_rank = rank_num  # Approximation for single-node
                        pr.world_size = len(rank_dirs)
                        pr.launcher = "torchrun"

                    results.extend(rank_results)
                    logger.info(f"  rank_{rank_num}: {len(rank_results)} dispatch(es)")
            else:
                results = self._parse_output(output_dir)
                logger.info(f"Parsed {len(results)} kernel dispatch(es)")

            # Post-filter in timing-only mode (kernel-trace ignores regex)
            if not counters and kernel_filter and results:
                try:
                    pattern = re.compile(kernel_filter)
                except re.error as exc:
                    raise RuntimeError(
                        f"Invalid kernel_filter regex '{kernel_filter}': {exc}"
                    ) from exc
                before = len(results)
                results = [r for r in results if pattern.search(r.kernel_name)]
                logger.info(f"Filtered {before} -> {len(results)} dispatches by kernel_filter")

            # Populate distributed context for non-wrapper mode
            if not use_wrapper:
                dist_context = detect_distributed_context(run_env)
                for profile_result in results:
                    profile_result.global_rank = dist_context.global_rank
                    profile_result.local_rank = dist_context.local_rank
                    profile_result.world_size = dist_context.world_size
                    profile_result.node_rank = dist_context.node_rank
                    profile_result.hostname = dist_context.hostname
                    profile_result.launcher = dist_context.launcher

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

    def _create_input_yaml(
        self,
        counters: List[str],
        output_dir: Path,
        kernel_filter: Optional[str] = None,
        kernel_iteration_range: Optional[str] = None,
        counter_defs_file: Optional[Path] = None,
        arch: Optional[str] = None,
    ) -> Path:
        """
        Create rocprofv3 input YAML file with jobs section + rocprofiler-sdk section.


        Args:
            counters: List of counter names to collect
            output_dir: Output directory
            kernel_filter: Optional kernel filter regex
            kernel_iteration_range: Optional iteration range
            counter_defs_file: Optional path to counter definitions YAML
            arch: GPU architecture to filter counter definitions by

        Returns:
            Path to created input YAML file
        """
        input_file = output_dir / "rocprof_input.yaml"

        # Build the YAML structure
        yaml_content = {}

        # Load custom counter definitions if available
        if counter_defs_file and counter_defs_file.exists():
            logger.debug(f"Loading counter definitions from {counter_defs_file}")
            with open(counter_defs_file, "r") as f:
                counter_defs = yaml.safe_load(f)
                if "rocprofiler-sdk" in counter_defs:
                    sdk_section = counter_defs["rocprofiler-sdk"].copy()
                    if "counters" in sdk_section:
                        filtered_counters = []
                        for counter in sdk_section["counters"]:
                            should_include = False
                            if "definitions" in counter:
                                arch_matched_defs = []
                                for defn in counter["definitions"]:
                                    if arch:
                                        archs = defn.get("architectures", [])
                                        if archs and arch not in archs:
                                            continue
                                        # Strip non-matching architectures so rocprofv3
                                        # doesn't try to build ASTs for other GPUs
                                        if archs and len(archs) > 1:
                                            defn = dict(defn)
                                            defn["architectures"] = [arch]
                                    if "expression" in defn or (
                                        "block" in defn and "event" in defn
                                    ):
                                        should_include = True
                                        arch_matched_defs.append(defn)
                                if should_include and arch_matched_defs:
                                    counter = dict(counter)
                                    counter["definitions"] = arch_matched_defs
                            if should_include:
                                filtered_counters.append(counter)
                        sdk_section["counters"] = filtered_counters
                    yaml_content["rocprofiler-sdk"] = sdk_section
                    logger.debug(
                        f"Loaded {len(sdk_section.get('counters', []))} counter definitions for arch={arch} (excluded builtin and non-matching counters)"
                    )

        # Create jobs section
        job = {
            "kernel_include_regex": kernel_filter if kernel_filter else ".*",
            "output_file": "out",
            "output_directory": str(output_dir),
            "output_format": ["csv", "json"],
            "truncate_kernels": True,
        }

        if kernel_iteration_range:
            job["kernel_iteration_range"] = kernel_iteration_range

        if counters:
            job["pmc"] = counters
        else:
            # Timing-only mode: empty pmc list (rocprofv3 will just trace kernels)
            job["pmc"] = []

        yaml_content["jobs"] = [job]

        # Write YAML file
        with open(input_file, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        logger.debug(f"Created input YAML: {input_file}")
        logger.debug(f"YAML content:\n{yaml.dump(yaml_content, default_flow_style=False)[:500]}")

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

                except (KeyError, ValueError, TypeError) as e:
                    # TypeError: int(None) from multi-process traces with null fields
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

                except (KeyError, ValueError, TypeError) as e:
                    # TypeError: int(None) from multi-process traces with null fields
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
