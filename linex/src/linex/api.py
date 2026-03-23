# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Linex API - Source-level GPU performance profiling
"""

import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .distributed import DistributedContext, detect_distributed_context, normalize_command_argv


@dataclass
class InstructionData:
    """
    Single ISA instruction with performance metrics.

    Attributes:
        isa: ISA instruction text (e.g., "s_load_dwordx2 s[0:1], s[0:1], 0x0")
        instruction_index: Instruction index in the kernel
        source_location: Source location string (e.g., "file.hip:123"). Without -g
            this is typically "" or a placeholder (e.g. ";"), so ``file``/``line`` are "" and 0.
        code_object_id: Code object ID
        instruction_address: Instruction virtual address in GPU memory
        execution_count: Number of times this instruction was executed
        latency_cycles: Total GPU cycles consumed by this instruction
        stall_cycles: GPU cycles spent stalled/waiting (e.g., for memory, dependencies)
        idle_cycles: GPU cycles spent idle

    """

    isa: str
    instruction_index: int
    source_location: str
    code_object_id: int
    instruction_address: int
    execution_count: int
    latency_cycles: int
    stall_cycles: int
    idle_cycles: int

    @property
    def file(self) -> str:
        """Source file path parsed from source_location."""
        if ":" in self.source_location:
            return self.source_location.rsplit(":", 1)[0]
        return self.source_location

    @property
    def line(self) -> int:
        """Source line number parsed from source_location. 0 when no debug info (-g)."""
        if ":" in self.source_location:
            try:
                return int(self.source_location.rsplit(":", 1)[1])
            except ValueError:
                return 0
        return 0

    @property
    def stall_percent(self) -> float:
        """Stall percentage (stall_cycles / latency_cycles)."""
        return 100.0 * self.stall_cycles / self.latency_cycles if self.latency_cycles > 0 else 0.0


@dataclass
class SourceLine:
    """
    Aggregated performance metrics for a source code line.

    Multiple ISA instructions can map to the same source line, so this
    aggregates their metrics together.

    Attributes:
        file: Source file path
        line_number: Source line number
        source_location: Full source location string (e.g., "file.hip:123")
        execution_count: Total execution count across all instructions
        total_cycles: Total GPU cycles consumed by all instructions on this line
        stall_cycles: Total GPU cycles spent stalled/waiting
        idle_cycles: Total GPU cycles spent idle
        instructions: List of ISA instructions that map to this source line
    """

    file: str
    line_number: int
    source_location: str
    execution_count: int
    total_cycles: int
    stall_cycles: int
    idle_cycles: int
    instructions: List[InstructionData]

    @property
    def stall_percent(self) -> float:
        """Stall percentage (stall_cycles / total_cycles)."""
        return 100.0 * self.stall_cycles / self.total_cycles if self.total_cycles > 0 else 0.0


@dataclass
class RankProfile:
    """Per-rank trace data produced by a Linex profiling run."""

    rank_key: str
    global_rank: int
    local_rank: int
    world_size: int
    hostname: str
    launcher: str
    ui_output_dir: str
    source_lines: List[SourceLine]
    instructions: List[InstructionData]


class Linex:
    """
    Linex - Source-Level GPU Performance Profiler

    Maps GPU performance metrics (cycles, stalls, etc.) to source code lines.
    Automatically uses temp directories for profiling output.

    Example:
        profiler = Linex()
        profiler.profile("./my_app", kernel_filter="my_kernel")

        # Analyze source lines
        for line in profiler.source_lines:
            print(f"{line.file}:{line.line_number}")
            print(f"  Total: {line.total_cycles:,} cycles")
            print(f"  Stall: {line.stall_cycles:,} ({line.stall_percent:.1f}%)")
    """

    # Default rocprofv3 parameters
    DEFAULT_DECODER_URL = "https://raw.githubusercontent.com/ROCm/rocprof-trace-decoder/12c9d5871b3f803937ef92f1adce9294dfc549a7/releases/linux_glibc_2_28_x86_64/librocprof-trace-decoder.so"
    DEFAULT_DECODER_PATH = Path.home() / ".cache" / "linex" / "librocprof-trace-decoder.so"

    def __init__(
        self,
        target_cu: int = 0,
        shader_engine_mask: str = "0xFFFFFFFF",
        activity: int = 10,
    ):
        """
        Initialize Linex profiler.

        Args:
            target_cu: Target CU for detailed trace (default: 0)
            shader_engine_mask: Shader engine mask (default: 0xFFFFFFFF for all)
            activity: Activity counter polling interval (default: 10)
        """
        self.decoder_lib = self._ensure_decoder()
        self.target_cu = target_cu
        self.shader_engine_mask = shader_engine_mask
        self.activity = activity

        # Data storage
        self._instructions: List[InstructionData] = []
        self._source_lines: Dict[str, SourceLine] = {}
        self._rank_profiles: Dict[str, RankProfile] = {}
        self._distributed_context: DistributedContext = DistributedContext()

    def _ensure_decoder(self) -> Path:
        """Ensure decoder library is available, download if needed."""
        if self.DEFAULT_DECODER_PATH.exists():
            return self.DEFAULT_DECODER_PATH

        # Download decoder library silently
        self.DEFAULT_DECODER_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(self.DEFAULT_DECODER_URL, self.DEFAULT_DECODER_PATH)
        self.DEFAULT_DECODER_PATH.chmod(0o755)

        return self.DEFAULT_DECODER_PATH

    def profile(
        self,
        command: str | Sequence[str],
        output_dir: Optional[str] = None,
        kernel_filter: Optional[str] = None,
        force_cu_mask: bool = True,
        env: Optional[Dict[str, str]] = None,
        launcher: Optional[str | Sequence[str]] = None,
    ) -> "Linex":
        """
        Profile an application and collect source-level performance data.

        ISA instructions and cycle/stall metrics are always produced. Source-line
        mapping (file:line per instruction) requires the binary to be built with
        debug symbols (-g); without -g, ``instructions`` is still populated but
        ``source_lines`` may be empty.

        Args:
            command: Command to profile (e.g., "./my_app" or "./my_app arg1 arg2")
            output_dir: Output directory for traces (default: temp directory)
            kernel_filter: Regex filter for kernel names (default: None = all kernels)
            force_cu_mask: Force waves to target CU using HSA_CU_MASK (default: True)

        Returns:
            self for chaining
        """
        import tempfile

        # Use temp directory if not specified
        if output_dir is None:
            base_output_dir = Path(tempfile.mkdtemp(prefix="linex_"))
        else:
            base_output_dir = Path(output_dir)

        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        dist_context = detect_distributed_context(run_env)
        self._distributed_context = dist_context

        output_path = base_output_dir
        if dist_context.is_distributed:
            output_path = base_output_dir / dist_context.rank_tag
        output_path.mkdir(parents=True, exist_ok=True)

        command_argv = normalize_command_argv(command)

        cmd = [
            "rocprofv3",
            "--att",
            "--att-library-path",
            str(self.decoder_lib.parent),
            "--att-activity",
            str(self.activity),
            "--att-target-cu",
            str(self.target_cu),
            "--att-shader-engine-mask",
            self.shader_engine_mask,
            "-d",
            str(output_path),
        ]

        if kernel_filter:
            cmd.extend(["--kernel-include-regex", kernel_filter])

        cmd.extend(["--", *command_argv])

        # If a launcher is specified, prepend it: launcher rocprofv3 ... -- app
        if launcher is not None:
            launcher_argv = normalize_command_argv(launcher)
            cmd = launcher_argv + cmd

        if force_cu_mask and "HSA_CU_MASK" not in run_env:
            run_env["HSA_CU_MASK"] = "0x1"  # Force to CU 0 unless caller already set a mask

        result = subprocess.run(cmd, env=run_env, capture_output=True, text=True)

        if result.returncode != 0:
            # Include stderr in error message for debugging
            raise RuntimeError(f"rocprofv3 failed with code {result.returncode}\n{result.stderr}")

        # Find generated ui_output directory
        ui_dirs = sorted(output_path.glob("ui_output_*"), key=lambda p: p.name)

        if not ui_dirs:
            raise RuntimeError(f"No ui_output directories found in {output_path}")

        self._rank_profiles = {}
        for idx, ui_dir in enumerate(ui_dirs):
            instructions, source_lines = self._load_ui_output_data(ui_dir)
            if idx == 0:
                rank_key = dist_context.rank_tag
            else:
                rank_key = f"{dist_context.rank_tag}_dispatch{idx + 1:03d}"
            self._rank_profiles[rank_key] = RankProfile(
                rank_key=rank_key,
                global_rank=dist_context.global_rank,
                local_rank=dist_context.local_rank,
                world_size=dist_context.world_size,
                hostname=dist_context.hostname,
                launcher=dist_context.launcher,
                ui_output_dir=str(ui_dir),
                source_lines=sorted(source_lines.values(), key=lambda x: x.total_cycles, reverse=True),
                instructions=instructions,
            )

        # Preserve existing API behavior by exposing the first rank profile as top-level fields.
        primary_rank = next(iter(self._rank_profiles.values()))
        self._instructions = primary_rank.instructions
        self._source_lines = {
            line.source_location: line for line in primary_rank.source_lines if line.source_location
        }
        return self

    def _load_ui_output_data(
        self, ui_output_dir: Path
    ) -> tuple[List[InstructionData], Dict[str, SourceLine]]:
        """Internal: Load performance trace data from ui_output directory."""
        code_file = ui_output_dir / "code.json"

        if not code_file.exists():
            raise FileNotFoundError(f"code.json not found in {ui_output_dir}")

        with open(code_file, "r") as f:
            data = json.load(f)

        if data["code"] is None or len(data["code"]) == 0:
            raise ValueError(
                "No code traced - target CU had no waves. Try larger workload or HSA_CU_MASK=0x1"
            )

        # Parse instructions
        instructions: List[InstructionData] = []
        for entry in data["code"]:
            inst = InstructionData(
                isa=entry[0],
                instruction_index=entry[2],
                source_location=entry[3],
                code_object_id=entry[4],
                instruction_address=entry[5],
                execution_count=entry[6],
                latency_cycles=entry[7],
                stall_cycles=entry[8],
                idle_cycles=entry[9],
            )
            instructions.append(inst)

        return instructions, self._aggregate_source_lines(instructions)

    def _load_ui_output(self, ui_output_dir: Path) -> None:
        """Backward-compatible loader for a single ui_output directory."""
        instructions, source_lines = self._load_ui_output_data(ui_output_dir)
        self._instructions = instructions
        self._source_lines = source_lines

    def _aggregate_source_lines(self, instructions: Optional[List[InstructionData]] = None):
        """Aggregate instruction data by source line."""
        source_lines: Dict[str, SourceLine] = {}
        instructions_to_aggregate = self._instructions if instructions is None else instructions

        for inst in instructions_to_aggregate:
            source = inst.source_location
            if not source or source.startswith(";"):
                continue

            if source not in source_lines:
                # Parse file:line from source
                if ":" in source:
                    parts = source.rsplit(":", 1)
                    file = parts[0]
                    try:
                        line = int(parts[1])
                    except ValueError:
                        file = source
                        line = 0
                else:
                    file = source
                    line = 0

                source_lines[source] = SourceLine(
                    file=file,
                    line_number=line,
                    source_location=source,
                    execution_count=0,
                    total_cycles=0,
                    stall_cycles=0,
                    idle_cycles=0,
                    instructions=[],
                )

            sl = source_lines[source]
            sl.execution_count += inst.execution_count
            sl.total_cycles += inst.latency_cycles
            sl.stall_cycles += inst.stall_cycles
            sl.idle_cycles += inst.idle_cycles
            sl.instructions.append(inst)
        if instructions is None:
            self._source_lines = source_lines
        return source_lines

    @property
    def source_lines(self) -> List[SourceLine]:
        """Get all source lines sorted by total_cycles (hottest first)."""
        return sorted(self._source_lines.values(), key=lambda x: x.total_cycles, reverse=True)

    @property
    def instructions(self) -> List[InstructionData]:
        """Get all instructions."""
        return self._instructions

    @property
    def rank_profiles(self) -> Dict[str, RankProfile]:
        """Get per-rank profiles for distributed runs."""
        return self._rank_profiles

    @property
    def distributed_context(self) -> DistributedContext:
        """Get distributed runtime metadata detected for this profile run."""
        return self._distributed_context
