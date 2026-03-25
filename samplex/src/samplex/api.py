"""
High-level Samplex API - Stochastic PC sampling made simple.

    sampler = Samplex()
    results = sampler.sample("./my_app")
    for kernel in results.kernels:
        print(f"{kernel.name}: {kernel.top_instructions[0]}")
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .profiler.rocprof_wrapper import PCSamplingWrapper, PCSample, KernelDispatch
from .logger import logger


@dataclass
class InstructionHotspot:
    """A frequently sampled instruction (hotspot)."""

    instruction: str
    opcode: str
    sample_count: int
    percentage: float
    issued_count: int = 0
    stalled_count: int = 0
    stall_reasons: Dict[str, int] = field(default_factory=dict)
    instruction_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class KernelSamplingResult:
    """PC sampling results for a single kernel."""

    name: str
    dispatch_ids: List[int]
    total_samples: int
    duration_us: float
    top_instructions: List[InstructionHotspot]
    full_mask_pct: float = 0.0
    empty_instruction_count: int = 0
    issued_pct: float = 0.0
    top_stall_reasons: Optional[Dict[str, float]] = None


@dataclass
class SamplingResults:
    """Complete results from a samplex run."""

    command: str
    total_samples: int
    total_dispatches: int
    interval: int
    kernels: List[KernelSamplingResult]
    global_top_opcodes: List[InstructionHotspot]


class Samplex:
    """
    High-level stochastic PC sampling API.

    Uses hardware-based sampling with cycle-accurate precision and zero skid.
    Provides stall reasons, instruction types, and wave counts.
    Requires MI300+ (gfx942 and later).

    Usage:
        sampler = Samplex()
        results = sampler.sample("./my_app")

        for kernel in results.kernels:
            print(f"{kernel.name}: {kernel.total_samples} samples")
            for hotspot in kernel.top_instructions[:5]:
                print(f"  {hotspot.percentage:.1f}% {hotspot.opcode}")
            if kernel.top_stall_reasons:
                for reason, pct in kernel.top_stall_reasons.items():
                    print(f"  stall: {pct:.1f}% {reason}")
    """

    def __init__(self, timeout_seconds: Optional[int] = 0):
        self.wrapper = PCSamplingWrapper(timeout_seconds=timeout_seconds)

    def list_configs(self) -> str:
        """List available PC sampling configurations for all GPUs."""
        return self.wrapper.list_configs()

    def sample(
        self,
        command: str,
        interval: int = 65536,
        kernel_filter: Optional[str] = None,
        top_n: int = 10,
        cwd: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> SamplingResults:
        """
        Run stochastic PC sampling on a command and return analyzed results.

        Args:
            command: Command to profile
            interval: Sampling interval in cycles, power of 2. Default 65536.
                      Lower = more samples but higher overhead. 4096+ recommended.
            kernel_filter: Regex to filter kernels
            top_n: Number of top instructions to report per kernel
            cwd: Working directory
            output_dir: Output directory (temp dir if None)

        Returns:
            SamplingResults with per-kernel analysis including stall reasons
        """
        raw = self.wrapper.sample(
            command=command,
            interval=interval,
            kernel_filter=kernel_filter,
            cwd=cwd,
            output_dir=output_dir,
        )

        # Build dispatch_id -> kernel_name mapping
        dispatch_to_kernel = {}
        dispatch_to_duration = {}
        for d in raw.dispatches:
            dispatch_to_kernel[d.dispatch_id] = d.kernel_name
            dispatch_to_duration[d.dispatch_id] = d.duration_us

        # Group samples by kernel name
        kernel_samples: Dict[str, List[PCSample]] = defaultdict(list)
        kernel_dispatch_ids: Dict[str, set] = defaultdict(set)

        for sample in raw.samples:
            kernel_name = dispatch_to_kernel.get(sample.dispatch_id, f"unknown_dispatch_{sample.dispatch_id}")
            kernel_samples[kernel_name].append(sample)
            kernel_dispatch_ids[kernel_name].add(sample.dispatch_id)

        # Analyze each kernel
        kernel_results = []
        for kernel_name, samples in kernel_samples.items():
            result = self._analyze_kernel(
                kernel_name, samples, list(kernel_dispatch_ids[kernel_name]),
                dispatch_to_duration, top_n,
            )
            kernel_results.append(result)

        # Sort by sample count (most sampled first)
        kernel_results.sort(key=lambda k: k.total_samples, reverse=True)

        # Global opcode stats
        global_opcodes = self._compute_opcode_stats(raw.samples, top_n)

        return SamplingResults(
            command=command,
            total_samples=len(raw.samples),
            total_dispatches=len(raw.dispatches),
            interval=raw.interval,
            kernels=kernel_results,
            global_top_opcodes=global_opcodes,
        )

    def _analyze_kernel(
        self,
        kernel_name: str,
        samples: List[PCSample],
        dispatch_ids: List[int],
        dispatch_to_duration: Dict[int, float],
        top_n: int,
    ) -> KernelSamplingResult:
        """Analyze PC samples for a single kernel."""
        total = len(samples)

        duration_us = sum(dispatch_to_duration.get(d, 0.0) for d in dispatch_ids)
        hotspots = self._compute_instruction_stats(samples, top_n)

        # Exec mask stats
        full_mask = sum(1 for s in samples if s.exec_mask == 0xFFFFFFFFFFFFFFFF)
        empty_instr = sum(1 for s in samples if not s.instruction.strip())

        # Stall analysis
        issued = sum(1 for s in samples if s.wave_issued)
        issued_pct = (issued / total * 100) if total > 0 else 0.0

        stall_counter = Counter()
        for s in samples:
            if not s.wave_issued and s.stall_reason:
                reason = s.stall_reason.replace(
                    "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_", ""
                )
                stall_counter[reason] += 1
        stalled = total - issued
        top_stall_reasons = None
        if stalled > 0:
            top_stall_reasons = {
                r: count / stalled * 100 for r, count in stall_counter.most_common(10)
            }

        return KernelSamplingResult(
            name=kernel_name,
            dispatch_ids=dispatch_ids,
            total_samples=total,
            duration_us=duration_us,
            top_instructions=hotspots,
            full_mask_pct=(full_mask / total * 100) if total > 0 else 0.0,
            empty_instruction_count=empty_instr,
            issued_pct=issued_pct,
            top_stall_reasons=top_stall_reasons,
        )

    def _compute_instruction_stats(
        self, samples: List[PCSample], top_n: int,
    ) -> List[InstructionHotspot]:
        """Compute instruction-level hotspot statistics."""
        total = len(samples)
        instr_counter = Counter()
        instr_full: Dict[str, str] = {}

        instr_issued: Dict[str, int] = defaultdict(int)
        instr_stalled: Dict[str, int] = defaultdict(int)
        instr_stall_reasons: Dict[str, Counter] = defaultdict(Counter)
        instr_types: Dict[str, Counter] = defaultdict(Counter)

        for s in samples:
            parts = s.instruction.split()
            opcode = parts[0] if parts else "(empty)"
            instr_counter[opcode] += 1
            if parts:
                instr_full[opcode] = s.instruction

            if s.wave_issued:
                instr_issued[opcode] += 1
            else:
                instr_stalled[opcode] += 1
            if s.stall_reason:
                reason = s.stall_reason.replace(
                    "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_", ""
                )
                instr_stall_reasons[opcode][reason] += 1
            if s.instruction_type:
                itype = s.instruction_type.replace(
                    "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_", ""
                )
                instr_types[opcode][itype] += 1

        hotspots = []
        for opcode, count in instr_counter.most_common(top_n):
            hotspot = InstructionHotspot(
                instruction=instr_full.get(opcode, opcode),
                opcode=opcode,
                sample_count=count,
                percentage=(count / total * 100) if total > 0 else 0.0,
                issued_count=instr_issued.get(opcode, 0),
                stalled_count=instr_stalled.get(opcode, 0),
                stall_reasons=dict(instr_stall_reasons.get(opcode, {})),
                instruction_types=dict(instr_types.get(opcode, {})),
            )
            hotspots.append(hotspot)

        return hotspots

    def _compute_opcode_stats(
        self, samples: List[PCSample], top_n: int
    ) -> List[InstructionHotspot]:
        """Compute global opcode frequency stats."""
        total = len(samples)
        counter = Counter()
        instr_full = {}

        for s in samples:
            parts = s.instruction.split()
            opcode = parts[0] if parts else "(empty)"
            counter[opcode] += 1
            if parts:
                instr_full[opcode] = s.instruction

        return [
            InstructionHotspot(
                instruction=instr_full.get(op, op),
                opcode=op,
                sample_count=count,
                percentage=(count / total * 100) if total > 0 else 0.0,
            )
            for op, count in counter.most_common(top_n)
        ]
