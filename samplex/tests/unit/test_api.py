# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for samplex API (no GPU required)."""

import pytest
from unittest.mock import MagicMock

from samplex.api import Samplex, SamplingResults, KernelSamplingResult, InstructionHotspot
from samplex.profiler.rocprof_wrapper import PCSample, KernelDispatch, PCSamplingResult


def _make_sample(
    dispatch_id=1,
    instruction="s_waitcnt vmcnt(0)",
    exec_mask=0xFFFFFFFFFFFFFFFF,
    wave_issued=False,
    stall_reason="",
    instruction_type="",
    wave_count=1,
):
    """Create a stochastic-style sample with all fields."""
    return PCSample(
        timestamp=1000000,
        exec_mask=exec_mask,
        dispatch_id=dispatch_id,
        instruction=instruction,
        instruction_comment="",
        correlation_id=dispatch_id,
        wave_issued=wave_issued,
        instruction_type=instruction_type,
        stall_reason=stall_reason,
        wave_count=wave_count,
    )


def _make_host_trap_sample(
    dispatch_id=1, instruction="s_waitcnt vmcnt(0)", exec_mask=0xFFFFFFFFFFFFFFFF
):
    """Create a host_trap-style sample (no stall/issued/type fields)."""
    return PCSample(
        timestamp=1000000,
        exec_mask=exec_mask,
        dispatch_id=dispatch_id,
        instruction=instruction,
        instruction_comment="",
        correlation_id=dispatch_id,
    )


def _make_dispatch(dispatch_id=1, kernel_name="test_kernel"):
    return KernelDispatch(
        dispatch_id=dispatch_id,
        kernel_name=kernel_name,
        agent_id="Agent 0",
        start_timestamp=1000000,
        end_timestamp=2000000,
    )


class TestSamplexAnalysis:
    """Test the analysis logic without running rocprofv3."""

    def test_analyze_kernel_basic(self):
        sampler = Samplex.__new__(Samplex)
        sampler.wrapper = MagicMock()

        samples = [
            _make_sample(instruction="s_waitcnt vmcnt(0)"),
            _make_sample(instruction="s_waitcnt vmcnt(0)"),
            _make_sample(instruction="s_waitcnt lgkmcnt(0)"),
            _make_sample(instruction="global_load_dwordx4 v[0:3], v[4:5], off"),
            _make_sample(instruction="v_mfma_f32_16x16x32_f16 a[0:3], v[0:1], v[2:3], a[0:3]"),
        ]

        result = sampler._analyze_kernel("my_kernel", samples, [1], {1: 500.0}, top_n=5)

        assert result.name == "my_kernel"
        assert result.total_samples == 5
        assert result.duration_us == 500.0
        # 3 unique opcodes: s_waitcnt (3 samples), global_load_dwordx4, v_mfma_*
        assert len(result.top_instructions) == 3
        assert result.top_instructions[0].opcode == "s_waitcnt"
        assert result.top_instructions[0].sample_count == 3
        assert result.top_instructions[0].percentage == 60.0

    def test_analyze_kernel_stall_reasons(self):
        sampler = Samplex.__new__(Samplex)
        sampler.wrapper = MagicMock()

        samples = [
            _make_sample(
                wave_issued=False,
                stall_reason="ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT",
            ),
            _make_sample(
                wave_issued=False,
                stall_reason="ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY",
            ),
            _make_sample(
                wave_issued=True,
                stall_reason="ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
            ),
        ]

        result = sampler._analyze_kernel("my_kernel", samples, [1], {1: 300.0}, top_n=5)

        assert result.issued_pct == pytest.approx(33.33, abs=0.1)
        assert result.top_stall_reasons is not None
        assert "WAITCNT" in result.top_stall_reasons
        assert "ALU_DEPENDENCY" in result.top_stall_reasons

    def test_empty_instruction_holes(self):
        sampler = Samplex.__new__(Samplex)
        sampler.wrapper = MagicMock()

        samples = [
            _make_sample(instruction="s_waitcnt vmcnt(0)"),
            _make_sample(instruction=""),
            _make_sample(instruction="  "),
            _make_sample(instruction="v_mov_b32_e32 v0, 0"),
        ]

        result = sampler._analyze_kernel("my_kernel", samples, [1], {1: 100.0}, top_n=5)

        assert result.empty_instruction_count == 2

    def test_exec_mask_stats(self):
        sampler = Samplex.__new__(Samplex)
        sampler.wrapper = MagicMock()

        samples = [
            _make_sample(exec_mask=0xFFFFFFFFFFFFFFFF),
            _make_sample(exec_mask=0xFFFFFFFFFFFFFFFF),
            _make_sample(exec_mask=0xFF),
            _make_sample(exec_mask=0x0),
        ]

        result = sampler._analyze_kernel("my_kernel", samples, [1], {1: 100.0}, top_n=5)

        assert result.full_mask_pct == 50.0

    def test_global_opcode_stats(self):
        sampler = Samplex.__new__(Samplex)
        sampler.wrapper = MagicMock()

        samples = [
            _make_sample(instruction="s_waitcnt vmcnt(0)"),
            _make_sample(instruction="s_waitcnt lgkmcnt(0)"),
            _make_sample(instruction="s_endpgm"),
            _make_sample(instruction="v_mfma_f32_16x16x32_f16 a[0:3], v[0:1], v[2:3], a[0:3]"),
        ]

        stats = sampler._compute_opcode_stats(samples, top_n=5)
        assert stats[0].opcode == "s_waitcnt"
        assert stats[0].sample_count == 2
        assert stats[0].percentage == 50.0

    def test_multiple_kernels(self):
        sampler = Samplex.__new__(Samplex)
        sampler.wrapper = MagicMock()

        raw = PCSamplingResult(
            command="./app",
            interval=256,
            method="stochastic",
            samples=[
                _make_sample(dispatch_id=1, instruction="s_waitcnt vmcnt(0)"),
                _make_sample(dispatch_id=1, instruction="s_waitcnt vmcnt(0)"),
                _make_sample(dispatch_id=2, instruction="v_mfma_f32_16x16x32_f16 a, v, v, a"),
            ],
            dispatches=[
                _make_dispatch(dispatch_id=1, kernel_name="kernel_A"),
                _make_dispatch(dispatch_id=2, kernel_name="kernel_B"),
            ],
        )

        sampler.wrapper.sample.return_value = raw
        results = sampler.sample("./app")

        assert len(results.kernels) == 2
        assert results.kernels[0].total_samples == 2  # kernel_A has more samples
        assert results.kernels[0].name == "kernel_A"
        assert results.kernels[1].name == "kernel_B"
        assert results.interval == 256
        assert results.method == "stochastic"

    def test_issued_vs_stalled_per_instruction(self):
        sampler = Samplex.__new__(Samplex)
        sampler.wrapper = MagicMock()

        samples = [
            _make_sample(
                instruction="s_waitcnt vmcnt(0)",
                wave_issued=False,
                stall_reason="ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT",
                instruction_type="ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST",
            ),
            _make_sample(
                instruction="s_waitcnt vmcnt(0)",
                wave_issued=False,
                stall_reason="ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT",
                instruction_type="ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST",
            ),
            _make_sample(
                instruction="v_mfma_f32_16x16x32_f16 a, v, v, a",
                wave_issued=True,
                instruction_type="ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU",
            ),
        ]

        result = sampler._analyze_kernel("my_kernel", samples, [1], {1: 100.0}, top_n=5)

        waitcnt = result.top_instructions[0]
        assert waitcnt.opcode == "s_waitcnt"
        assert waitcnt.issued_count == 0
        assert waitcnt.stalled_count == 2
        assert "WAITCNT" in waitcnt.stall_reasons

        mfma = result.top_instructions[1]
        assert mfma.opcode == "v_mfma_f32_16x16x32_f16"
        assert mfma.issued_count == 1
        assert mfma.stalled_count == 0


class TestHostTrapAnalysis:
    """Test analysis with host_trap samples (no stall/issued data)."""

    def test_host_trap_basic_analysis(self):
        sampler = Samplex.__new__(Samplex)
        sampler.wrapper = MagicMock()

        samples = [
            _make_host_trap_sample(instruction="s_waitcnt vmcnt(0)"),
            _make_host_trap_sample(instruction="s_waitcnt vmcnt(0)"),
            _make_host_trap_sample(instruction="global_load_dwordx4 v[0:3], v[4:5], off"),
            _make_host_trap_sample(
                instruction="v_mfma_f32_16x16x32_f16 a[0:3], v[0:1], v[2:3], a[0:3]"
            ),
        ]

        result = sampler._analyze_kernel("my_kernel", samples, [1], {1: 500.0}, top_n=5)

        assert result.name == "my_kernel"
        assert result.total_samples == 4
        # No issued/stalled data for host_trap — all default to wave_issued=False
        assert result.issued_pct == 0.0
        # No stall reasons populated (stall_reason is empty string for host_trap)
        assert result.top_stall_reasons == {}

    def test_host_trap_multiple_kernels(self):
        sampler = Samplex.__new__(Samplex)
        sampler.wrapper = MagicMock()

        raw = PCSamplingResult(
            command="./app",
            interval=1000,
            method="host_trap",
            samples=[
                _make_host_trap_sample(dispatch_id=1, instruction="s_waitcnt vmcnt(0)"),
                _make_host_trap_sample(dispatch_id=1, instruction="s_waitcnt vmcnt(0)"),
                _make_host_trap_sample(dispatch_id=2, instruction="v_mov_b32_e32 v0, 0"),
            ],
            dispatches=[
                _make_dispatch(dispatch_id=1, kernel_name="kernel_A"),
                _make_dispatch(dispatch_id=2, kernel_name="kernel_B"),
            ],
        )

        sampler.wrapper.sample.return_value = raw
        results = sampler.sample("./app", method="host_trap")

        assert len(results.kernels) == 2
        assert results.method == "host_trap"
        assert results.kernels[0].name == "kernel_A"
        # No stall reasons populated for host_trap
        assert results.kernels[0].top_stall_reasons == {}

    def test_host_trap_instruction_hotspots(self):
        sampler = Samplex.__new__(Samplex)
        sampler.wrapper = MagicMock()

        samples = [
            _make_host_trap_sample(instruction="s_waitcnt vmcnt(0)"),
            _make_host_trap_sample(instruction="s_waitcnt vmcnt(0)"),
            _make_host_trap_sample(instruction="s_waitcnt lgkmcnt(0)"),
            _make_host_trap_sample(instruction="v_mov_b32_e32 v0, 0"),
        ]

        result = sampler._analyze_kernel("my_kernel", samples, [1], {1: 100.0}, top_n=5)

        # s_waitcnt appears 3 times (grouped by opcode)
        assert result.top_instructions[0].opcode == "s_waitcnt"
        assert result.top_instructions[0].sample_count == 3
        # issued_count and stalled_count should be 0 and 3 respectively
        # (default wave_issued=False means all counted as stalled)
        assert result.top_instructions[0].issued_count == 0
        assert result.top_instructions[0].stalled_count == 3
