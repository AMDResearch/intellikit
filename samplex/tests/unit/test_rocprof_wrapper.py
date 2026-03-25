"""Unit tests for rocprof_wrapper (no GPU required)."""

import csv
import os
import tempfile
import pytest

from samplex.profiler.rocprof_wrapper import PCSamplingWrapper, PCSample, KernelDispatch


class TestCSVParsing:
    """Test CSV parsing logic with synthetic files."""

    def test_parse_host_trap_samples(self):
        wrapper = PCSamplingWrapper.__new__(PCSamplingWrapper)
        wrapper.timeout = None

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "out_pc_sampling_host_trap.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow([
                    "Sample_Timestamp", "Exec_Mask", "Dispatch_Id",
                    "Instruction", "Instruction_Comment", "Correlation_Id",
                ])
                writer.writerow([
                    "1000000", "18446744073709551615", "1",
                    "s_waitcnt vmcnt(0)", "", "1",
                ])
                writer.writerow([
                    "1000100", "255", "2",
                    "global_load_dwordx4 v[0:3], v[4:5], off", "src.hip:42", "2",
                ])

            from pathlib import Path
            samples = wrapper._parse_samples(Path(tmpdir), "host_trap")

            assert len(samples) == 2
            assert samples[0].timestamp == 1000000
            assert samples[0].exec_mask == 18446744073709551615
            assert samples[0].dispatch_id == 1
            assert samples[0].instruction == "s_waitcnt vmcnt(0)"
            assert samples[0].wave_issued is None  # host_trap has no wave_issued

            assert samples[1].dispatch_id == 2
            assert samples[1].exec_mask == 255
            assert samples[1].instruction_comment == "src.hip:42"

    def test_parse_stochastic_samples(self):
        wrapper = PCSamplingWrapper.__new__(PCSamplingWrapper)
        wrapper.timeout = None

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "out_pc_sampling_stochastic.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow([
                    "Sample_Timestamp", "Exec_Mask", "Dispatch_Id",
                    "Instruction", "Instruction_Comment", "Correlation_Id",
                    "Wave_Issued_Instruction", "Instruction_Type",
                    "Stall_Reason", "Wave_Count",
                ])
                writer.writerow([
                    "2000000", "18446744073709551615", "1",
                    "v_pk_mul_f32 v[0:1], s[2:3], v[0:1]", "", "1",
                    "1", "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU",
                    "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT", "3",
                ])
                writer.writerow([
                    "2000256", "18446744073709551615", "1",
                    "s_waitcnt lgkmcnt(0)", "", "1",
                    "0", "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST",
                    "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT", "2",
                ])

            from pathlib import Path
            samples = wrapper._parse_samples(Path(tmpdir), "stochastic")

            assert len(samples) == 2
            assert samples[0].wave_issued is True
            assert samples[0].instruction_type == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU"
            assert samples[0].wave_count == 3

            assert samples[1].wave_issued is False
            assert "WAITCNT" in samples[1].stall_reason

    def test_parse_kernel_trace(self):
        wrapper = PCSamplingWrapper.__new__(PCSamplingWrapper)
        wrapper.timeout = None

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "out_kernel_trace.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow([
                    "Kind", "Agent_Id", "Queue_Id", "Stream_Id", "Thread_Id",
                    "Dispatch_Id", "Kernel_Id", "Kernel_Name", "Correlation_Id",
                    "Start_Timestamp", "End_Timestamp", "LDS_Block_Size",
                    "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count",
                    "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z",
                    "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z",
                ])
                writer.writerow([
                    "KERNEL_DISPATCH", "Agent 6", "16", "0", "8",
                    "1", "42", "test_gemm_kernel", "1",
                    "1000000", "2000000", "0",
                    "0", "128", "0", "64",
                    "256", "1", "1",
                    "65536", "1", "1",
                ])

            from pathlib import Path
            dispatches = wrapper._parse_kernel_trace(Path(tmpdir))

            assert len(dispatches) == 1
            assert dispatches[0].dispatch_id == 1
            assert dispatches[0].kernel_name == "test_gemm_kernel"
            assert dispatches[0].duration_ns == 1000000
            assert dispatches[0].duration_us == 1000.0
            assert dispatches[0].vgpr_count == 128
            assert dispatches[0].workgroup_size == (256, 1, 1)

    def test_empty_output(self):
        wrapper = PCSamplingWrapper.__new__(PCSamplingWrapper)
        wrapper.timeout = None

        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            samples = wrapper._parse_samples(Path(tmpdir), "host_trap")
            assert samples == []

            dispatches = wrapper._parse_kernel_trace(Path(tmpdir))
            assert dispatches == []
