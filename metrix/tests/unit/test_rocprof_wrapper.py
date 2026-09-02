"""
Unit tests for ROCProfiler V3 wrapper
Testing CSV parsing and data structure handling
"""

import pytest
import tempfile
import csv
from pathlib import Path
from unittest.mock import patch, MagicMock
from metrix.profiler.rocprof_wrapper import (
    ROCProfV3Wrapper,
    ProfileResult,
    _extract_missing_counters,
)


class TestProfileResult:
    """Test ProfileResult dataclass"""

    def test_create_profile_result(self):
        """Create a ProfileResult object"""
        result = ProfileResult(
            dispatch_id=1,
            kernel_name="test_kernel",
            gpu_id=0,
            duration_ns=1000,
            grid_size=(256, 1, 1),
            workgroup_size=(64, 1, 1),
            counters={"TCC_HIT_sum": 100.0, "TCC_MISS_sum": 50.0},
        )

        assert result.dispatch_id == 1
        assert result.kernel_name == "test_kernel"
        assert result.duration_ns == 1000
        assert result.grid_size == (256, 1, 1)
        assert result.counters["TCC_HIT_sum"] == 100.0


class TestROCProfV3Wrapper:
    """Test ROCProfiler wrapper"""

    @pytest.fixture
    def wrapper(self):
        return ROCProfV3Wrapper(timeout_seconds=60)

    def test_wrapper_creation(self, wrapper):
        """Wrapper can be created"""
        assert wrapper.timeout == 60

    def test_create_input_yaml(self, wrapper):
        """Input YAML generation works correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            counters = ["TCC_HIT_sum", "TCC_MISS_sum", "SQ_WAVES"]

            input_file = wrapper._create_input_yaml(counters, tmppath)

            assert input_file.exists()
            content = input_file.read_text()

            assert "TCC_HIT_sum" in content
            assert "TCC_MISS_sum" in content
            assert "SQ_WAVES" in content

    def test_parse_csv_row(self, wrapper):
        """CSV row parsing works correctly"""
        # Mock CSV row
        row = {
            "Dispatch_ID": "1",
            "Kernel_Name": "test_kernel(int*)",
            "GPU_ID": "0",
            "Grid_Size": "8192",
            "Workgroup_Size": "256",
            "LDS_Per_Workgroup": "0",
            "Scratch_Per_Workitem": "0",
            "Arch_VGPR": "4",
            "Accum_VGPR": "4",
            "SGPR": "16",
            "wave_size": "64",
            "obj": "0x7fa979c88580",
            "Start_Timestamp": "2525223085264657",
            "End_Timestamp": "2525223085267982",
            "TCC_HIT_sum": "1000.0",
            "TCC_MISS_sum": "500.0",
            "SQ_WAVES": "128",
        }

        result = wrapper._parse_csv_row(row)

        assert result.dispatch_id == 1
        assert result.kernel_name == "test_kernel(int*)"
        assert result.gpu_id == 0
        assert result.duration_ns == 3325  # end - start
        assert result.grid_size == (8192,)
        assert result.workgroup_size == (256,)
        assert result.arch_vgpr == 4
        assert result.sgpr == 16

        # Check counters
        assert result.counters["TCC_HIT_sum"] == 1000.0
        assert result.counters["TCC_MISS_sum"] == 500.0
        assert result.counters["SQ_WAVES"] == 128.0

    def test_parse_csv_row_with_3d_grid(self, wrapper):
        """Parse row with 3D grid/workgroup sizes"""
        row = {
            "Dispatch_ID": "2",
            "Kernel_Name": "kernel_3d",
            "GPU_ID": "0",
            "Grid_Size": "256 256 1",  # Space-separated
            "Workgroup_Size": "16,16,1",  # Comma-separated
            "LDS_Per_Workgroup": "1024",
            "Arch_VGPR": "8",
            "Accum_VGPR": "0",
            "SGPR": "32",
            "wave_size": "64",
            "obj": "0x123",
            "Start_Timestamp": "1000",
            "End_Timestamp": "2000",
        }

        result = wrapper._parse_csv_row(row)

        assert result.grid_size == (256, 256, 1)
        assert result.workgroup_size == (16, 16, 1)
        assert result.lds_per_workgroup == 1024

    @pytest.mark.skip(reason="rocprofv3 format changed - covered by integration tests")
    def test_parse_output_csv(self, wrapper):
        """Parse full CSV file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            csv_file = tmppath / "pmc_perf.csv"

            # Create mock CSV
            rows = [
                {
                    "Dispatch_ID": "1",
                    "Kernel_Name": "kernel_1",
                    "GPU_ID": "0",
                    "Grid_Size": "1024",
                    "Workgroup_Size": "256",
                    "LDS_Per_Workgroup": "0",
                    "Arch_VGPR": "4",
                    "Accum_VGPR": "0",
                    "SGPR": "16",
                    "wave_size": "64",
                    "obj": "0x1",
                    "Start_Timestamp": "1000",
                    "End_Timestamp": "2000",
                    "TCC_HIT_sum": "100",
                    "TCC_MISS_sum": "50",
                },
                {
                    "Dispatch_ID": "2",
                    "Kernel_Name": "kernel_2",
                    "GPU_ID": "0",
                    "Grid_Size": "2048",
                    "Workgroup_Size": "256",
                    "LDS_Per_Workgroup": "512",
                    "Arch_VGPR": "8",
                    "Accum_VGPR": "4",
                    "SGPR": "32",
                    "wave_size": "64",
                    "obj": "0x2",
                    "Start_Timestamp": "3000",
                    "End_Timestamp": "5000",
                    "TCC_HIT_sum": "200",
                    "TCC_MISS_sum": "100",
                },
            ]

            # Write CSV
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            # Parse
            results = wrapper._parse_output(tmppath)

            assert len(results) == 2
            assert results[0].kernel_name == "kernel_1"
            assert results[1].kernel_name == "kernel_2"
            assert results[0].counters["TCC_HIT_sum"] == 100.0
            assert results[1].counters["TCC_HIT_sum"] == 200.0

    @pytest.fixture
    def wrapper_no_rocm_check(self):
        with patch.object(ROCProfV3Wrapper, "_check_rocprofv3"):
            return ROCProfV3Wrapper(timeout_seconds=60)

    def test_command_with_quoted_args_preserved(self, wrapper_no_rocm_check):
        """Commands with quoted arguments must be split correctly (shlex, not str.split).

        This is a regression test: str.split() would turn
            python -c "import torch; print(1)"
        into
            ["python", "-c", "\"import", "torch;", "print(1)\""]
        which causes SyntaxError in the spawned Python process.
        shlex.split() correctly produces:
            ["python", "-c", "import torch; print(1)"]
        """
        wrapper = wrapper_no_rocm_check
        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch.object(wrapper, "_parse_output", return_value=[]),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            wrapper.profile(
                command='python -c "import torch; print(torch.cuda.is_available())"',
                counters=["SQ_WAVES"],
                output_dir=Path(tmpdir),
            )

        # Find everything after "--"
        sep_idx = captured_cmd.index("--")
        target_args = captured_cmd[sep_idx + 1 :]

        assert target_args == [
            "python",
            "-c",
            "import torch; print(torch.cuda.is_available())",
        ], f"Quoted argument was mangled: {target_args}"

    def test_command_with_single_quoted_args(self, wrapper_no_rocm_check):
        """Single-quoted arguments should also be handled correctly"""
        wrapper = wrapper_no_rocm_check
        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch.object(wrapper, "_parse_output", return_value=[]),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            wrapper.profile(
                command="python -c 'print(1+1)'",
                counters=["SQ_WAVES"],
                output_dir=Path(tmpdir),
            )

        sep_idx = captured_cmd.index("--")
        target_args = captured_cmd[sep_idx + 1 :]

        assert target_args == ["python", "-c", "print(1+1)"]

    def test_command_with_unmatched_quotes_raises(self, wrapper_no_rocm_check):
        """Commands with unmatched quotes should raise RuntimeError, not ValueError"""
        wrapper = wrapper_no_rocm_check

        def fake_run(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch.object(wrapper, "_parse_output", return_value=[]),
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(RuntimeError, match="Failed to parse command"),
        ):
            wrapper.profile(
                command='python -c "unterminated',
                counters=["SQ_WAVES"],
                output_dir=Path(tmpdir),
            )

    def test_simple_command_still_works(self, wrapper_no_rocm_check):
        """Simple commands without quotes should still be split normally"""
        wrapper = wrapper_no_rocm_check
        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch.object(wrapper, "_parse_output", return_value=[]),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            wrapper.profile(
                command="./benchmark --size 1024 --warmup 5",
                counters=["SQ_WAVES"],
                output_dir=Path(tmpdir),
            )

        sep_idx = captured_cmd.index("--")
        target_args = captured_cmd[sep_idx + 1 :]

        assert target_args == ["./benchmark", "--size", "1024", "--warmup", "5"]

    def test_kernel_filter_uses_kernel_include_regex(self, wrapper_no_rocm_check):
        """kernel_filter passes the value directly to --kernel-include-regex"""
        wrapper = wrapper_no_rocm_check
        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch.object(wrapper, "_parse_output", return_value=[]),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            wrapper.profile(
                command="true",
                counters=[],
                output_dir=Path(tmpdir),
                kernel_filter="my_kernel",
            )

        assert "--kernel-include-regex" in captured_cmd
        idx = captured_cmd.index("--kernel-include-regex")
        assert captured_cmd[idx + 1] == "my_kernel"

    def test_kernel_filter_passes_value_unchanged(self, wrapper_no_rocm_check):
        """kernel_filter with a regex pattern is passed through unmodified"""
        wrapper = wrapper_no_rocm_check
        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result

        regex_pattern = "kernel_.*_v2"
        with (
            patch("subprocess.run", side_effect=fake_run),
            patch.object(wrapper, "_parse_output", return_value=[]),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            wrapper.profile(
                command="true",
                counters=[],
                output_dir=Path(tmpdir),
                kernel_filter=regex_pattern,
            )

        idx = captured_cmd.index("--kernel-include-regex")
        assert captured_cmd[idx + 1] == regex_pattern

    def test_kernel_filter_post_filter_timing_only(self, wrapper_no_rocm_check):
        """In timing-only mode, post-filter drops results that do not match kernel_filter."""
        wrapper = wrapper_no_rocm_check
        matching = ProfileResult(
            dispatch_id=1,
            kernel_name="my_kernel(float*, int)",
            gpu_id=0,
            duration_ns=1000,
            grid_size=(256, 1, 1),
            workgroup_size=(64, 1, 1),
            counters={},
        )
        non_matching = ProfileResult(
            dispatch_id=2,
            kernel_name="__amd_rocclr_copyBuffer",
            gpu_id=0,
            duration_ns=500,
            grid_size=(512, 1, 1),
            workgroup_size=(512, 1, 1),
            counters={},
        )
        parsed = [non_matching, matching, non_matching]

        def fake_run(cmd, **kwargs):
            m = MagicMock()
            m.returncode = 0
            m.stdout = ""
            m.stderr = ""
            return m

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch.object(wrapper, "_parse_output", return_value=parsed),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            results = wrapper.profile(
                command="true",
                counters=[],
                output_dir=Path(tmpdir),
                kernel_filter="my_kernel",
            )

        assert len(results) == 1
        assert results[0].kernel_name == "my_kernel(float*, int)"

    def test_parse_missing_optional_fields(self, wrapper):
        """Handle missing optional fields gracefully"""
        row = {
            "Dispatch_ID": "1",
            "Kernel_Name": "kernel",
            "GPU_ID": "0",
            "Grid_Size": "1024",
            "Workgroup_Size": "256",
            "wave_size": "64",
            "obj": "0x1",
            "Start_Timestamp": "1000",
            "End_Timestamp": "2000",
            # Missing: LDS_Per_Workgroup, VGPRs, etc.
        }

        result = wrapper._parse_csv_row(row)

        # Should use defaults
        assert result.lds_per_workgroup == 0
        assert result.arch_vgpr == 0
        assert result.accum_vgpr == 0
        assert result.sgpr == 0


class TestCSVParsingRobustness:
    """Test CSV parsing edge cases"""

    @pytest.fixture
    def wrapper(self):
        return ROCProfV3Wrapper()

    def test_handle_non_numeric_counter_values(self, wrapper):
        """Handle non-numeric values in counter columns"""
        row = {
            "Dispatch_ID": "1",
            "Kernel_Name": "kernel",
            "GPU_ID": "0",
            "Grid_Size": "1024",
            "Workgroup_Size": "256",
            "wave_size": "64",
            "obj": "0x1",
            "Start_Timestamp": "1000",
            "End_Timestamp": "2000",
            "TCC_HIT_sum": "100.5",
            "SOME_STRING_FIELD": "text_value",
        }

        result = wrapper._parse_csv_row(row)

        # Numeric value parsed
        assert result.counters["TCC_HIT_sum"] == 100.5
        # String value kept as string
        assert result.counters["SOME_STRING_FIELD"] == "text_value"

    def test_grid_size_formats(self, wrapper):
        """Handle different grid size formats"""
        test_cases = [
            ("1024", (1024,)),
            ("256 256", (256, 256)),
            ("128,128,1", (128, 128, 1)),
            ("64 64 64", (64, 64, 64)),
        ]

        for grid_str, expected in test_cases:
            row = {
                "Dispatch_ID": "1",
                "Kernel_Name": "k",
                "GPU_ID": "0",
                "Grid_Size": grid_str,
                "Workgroup_Size": "256",
                "wave_size": "64",
                "obj": "0x1",
                "Start_Timestamp": "1000",
                "End_Timestamp": "2000",
            }

            result = wrapper._parse_csv_row(row)
            assert result.grid_size == expected


class TestMissingCounterDiagnostics:
    """rocprofv3 exits 0 and writes nothing when a counter is unavailable.

    The only signal is a warning on stderr; without it the failure surfaces as
    a bare 'No output CSV found', which says nothing about the cause.
    """

    # Verbatim from rocprofv3 (ROCm 7.2.3) on a gfx1030 agent.
    STDERR = (
        "W20260901 13:55:05.889322 127091653975872 tool.cpp:2422] HSA version 8.20.3 initialized\n"
        "W20260901 13:55:05.921218 127091653975872 tool.cpp:1189] Unable to find all counters "
        "for agent 1 (gpu-0, gfx1030) in [MeanOccupancyPerActiveCU]. "
        "Found: []. Missing: [MeanOccupancyPerActiveCU]\n"
    )

    def test_missing_counters_are_extracted_from_stderr(self):
        assert _extract_missing_counters(self.STDERR) == ["MeanOccupancyPerActiveCU"]

    def test_several_missing_counters_are_all_reported(self):
        stderr = "tool.cpp:1189] ... Found: [SQ_WAVES]. Missing: [TA_BUFFER_LOAD_WAVEFRONTS_sum, GRBM_FOO]\n"
        assert _extract_missing_counters(stderr) == [
            "TA_BUFFER_LOAD_WAVEFRONTS_sum",
            "GRBM_FOO",
        ]

    def test_unremarkable_stderr_yields_no_counters(self):
        assert _extract_missing_counters("HSA version 8.20.3 initialized\n") == []
        assert _extract_missing_counters("") == []

    def test_error_names_the_missing_counter_not_just_the_directory(self):
        wrapper = ROCProfV3Wrapper()
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(RuntimeError) as exc:
                wrapper._parse_output(Path(tmp), stderr=self.STDERR, arch="gfx1030")

        message = str(exc.value)
        assert "MeanOccupancyPerActiveCU" in message
        assert "gfx1030" in message
        assert "--list-avail" in message, "should point at how to find valid names"

    def test_error_falls_back_to_stderr_when_no_pattern_matches(self):
        """An unrecognized failure must still surface rocprofv3's own output."""
        wrapper = ROCProfV3Wrapper()
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(RuntimeError) as exc:
                wrapper._parse_output(Path(tmp), stderr="something else went wrong\n")

        assert "something else went wrong" in str(exc.value)

    def test_error_without_stderr_still_reports_the_directory(self):
        wrapper = ROCProfV3Wrapper()
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(RuntimeError, match="No output CSV found"):
                wrapper._parse_output(Path(tmp))
