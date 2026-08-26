"""Unit tests for kerncap.profiler — rocprofv3 invocation and CSV parsing.

``run_streaming`` and ``shutil.which`` are stubbed at their call sites so
``run_profile`` runs with no ROCm and no GPU.  The ``run_streaming`` stub
reads the ``--output-directory`` out of the argv it is handed and writes a
real CSV there, which keeps the file-discovery and parsing logic under test
rather than mocked away.
"""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from kerncap.profiler import (
    KernelStat,
    _find_stats_csv,
    _list_tree,
    _write_profile_json,
    parse_kernel_trace_stats,
    run_profile,
)

FIXTURES = Path(__file__).parent / "fixtures"

STATS_CSV = (
    '"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"\n'
    '"matmul_kernel",1024,580000000,566406,42.3,480000,720000,45000.0\n'
    '"add_kernel",64,1500000,23437,0.1,20000,30000,1500.0\n'
)


def fake_run_streaming(returncode=0, csv_text=STATS_CSV, subdir="myhost", write_csv=True):
    """Stand in for ``run_streaming``, materialising rocprofv3's CSV output.

    rocprofv3 writes to ``<output-dir>/<hostname>/<pid>_kernel_stats.csv``,
    so the stub reproduces that nesting to exercise the rglob in
    ``_find_stats_csv``.
    """

    def _run(argv, **kwargs):
        outdir = argv[argv.index("--output-directory") + 1]
        if write_csv:
            nested = Path(outdir) / subdir
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "4242_kernel_stats.csv").write_text(csv_text)
        return subprocess.CompletedProcess(
            args=list(argv), returncode=returncode, stdout="app output", stderr="app errors"
        )

    return _run


class TestParseKernelTraceStats:
    """Tests for the rocprofv3 kernel_trace_stats.csv parser."""

    def test_parse_rocprofv3_stats(self):
        """Parse rocprofv3 kernel_trace_stats.csv with all columns."""
        csv_path = str(FIXTURES / "rocprofv3_kernel_stats.csv")
        kernels = parse_kernel_trace_stats(csv_path)

        assert len(kernels) == 5

        # Should be sorted by total duration descending
        assert kernels[0].total_duration_ns == 580000000
        assert "matmul_kernel" in kernels[0].name
        assert kernels[0].calls == 1024
        assert kernels[0].percentage == pytest.approx(42.3)
        assert kernels[0].min_duration_ns == 480000
        assert kernels[0].max_duration_ns == 720000
        assert kernels[0].stddev_ns == pytest.approx(45000.0)

        # Last kernel should have the smallest total
        assert kernels[-1].total_duration_ns == 140000000

    def test_sorted_by_total_duration(self):
        """Results should always be sorted descending by total duration."""
        csv_path = str(FIXTURES / "rocprofv3_kernel_stats.csv")
        kernels = parse_kernel_trace_stats(csv_path)

        durations = [k.total_duration_ns for k in kernels]
        assert durations == sorted(durations, reverse=True)

    def test_avg_duration_populated(self):
        """AverageNs should be parsed from the stats CSV."""
        csv_path = str(FIXTURES / "rocprofv3_kernel_stats.csv")
        kernels = parse_kernel_trace_stats(csv_path)

        for k in kernels:
            assert k.avg_duration_ns > 0

    def test_empty_csv(self, tmp_path):
        """Empty CSV should return empty list."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        kernels = parse_kernel_trace_stats(str(csv_file))
        assert kernels == []

    def test_missing_columns_raises(self, tmp_path):
        """CSV without a name column should raise ValueError."""
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("foo,bar\n1,2\n")
        with pytest.raises(ValueError, match="kernel name column"):
            parse_kernel_trace_stats(str(csv_file))

    def test_comment_lines_are_stripped(self, tmp_path):
        """rocprofv3 may prefix the CSV with '#' banner lines."""
        csv_file = tmp_path / "commented.csv"
        csv_file.write_text("# rocprofv3 v1\n# generated\n" + STATS_CSV)
        kernels = parse_kernel_trace_stats(str(csv_file))
        assert [k.name for k in kernels] == ["matmul_kernel", "add_kernel"]

    def test_rows_without_a_name_are_skipped(self, tmp_path):
        """Trailing blank rows must not become nameless kernels."""
        csv_file = tmp_path / "blank.csv"
        csv_file.write_text(STATS_CSV + '"",0,0,0,0.0,0,0,0.0\n')
        assert len(parse_kernel_trace_stats(str(csv_file))) == 2

    def test_avg_is_derived_when_the_column_is_absent(self, tmp_path):
        """Without AverageNs, avg falls back to total // calls."""
        csv_file = tmp_path / "noavg.csv"
        csv_file.write_text('"Name","Calls","TotalDurationNs"\n"k",4,1000\n')
        (kernel,) = parse_kernel_trace_stats(str(csv_file))
        assert kernel.avg_duration_ns == 250

    def test_derived_avg_does_not_divide_by_zero(self, tmp_path):
        csv_file = tmp_path / "zero.csv"
        csv_file.write_text('"Name","Calls","TotalDurationNs"\n"k",0,1000\n')
        (kernel,) = parse_kernel_trace_stats(str(csv_file))
        assert kernel.avg_duration_ns == 0

    def test_column_names_are_matched_case_and_space_insensitively(self, tmp_path):
        csv_file = tmp_path / "loose.csv"
        csv_file.write_text('  name , Calls , TotalDurationNs \n"k",2,100\n')
        (kernel,) = parse_kernel_trace_stats(str(csv_file))
        assert kernel.name == "k"
        assert kernel.calls == 2


# --------------------------------------------------------------------------
# run_profile
# --------------------------------------------------------------------------


class TestRunProfile:
    def test_returns_parsed_kernels(self, tmp_path):
        with (
            patch("shutil.which", return_value="/opt/rocm/bin/rocprofv3"),
            patch("kerncap.profiler.run_streaming", fake_run_streaming()),
        ):
            kernels = run_profile(["./my_app", "--flag"])

        assert [k.name for k in kernels] == ["matmul_kernel", "add_kernel"]
        assert kernels[0].total_duration_ns == 580000000

    def test_builds_the_rocprofv3_command(self, tmp_path):
        """The app command must sit after ``--`` so its flags aren't eaten."""
        seen = {}

        def _run(argv, **kwargs):
            seen["argv"] = list(argv)
            seen["timeout"] = kwargs.get("timeout")
            return fake_run_streaming()(argv, **kwargs)

        with (
            patch("shutil.which", return_value="/opt/rocm/bin/rocprofv3"),
            patch("kerncap.profiler.run_streaming", _run),
        ):
            run_profile(["./my_app", "--flag"], timeout=90)

        argv = seen["argv"]
        assert argv[0] == "/opt/rocm/bin/rocprofv3"
        assert "--kernel-trace" in argv
        assert "--stats" in argv
        assert argv[argv.index("--output-format") + 1] == "csv"
        assert argv[argv.index("--") + 1 :] == ["./my_app", "--flag"]
        assert seen["timeout"] == 90

    def test_explicit_rocprof_bin_skips_lookup(self, tmp_path):
        with (
            patch("shutil.which") as which,
            patch("kerncap.profiler.run_streaming", fake_run_streaming()) as _,
        ):
            run_profile(["./app"], rocprof_bin="/custom/rocprofv3")

        which.assert_not_called()

    def test_missing_rocprofv3_raises_with_guidance(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="rocprofv3 not found on PATH"):
                run_profile(["./app"])

    def test_timeout_is_translated(self):
        """subprocess.TimeoutExpired becomes a TimeoutError naming the limit."""
        with (
            patch("shutil.which", return_value="/x/rocprofv3"),
            patch(
                "kerncap.profiler.run_streaming",
                side_effect=subprocess.TimeoutExpired("rocprofv3", 30),
            ),
        ):
            with pytest.raises(TimeoutError, match="did not complete within 30s"):
                run_profile(["./app"], timeout=30)

    def test_nonzero_exit_without_output_raises(self):
        """No CSV and a bad exit code is a hard failure, with both streams."""
        with (
            patch("shutil.which", return_value="/x/rocprofv3"),
            patch(
                "kerncap.profiler.run_streaming",
                fake_run_streaming(returncode=1, write_csv=False),
            ),
        ):
            with pytest.raises(RuntimeError) as exc:
                run_profile(["./app"])

        assert "exited with code 1" in str(exc.value)
        assert "app output" in str(exc.value)
        assert "app errors" in str(exc.value)

    def test_nonzero_exit_with_output_still_succeeds(self):
        """A crashing app that still produced stats is usable, not fatal.

        This is a distinct branch from the failure above; collapsing the two
        would throw away a perfectly good profile whenever the workload
        exits non-zero after the kernels have already run.
        """
        with (
            patch("shutil.which", return_value="/x/rocprofv3"),
            patch("kerncap.profiler.run_streaming", fake_run_streaming(returncode=139)),
        ):
            kernels = run_profile(["./app"])

        assert len(kernels) == 2

    def test_clean_exit_without_csv_raises(self):
        """Exit 0 but no stats file means rocprofv3 traced nothing."""
        with (
            patch("shutil.which", return_value="/x/rocprofv3"),
            patch("kerncap.profiler.run_streaming", fake_run_streaming(write_csv=False)),
        ):
            with pytest.raises(FileNotFoundError, match="did not produce a kernel_stats.csv"):
                run_profile(["./app"])

    def test_writes_json_when_output_path_given(self, tmp_path):
        out = tmp_path / "profile.json"
        with (
            patch("shutil.which", return_value="/x/rocprofv3"),
            patch("kerncap.profiler.run_streaming", fake_run_streaming()),
        ):
            run_profile(["./my_app", "--flag"], output_path=str(out))

        data = json.loads(out.read_text())
        assert data["cmd"] == "./my_app --flag"
        assert [k["name"] for k in data["kernels"]] == ["matmul_kernel", "add_kernel"]

    def test_no_json_written_without_output_path(self, tmp_path):
        with (
            patch("shutil.which", return_value="/x/rocprofv3"),
            patch("kerncap.profiler.run_streaming", fake_run_streaming()),
        ):
            run_profile(["./app"])

        assert list(tmp_path.iterdir()) == []

    def test_temp_directory_is_cleaned_up(self):
        captured = {}

        def _run(argv, **kwargs):
            captured["outdir"] = argv[argv.index("--output-directory") + 1]
            return fake_run_streaming()(argv, **kwargs)

        with (
            patch("shutil.which", return_value="/x/rocprofv3"),
            patch("kerncap.profiler.run_streaming", _run),
        ):
            run_profile(["./app"])

        assert "kerncap_prof_" in captured["outdir"]
        assert not os.path.exists(captured["outdir"])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class TestFindStatsCsv:
    def test_finds_a_nested_stats_file(self, tmp_path):
        """rocprofv3 nests output under <hostname>/<pid>_kernel_stats.csv."""
        nested = tmp_path / "myhost"
        nested.mkdir()
        target = nested / "123_kernel_stats.csv"
        target.write_text(STATS_CSV)

        assert _find_stats_csv(str(tmp_path)) == str(target)

    def test_returns_none_when_absent(self, tmp_path):
        (tmp_path / "other.csv").write_text("x")
        assert _find_stats_csv(str(tmp_path)) is None


class TestListTree:
    def test_lists_files_relative_to_base(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        assert sorted(_list_tree(str(tmp_path))) == ["b.txt", os.path.join("sub", "a.txt")]

    def test_directories_are_omitted(self, tmp_path):
        (tmp_path / "emptydir").mkdir()
        assert _list_tree(str(tmp_path)) == []


class TestWriteProfileJson:
    def test_writes_every_field(self, tmp_path):
        out = tmp_path / "p.json"
        stat = KernelStat(
            name="k",
            calls=2,
            total_duration_ns=100,
            avg_duration_ns=50,
            percentage=12.5,
            min_duration_ns=40,
            max_duration_ns=60,
            stddev_ns=7.5,
        )
        _write_profile_json([stat], str(out), ["./app", "--x"])

        data = json.loads(out.read_text())
        assert data["kernels"][0] == {
            "name": "k",
            "calls": 2,
            "total_duration_ns": 100,
            "avg_duration_ns": 50,
            "percentage": 12.5,
            "min_duration_ns": 40,
            "max_duration_ns": 60,
            "stddev_ns": 7.5,
        }
        assert "timestamp" in data

    def test_command_is_shell_quoted(self, tmp_path):
        """shlex.join keeps the record copy-pasteable for an argument with spaces."""
        out = tmp_path / "p.json"
        _write_profile_json([], str(out), ["./app", "--msg", "hello world"])

        assert json.loads(out.read_text())["cmd"] == "./app --msg 'hello world'"
