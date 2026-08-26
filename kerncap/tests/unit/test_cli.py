"""Unit tests for kerncap.cli — argument wiring, output shape, exit codes.

No GPU is required.  The four pipeline entry points the CLI reaches for
(``run_profile``, ``run_extract``, ``validate_reproducer`` and
``_get_replay_path``) are stubbed at their call sites, which is precisely
what makes the error branches reachable at all.

Fixtures build real ``KernelStat`` / ``ExtractResult`` / ``ValidationResult``
objects rather than ``MagicMock``, so renaming a field breaks these tests
instead of letting them pass silently.
"""

import logging
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from kerncap import cli
from kerncap.extract import ExtractResult
from kerncap.profiler import KernelStat
from kerncap.validator import ValidationResult


def all_output(result) -> str:
    """Return stdout plus stderr, across click versions.

    Click < 8.2 folds stderr into ``output`` and raises from ``stderr``;
    click >= 8.2 keeps them apart.  Tests that only care *that* a message
    was emitted use this rather than pinning a click version.
    """
    out = result.output or ""
    try:
        err = result.stderr or ""
    except ValueError:  # click < 8.2, mix_stderr=True
        err = ""
    return out + err


@pytest.fixture(autouse=True)
def reset_kerncap_logger():
    """Undo ``_setup_logging``'s global mutation between tests.

    Every CLI invocation calls ``_setup_logging``, which appends a handler
    to the ``kerncap`` logger and sets its level.  Without this fixture the
    handlers accumulate and ``_is_verbose()`` leaks a DEBUG level out of any
    test that passed ``-v``, which would make later assertions depend on
    test ordering.
    """
    log = logging.getLogger("kerncap")
    saved_handlers = list(log.handlers)
    saved_level = log.level
    saved_propagate = log.propagate
    yield
    log.handlers[:] = saved_handlers
    log.setLevel(saved_level)
    log.propagate = saved_propagate


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def kernels():
    """Three ranked kernels, already sorted by total duration descending."""
    return [
        KernelStat(
            name="matmul_kernel",
            calls=1024,
            total_duration_ns=580_000_000,
            avg_duration_ns=566_406,
            percentage=42.3,
            min_duration_ns=480_000,
            max_duration_ns=720_000,
            stddev_ns=45_000.0,
        ),
        KernelStat(
            name="softmax_kernel",
            calls=256,
            total_duration_ns=120_000_000,
            avg_duration_ns=468_750,
            percentage=8.7,
        ),
        KernelStat(
            name="add_kernel",
            calls=64,
            total_duration_ns=1_500_000,
            avg_duration_ns=23_437,
            percentage=0.1,
        ),
    ]


# --------------------------------------------------------------------------
# _CliFormatter
# --------------------------------------------------------------------------


def make_record(level: int, msg: str, name: str = "kerncap.extract") -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, msg, None, None)


class TestCliFormatter:
    """The formatter is what the user actually reads on stderr."""

    def test_error_is_red_when_colour_enabled(self):
        out = cli._CliFormatter(use_color=True).format(make_record(logging.ERROR, "boom"))
        assert out == "\n  \033[31mERROR\033[0m: boom"

    def test_error_is_plain_when_colour_disabled(self):
        out = cli._CliFormatter(use_color=False).format(make_record(logging.ERROR, "boom"))
        assert out == "\n  ERROR: boom"
        assert "\033" not in out

    def test_warning_is_yellow_when_colour_enabled(self):
        out = cli._CliFormatter(use_color=True).format(make_record(logging.WARNING, "careful"))
        assert out == "\n  \033[33mWARNING\033[0m: careful"

    def test_warning_is_plain_when_colour_disabled(self):
        out = cli._CliFormatter(use_color=False).format(make_record(logging.WARNING, "careful"))
        assert out == "\n  WARNING: careful"

    def test_info_is_bare_indented_message(self):
        """INFO carries no level prefix — it is the normal narration channel."""
        out = cli._CliFormatter(use_color=True).format(make_record(logging.INFO, "working"))
        assert out == "  working"

    def test_debug_includes_module_name_for_traceability(self):
        out = cli._CliFormatter(use_color=True).format(
            make_record(logging.DEBUG, "detail", name="kerncap.source_finder")
        )
        assert out == "  DEBUG (kerncap.source_finder): detail"

    def test_critical_uses_the_error_branch(self):
        """CRITICAL is >= ERROR, so it must not fall through to INFO."""
        out = cli._CliFormatter(use_color=False).format(make_record(logging.CRITICAL, "fatal"))
        assert out == "\n  CRITICAL: fatal".replace("CRITICAL", "ERROR")

    def test_message_args_are_interpolated(self):
        record = logging.LogRecord(
            "kerncap", logging.INFO, __file__, 1, "found %d kernels", (7,), None
        )
        assert cli._CliFormatter(use_color=False).format(record) == "  found 7 kernels"


class TestSetupLogging:
    def test_sets_level_and_stops_propagation(self):
        cli._setup_logging(logging.DEBUG)
        log = logging.getLogger("kerncap")
        assert log.level == logging.DEBUG
        assert log.propagate is False
        assert log.handlers

    def test_is_verbose_tracks_the_configured_level(self):
        cli._setup_logging(logging.INFO)
        assert cli._is_verbose() is False
        cli._setup_logging(logging.DEBUG)
        assert cli._is_verbose() is True


# --------------------------------------------------------------------------
# kerncap profile
# --------------------------------------------------------------------------


class TestProfileCommand:
    def test_prints_ranked_table(self, runner, kernels):
        # ``--`` is required before an app that takes its own dashed flags,
        # otherwise click claims them for the profile command.
        with patch("kerncap.profiler.run_profile", return_value=kernels) as run:
            result = runner.invoke(cli.main, ["profile", "--", "./my_app", "--flag"])

        assert result.exit_code == 0
        run.assert_called_once_with(["./my_app", "--flag"], output_path=None, timeout=None)
        assert "Profiling: ./my_app --flag" in result.output
        # Ranks are 1-based and in the order run_profile returned.
        assert "1     matmul_kernel" in result.output
        assert "2     softmax_kernel" in result.output
        assert "3     add_kernel" in result.output

    def test_converts_nanoseconds_for_display(self, runner, kernels):
        """Totals render as milliseconds, averages as microseconds."""
        with patch("kerncap.profiler.run_profile", return_value=kernels):
            result = runner.invoke(cli.main, ["profile", "./my_app"])

        # 580_000_000 ns -> 580.000 ms ; 566_406 ns -> 566.4 us
        assert "580.000" in result.output
        assert "566.4" in result.output
        assert "42.3" in result.output

    def test_truncates_to_twenty_rows(self, runner):
        """Only the top 20 kernels are tabulated, however many are returned."""
        many = [
            KernelStat(
                name=f"kernel_{i:02d}",
                calls=1,
                total_duration_ns=1_000_000 * (30 - i),
                avg_duration_ns=1000,
                percentage=1.0,
            )
            for i in range(30)
        ]
        with patch("kerncap.profiler.run_profile", return_value=many):
            result = runner.invoke(cli.main, ["profile", "./my_app"])

        assert "kernel_19" in result.output
        assert "kernel_20" not in result.output

    def test_truncates_long_kernel_names(self, runner):
        """Names are clipped to 58 chars so the columns stay aligned."""
        long_name = "z" * 120
        stat = KernelStat(
            name=long_name,
            calls=1,
            total_duration_ns=1_000,
            avg_duration_ns=1_000,
            percentage=100.0,
        )
        with patch("kerncap.profiler.run_profile", return_value=[stat]):
            result = runner.invoke(cli.main, ["profile", "./my_app"])

        assert "z" * 58 in result.output
        assert "z" * 59 not in result.output

    def test_empty_profile_prints_notice_and_no_table(self, runner):
        with patch("kerncap.profiler.run_profile", return_value=[]):
            result = runner.invoke(cli.main, ["profile", "./my_app"])

        assert result.exit_code == 0
        assert "No kernels found in profile." in result.output
        assert "Rank" not in result.output

    def test_output_option_is_forwarded_and_confirmed(self, runner, kernels):
        with patch("kerncap.profiler.run_profile", return_value=kernels) as run:
            result = runner.invoke(cli.main, ["profile", "-o", "prof.json", "./my_app"])

        assert run.call_args.kwargs["output_path"] == "prof.json"
        assert "Profile saved to prof.json" in result.output

    def test_no_save_confirmation_without_output(self, runner, kernels):
        with patch("kerncap.profiler.run_profile", return_value=kernels):
            result = runner.invoke(cli.main, ["profile", "./my_app"])
        assert "Profile saved to" not in result.output

    def test_timeout_option_is_forwarded_as_int(self, runner, kernels):
        with patch("kerncap.profiler.run_profile", return_value=kernels) as run:
            runner.invoke(cli.main, ["profile", "--timeout", "45", "./my_app"])
        assert run.call_args.kwargs["timeout"] == 45

    def test_failure_exits_one_with_message(self, runner):
        with patch("kerncap.profiler.run_profile", side_effect=RuntimeError("rocprofv3 missing")):
            result = runner.invoke(cli.main, ["profile", "./my_app"])

        assert result.exit_code == 1
        assert "Error: rocprofv3 missing" in all_output(result)

    def test_command_is_required(self, runner):
        result = runner.invoke(cli.main, ["profile"])
        assert result.exit_code != 0


# --------------------------------------------------------------------------
# kerncap extract
# --------------------------------------------------------------------------


class TestExtractCommand:
    def test_forwards_every_option(self, runner):
        extracted = ExtractResult(
            output_dir="/out",
            capture_dir="/out/capture",
            language="hip",
            has_source=True,
            generated_files=["kernel_variant.cpp", "Makefile"],
        )
        with patch("kerncap.extract.run_extract", return_value=extracted) as run:
            result = runner.invoke(
                cli.main,
                [
                    "extract",
                    "mul_mat_q",
                    "--cmd",
                    "./llama-bench -m model.gguf",
                    "--source-dir",
                    "./ggml/src",
                    "-o",
                    "/out",
                    "--language",
                    "hip",
                    "--dispatch",
                    "3",
                    "-D",
                    "GGML_USE_HIP",
                    "-D",
                    "GGML_CUDA_FA_ALL_QUANTS",
                    "--timeout",
                    "120",
                ],
            )

        assert result.exit_code == 0
        kwargs = run.call_args.kwargs
        assert kwargs["kernel_name"] == "mul_mat_q"
        assert kwargs["cmd"] == "./llama-bench -m model.gguf"
        assert kwargs["source_dir"] == "./ggml/src"
        assert kwargs["output"] == "/out"
        assert kwargs["language"] == "hip"
        assert kwargs["dispatch"] == 3
        assert kwargs["defines"] == ["GGML_USE_HIP", "GGML_CUDA_FA_ALL_QUANTS"]
        assert kwargs["timeout"] == 120

    def test_defaults(self, runner):
        """Unspecified options must reach run_extract as documented defaults."""
        extracted = ExtractResult(output_dir="/out", capture_dir="/out/capture", language="hip")
        with patch("kerncap.extract.run_extract", return_value=extracted) as run:
            runner.invoke(cli.main, ["extract", "vector_add", "--cmd", "./app"])

        kwargs = run.call_args.kwargs
        assert kwargs["source_dir"] is None
        assert kwargs["output"] is None
        assert kwargs["language"] is None
        assert kwargs["dispatch"] == -1
        assert kwargs["timeout"] == 300
        assert kwargs["triton_backend"] == "hsa"

    def test_no_defines_becomes_none_not_empty_list(self, runner):
        """``-D`` is variadic; absent must be None so run_extract can tell
        'no defines given' from 'an empty list was requested'."""
        extracted = ExtractResult(output_dir="/out", capture_dir="/out/capture", language="hip")
        with patch("kerncap.extract.run_extract", return_value=extracted) as run:
            runner.invoke(cli.main, ["extract", "k", "--cmd", "./app"])
        assert run.call_args.kwargs["defines"] is None

    def test_prints_generated_files_and_next_steps(self, runner):
        extracted = ExtractResult(
            output_dir="/isolated/vec",
            capture_dir="/isolated/vec/capture",
            language="hip",
            generated_files=["kernel_variant.cpp", "Makefile"],
        )
        with patch("kerncap.extract.run_extract", return_value=extracted):
            result = runner.invoke(cli.main, ["extract", "vec", "--cmd", "./app"])

        assert "Generated: kernel_variant.cpp, Makefile" in result.output
        assert "Done." in result.output
        assert "Next steps:" in result.output

    def test_failure_exits_one_with_message(self, runner):
        with patch("kerncap.extract.run_extract", side_effect=RuntimeError("no dispatch matched")):
            result = runner.invoke(cli.main, ["extract", "nope", "--cmd", "./app"])

        assert result.exit_code == 1
        assert "Extract failed: no dispatch matched" in all_output(result)

    def test_cmd_is_required(self, runner):
        result = runner.invoke(cli.main, ["extract", "vec"])
        assert result.exit_code != 0

    def test_rejects_unknown_language(self, runner):
        result = runner.invoke(cli.main, ["extract", "vec", "--cmd", "./app", "--language", "cuda"])
        assert result.exit_code != 0

    def test_rejects_unknown_triton_backend(self, runner):
        result = runner.invoke(
            cli.main, ["extract", "vec", "--cmd", "./app", "--triton-backend", "ctypes"]
        )
        assert result.exit_code != 0

    def test_accepts_legacy_python_triton_backend(self, runner):
        extracted = ExtractResult(output_dir="/out", capture_dir="/out/capture", language="triton")
        with patch("kerncap.extract.run_extract", return_value=extracted) as run:
            result = runner.invoke(
                cli.main, ["extract", "k", "--cmd", "./app", "--triton-backend", "python"]
            )
        assert result.exit_code == 0
        assert run.call_args.kwargs["triton_backend"] == "python"


# --------------------------------------------------------------------------
# _pick_edit_file / _print_next_steps
# --------------------------------------------------------------------------


class TestPickEditFile:
    def test_triton_prefers_kernel_variant_py(self):
        result = ExtractResult(
            output_dir="/o",
            capture_dir="/o/capture",
            language="triton",
            generated_files=["reproducer.py", "kernel_variant.py", "attn.py"],
        )
        assert cli._pick_edit_file(result) == "kernel_variant.py"

    def test_triton_falls_back_to_first_other_python_file(self):
        """Flat-file path: a copy of the user's original source."""
        result = ExtractResult(
            output_dir="/o",
            capture_dir="/o/capture",
            language="triton",
            generated_files=["reproducer.py", "attn.py", "helpers.py"],
        )
        assert cli._pick_edit_file(result) == "attn.py"

    def test_triton_never_picks_reproducer_py(self):
        result = ExtractResult(
            output_dir="/o",
            capture_dir="/o/capture",
            language="triton",
            generated_files=["reproducer.py"],
        )
        assert cli._pick_edit_file(result) is None

    def test_triton_ignores_non_python_files(self):
        result = ExtractResult(
            output_dir="/o",
            capture_dir="/o/capture",
            language="triton",
            generated_files=["capture", "metadata.json"],
        )
        assert cli._pick_edit_file(result) is None

    def test_hip_picks_kernel_variant_cpp(self):
        result = ExtractResult(
            output_dir="/o",
            capture_dir="/o/capture",
            language="hip",
            generated_files=["kernel_variant.cpp", "Makefile", "vfs.yaml"],
        )
        assert cli._pick_edit_file(result) == "kernel_variant.cpp"

    def test_hip_without_variant_returns_none(self):
        """HIP has no fallback — an HSACO-only capture has nothing to edit."""
        result = ExtractResult(
            output_dir="/o",
            capture_dir="/o/capture",
            language="hip",
            generated_files=["Makefile", "harness.hip"],
        )
        assert cli._pick_edit_file(result) is None

    def test_empty_generated_files(self):
        result = ExtractResult(output_dir="/o", capture_dir="/o/capture", language="hip")
        assert cli._pick_edit_file(result) is None

    def test_none_generated_files_is_tolerated(self):
        result = ExtractResult(output_dir="/o", capture_dir="/o/capture", language="triton")
        result.generated_files = None
        assert cli._pick_edit_file(result) is None


class TestPrintNextSteps:
    def test_triton_rebuild_runs_the_reproducer(self, runner):
        result = ExtractResult(
            output_dir="/iso/attn",
            capture_dir="/iso/attn/capture",
            language="triton",
            generated_files=["kernel_variant.py", "reproducer.py"],
        )
        with runner.isolation() as streams:
            cli._print_next_steps(result)
            # Read inside the block: Click closes these streams on exit,
            # so getvalue() afterwards raises "I/O operation on closed file".
            out = streams[0].getvalue().decode()

        assert "edit:    /iso/attn/kernel_variant.py" in out
        assert "rebuild: cd /iso/attn && python3 reproducer.py" in out
        assert "verify:  kerncap validate /iso/attn" in out

    def test_hip_rebuild_uses_make(self, runner):
        result = ExtractResult(
            output_dir="/iso/gemm",
            capture_dir="/iso/gemm/capture",
            language="hip",
            generated_files=["kernel_variant.cpp"],
        )
        with runner.isolation() as streams:
            cli._print_next_steps(result)
            # Read inside the block: Click closes these streams on exit,
            # so getvalue() afterwards raises "I/O operation on closed file".
            out = streams[0].getvalue().decode()

        assert "edit:    /iso/gemm/kernel_variant.cpp" in out
        assert "rebuild: cd /iso/gemm && make recompile" in out

    def test_edit_line_omitted_when_nothing_is_editable(self, runner):
        """rebuild/verify still print — only the edit line is conditional."""
        result = ExtractResult(
            output_dir="/iso/blob",
            capture_dir="/iso/blob/capture",
            language="hip",
            generated_files=["Makefile"],
        )
        with runner.isolation() as streams:
            cli._print_next_steps(result)
            # Read inside the block: Click closes these streams on exit,
            # so getvalue() afterwards raises "I/O operation on closed file".
            out = streams[0].getvalue().decode()

        assert "edit:" not in out
        assert "rebuild:" in out
        assert "verify:" in out


# --------------------------------------------------------------------------
# kerncap replay
# --------------------------------------------------------------------------


def completed(returncode=0, stdout="", stderr=""):
    """A stand-in for ``subprocess.CompletedProcess`` with only what replay reads."""
    import subprocess

    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


REPLAY_STDOUT = "Replaying vector_add.kd\nAverage GPU time: 12.5 us\nMin: 11.0 us\nMax: 14.25 us\n"


class TestReplayCommand:
    def test_prefers_the_capture_subdirectory(self, runner, tmp_path):
        (tmp_path / "capture").mkdir()
        with (
            patch("kerncap._get_replay_path", return_value="/usr/bin/kerncap-replay"),
            patch("subprocess.run", return_value=completed()) as run,
        ):
            runner.invoke(cli.main, ["replay", str(tmp_path)])

        assert run.call_args.args[0][1] == str(tmp_path / "capture")

    def test_falls_back_to_the_directory_itself(self, runner, tmp_path):
        """Pointing straight at a capture/ dir must also work."""
        with (
            patch("kerncap._get_replay_path", return_value="/usr/bin/kerncap-replay"),
            patch("subprocess.run", return_value=completed()) as run,
        ):
            runner.invoke(cli.main, ["replay", str(tmp_path)])

        assert run.call_args.args[0][1] == str(tmp_path)

    def test_missing_replay_binary_exits_one(self, runner, tmp_path):
        with patch("kerncap._get_replay_path", side_effect=FileNotFoundError("no kerncap-replay")):
            result = runner.invoke(cli.main, ["replay", str(tmp_path)])

        assert result.exit_code == 1
        assert "Error: no kerncap-replay" in all_output(result)

    def test_bare_invocation_passes_no_extra_flags(self, runner, tmp_path):
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("subprocess.run", return_value=completed()) as run,
        ):
            runner.invoke(cli.main, ["replay", str(tmp_path)])

        assert run.call_args.args[0] == ["/bin/replay", str(tmp_path)]

    def test_single_iteration_is_left_implicit(self, runner, tmp_path):
        """``--iterations 1`` is the default, so it is not forwarded."""
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("subprocess.run", return_value=completed()) as run,
        ):
            runner.invoke(cli.main, ["replay", str(tmp_path), "-n", "1"])

        assert "--iterations" not in run.call_args.args[0]

    def test_multiple_iterations_are_forwarded(self, runner, tmp_path):
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("subprocess.run", return_value=completed()) as run,
        ):
            runner.invoke(cli.main, ["replay", str(tmp_path), "-n", "50"])

        argv = run.call_args.args[0]
        assert argv[argv.index("--iterations") + 1] == "50"

    def test_all_flags_are_forwarded(self, runner, tmp_path):
        hsaco = tmp_path / "optimized.hsaco"
        hsaco.write_bytes(b"\x7fELF")
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("subprocess.run", return_value=completed()) as run,
        ):
            runner.invoke(
                cli.main,
                [
                    "replay",
                    str(tmp_path),
                    "--hsaco",
                    str(hsaco),
                    "--dump-output",
                    "--hip-launch",
                ],
            )

        argv = run.call_args.args[0]
        assert argv[argv.index("--hsaco") + 1] == str(hsaco)
        assert "--dump-output" in argv
        assert "--hip-launch" in argv

    def test_nonexistent_hsaco_is_rejected_before_launching(self, runner, tmp_path):
        with patch("subprocess.run") as run:
            result = runner.invoke(
                cli.main, ["replay", str(tmp_path), "--hsaco", str(tmp_path / "missing.hsaco")]
            )
        assert result.exit_code != 0
        run.assert_not_called()

    def test_strips_kd_suffix_and_appends_result_line(self, runner, tmp_path):
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("subprocess.run", return_value=completed(stdout=REPLAY_STDOUT)),
        ):
            result = runner.invoke(cli.main, ["replay", str(tmp_path)])

        assert "Replaying vector_add" in result.output
        assert "vector_add.kd" not in result.output
        assert "Result: PASS" in result.output

    def test_extracts_timing_into_the_result_line(self, runner, tmp_path):
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("subprocess.run", return_value=completed(stdout=REPLAY_STDOUT)),
        ):
            result = runner.invoke(cli.main, ["replay", str(tmp_path)])

        assert "avg = 12.50 us" in result.output
        assert "min = 11.00" in result.output
        assert "max = 14.25" in result.output

    def test_json_mode_emits_raw_stdout_only(self, runner, tmp_path):
        """Byte-exactness matters here: downstream parsers read this."""
        payload = '{"kernel": "vector_add.kd", "avg_us": 12.5}'
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("subprocess.run", return_value=completed(stdout=payload)),
        ):
            result = runner.invoke(cli.main, ["replay", str(tmp_path), "--json"])

        assert result.output.strip() == payload
        # No Result footer, and .kd is deliberately *not* stripped.
        assert "Result:" not in result.output
        assert "vector_add.kd" in result.output

    def test_json_mode_still_reports_stderr_on_failure(self, runner, tmp_path):
        """Machine-readable stdout must not swallow the diagnostic on stderr."""
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch(
                "subprocess.run",
                return_value=completed(returncode=1, stdout="{}", stderr="hsa init failed\n"),
            ),
        ):
            result = runner.invoke(cli.main, ["replay", str(tmp_path), "--json"])

        assert result.exit_code == 1
        assert "hsa init failed" in all_output(result)

    def test_json_flag_is_forwarded(self, runner, tmp_path):
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("subprocess.run", return_value=completed(stdout="{}")) as run,
        ):
            runner.invoke(cli.main, ["replay", str(tmp_path), "--json"])
        assert "--json" in run.call_args.args[0]

    def test_stderr_is_suppressed_on_success(self, runner, tmp_path):
        """Stage chatter is noise unless something went wrong or -v was given."""
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch(
                "subprocess.run",
                return_value=completed(stdout="ok\n", stderr="Stage 0: setup\n"),
            ),
        ):
            result = runner.invoke(cli.main, ["replay", str(tmp_path)])

        assert "Stage 0" not in all_output(result)

    def test_stderr_is_shown_on_failure(self, runner, tmp_path):
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch(
                "subprocess.run",
                return_value=completed(returncode=3, stderr="hsa error\n"),
            ),
        ):
            result = runner.invoke(cli.main, ["replay", str(tmp_path)])

        assert "hsa error" in all_output(result)

    def test_stderr_is_shown_under_top_level_verbose(self, runner, tmp_path):
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch(
                "subprocess.run",
                return_value=completed(stdout="ok\n", stderr="Stage 0: setup\n"),
            ),
        ):
            result = runner.invoke(cli.main, ["-v", "replay", str(tmp_path)])

        assert "Stage 0" in all_output(result)

    def test_propagates_the_replay_exit_code(self, runner, tmp_path):
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("subprocess.run", return_value=completed(returncode=3)),
        ):
            result = runner.invoke(cli.main, ["replay", str(tmp_path)])

        assert result.exit_code == 3
        assert "Result: FAIL" in result.output

    def test_json_mode_propagates_the_exit_code(self, runner, tmp_path):
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("subprocess.run", return_value=completed(returncode=2, stdout="{}")),
        ):
            result = runner.invoke(cli.main, ["replay", str(tmp_path), "--json"])

        assert result.exit_code == 2

    def test_tolerates_none_streams(self, runner, tmp_path):
        """``proc.stdout``/``stderr`` are None when not captured as text."""
        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("subprocess.run", return_value=completed(stdout=None, stderr=None)),
        ):
            result = runner.invoke(cli.main, ["replay", str(tmp_path)])

        assert result.exit_code == 0
        assert "Result: PASS" in result.output


# --------------------------------------------------------------------------
# kerncap validate
# --------------------------------------------------------------------------


class TestValidateCommand:
    def test_passing_smoke_validation_exits_zero(self, runner, tmp_path):
        vr = ValidationResult(passed=True, details=["Replay succeeded"], mode="smoke")
        with patch("kerncap.validator.validate_reproducer", return_value=vr) as run:
            result = runner.invoke(cli.main, ["validate", str(tmp_path)])

        assert result.exit_code == 0
        assert f"Validating reproducer at {tmp_path} ..." in result.output
        assert "Replay succeeded" in result.output
        assert "Result: PASS" in result.output
        assert run.call_args.kwargs["tolerance"] == pytest.approx(1e-6)
        assert run.call_args.kwargs["rtol"] == pytest.approx(1e-5)
        assert run.call_args.kwargs["hsaco"] is None

    def test_failing_validation_exits_one(self, runner, tmp_path):
        vr = ValidationResult(
            passed=False,
            details=["region_0x1000.bin: FAIL"],
            mode="byte-exact",
            regions_total=4,
            regions_identical=3,
        )
        with patch("kerncap.validator.validate_reproducer", return_value=vr):
            result = runner.invoke(cli.main, ["validate", str(tmp_path)])

        assert result.exit_code == 1
        assert "Result: FAIL" in result.output

    def test_tolerances_are_forwarded(self, runner, tmp_path):
        vr = ValidationResult(passed=True, details=[], mode="numeric", atol=1e-4)
        with patch("kerncap.validator.validate_reproducer", return_value=vr) as run:
            runner.invoke(cli.main, ["validate", str(tmp_path), "-t", "1e-4", "--rtol", "1e-3"])

        assert run.call_args.kwargs["tolerance"] == pytest.approx(1e-4)
        assert run.call_args.kwargs["rtol"] == pytest.approx(1e-3)

    def test_hsaco_is_forwarded_and_announced(self, runner, tmp_path):
        hsaco = tmp_path / "candidate.hsaco"
        hsaco.write_bytes(b"\x7fELF")
        vr = ValidationResult(passed=True, details=[], mode="byte-exact")
        with patch("kerncap.validator.validate_reproducer", return_value=vr) as run:
            result = runner.invoke(cli.main, ["validate", str(tmp_path), "--hsaco", str(hsaco)])

        assert f"Using HSACO: {hsaco}" in result.output
        assert run.call_args.kwargs["hsaco"] == str(hsaco)

    def test_nonexistent_hsaco_is_rejected(self, runner, tmp_path):
        with patch("kerncap.validator.validate_reproducer") as run:
            result = runner.invoke(
                cli.main, ["validate", str(tmp_path), "--hsaco", str(tmp_path / "gone.hsaco")]
            )
        assert result.exit_code != 0
        run.assert_not_called()

    def test_failure_exits_one_with_message(self, runner, tmp_path):
        with patch(
            "kerncap.validator.validate_reproducer",
            side_effect=RuntimeError("capture/ not found"),
        ):
            result = runner.invoke(cli.main, ["validate", str(tmp_path)])

        assert result.exit_code == 1
        assert "Validation error: capture/ not found" in all_output(result)

    def test_details_have_kd_stripped(self, runner, tmp_path):
        vr = ValidationResult(passed=True, details=["ran triton_poi_fused_relu_0.kd"], mode="smoke")
        with patch("kerncap.validator.validate_reproducer", return_value=vr):
            result = runner.invoke(cli.main, ["validate", str(tmp_path)])

        assert "triton_poi_fused_relu_0" in result.output
        assert ".kd" not in result.output

    def test_region_lines_hidden_when_passing(self, runner, tmp_path):
        """Per-region PASS noise is suppressed on a clean run."""
        vr = ValidationResult(
            passed=True,
            details=["4 of 4 regions identical"],
            mode="byte-exact",
            region_lines=["region_0x1000.bin: PASS (identical)"],
        )
        with patch("kerncap.validator.validate_reproducer", return_value=vr):
            result = runner.invoke(cli.main, ["validate", str(tmp_path)])

        assert "region_0x1000.bin" not in result.output

    def test_region_lines_shown_with_subcommand_verbose(self, runner, tmp_path):
        vr = ValidationResult(
            passed=True,
            details=[],
            mode="byte-exact",
            region_lines=["region_0x1000.bin: PASS (identical)"],
        )
        with patch("kerncap.validator.validate_reproducer", return_value=vr):
            result = runner.invoke(cli.main, ["validate", str(tmp_path), "-v"])

        assert "region_0x1000.bin: PASS" in result.output

    def test_region_lines_shown_under_top_level_verbose(self, runner, tmp_path):
        vr = ValidationResult(
            passed=True,
            details=[],
            mode="byte-exact",
            region_lines=["region_0x1000.bin: PASS (identical)"],
        )
        with patch("kerncap.validator.validate_reproducer", return_value=vr):
            result = runner.invoke(cli.main, ["-v", "validate", str(tmp_path)])

        assert "region_0x1000.bin: PASS" in result.output

    def test_region_lines_shown_on_failure_without_verbose(self, runner, tmp_path):
        """A failing run always shows the detail, verbose or not."""
        vr = ValidationResult(
            passed=False,
            details=[],
            mode="byte-exact",
            region_lines=["region_0x2000.bin: DIFFERS"],
            regions_total=2,
            regions_identical=1,
        )
        with patch("kerncap.validator.validate_reproducer", return_value=vr):
            result = runner.invoke(cli.main, ["validate", str(tmp_path)])

        assert "region_0x2000.bin: DIFFERS" in result.output


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class TestGrepFloat:
    def test_extracts_a_match(self):
        assert cli._grep_float("Average GPU time: 12.5 us", r"time:\s*([\d.]+)") == 12.5

    def test_returns_none_without_a_match(self):
        assert cli._grep_float("nothing here", r"time:\s*([\d.]+)") is None

    def test_honours_regex_flags(self):
        import re

        text = "Min: 3.0 us\nMax: 9.0 us"
        assert cli._grep_float(text, r"^Max:\s*([\d.]+)", flags=re.MULTILINE) == 9.0

    def test_parses_scientific_notation(self):
        assert cli._grep_float("v = 1.5e-3", r"v = ([\d.eE+\-]+)") == pytest.approx(1.5e-3)

    def test_unparseable_group_returns_none(self):
        """A match whose group is not a float must not raise."""
        assert cli._grep_float("value: abc", r"value: (\w+)") is None

    def test_empty_text(self):
        assert cli._grep_float("", r"(\d+)") is None


class TestStripKd:
    def test_strips_the_suffix(self):
        assert cli._strip_kd("triton_poi_fused_relu_0.kd") == "triton_poi_fused_relu_0"

    def test_strips_every_occurrence(self):
        assert cli._strip_kd("a.kd and b.kd") == "a and b"

    def test_leaves_unrelated_text_alone(self):
        assert cli._strip_kd("no suffix here") == "no suffix here"

    def test_strips_kd_even_mid_filename(self):
        """``\\b`` after ``.kd`` matches before a following dot, so ``.kd`` is
        removed from the middle of a filename too.

        Pinning this because it is presentation-only: the on-disk artifacts
        keep the raw HSA symbol, so a stripped display name is harmless. If
        this ever feeds a path, this test is where it will surface.
        """
        assert cli._strip_kd("vector_add.kd.bin") == "vector_add.bin"

    def test_leaves_non_identifier_prefixes_alone(self):
        """The pattern requires an identifier before ``.kd``."""
        assert cli._strip_kd("9.kd") == "9.kd"

    def test_preserves_surrounding_whitespace_and_lines(self):
        assert cli._strip_kd("  ran k1.kd\n  ran k2.kd\n") == "  ran k1\n  ran k2\n"

    def test_empty_string(self):
        assert cli._strip_kd("") == ""


class TestTopLevelGroup:
    def test_verbose_flag_enables_debug_logging(self, runner):
        with patch("kerncap.profiler.run_profile", return_value=[]):
            runner.invoke(cli.main, ["-v", "profile", "./app"])
        assert logging.getLogger("kerncap").level == logging.DEBUG

    def test_default_is_info_logging(self, runner):
        with patch("kerncap.profiler.run_profile", return_value=[]):
            runner.invoke(cli.main, ["profile", "./app"])
        assert logging.getLogger("kerncap").level == logging.INFO

    def test_help_lists_every_subcommand(self, runner):
        result = runner.invoke(cli.main, ["--help"])
        assert result.exit_code == 0
        for name in ("profile", "extract", "replay", "validate"):
            assert name in result.output
