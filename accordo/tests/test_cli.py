# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the Accordo CLI.

``_run_validate`` ends in ``os._exit`` and rearranges file descriptors 1 and 2,
neither of which a test can survive directly.  Both are patched here so the
function can be driven to completion and its JSON payload inspected.
"""

import argparse
import io
import json
from unittest.mock import MagicMock, patch

import pytest

from accordo import cli


class _Exited(Exception):
    """Stands in for os._exit so the call is observable instead of fatal."""

    def __init__(self, code):
        super().__init__(f"exit {code}")
        self.code = code


def _exit_raiser(code):
    """side_effect for os._exit.

    Passing the exception *class* as side_effect will not do: mock raises it
    bare, losing the exit code. A callable receives the real argument.
    """
    raise _Exited(code)


def _mismatch(idx=0, name="output", typ="float*", max_diff=1.0, mean_diff=0.5, dispatch=None):
    m = MagicMock()
    m.arg_index = idx
    m.arg_name = name
    m.arg_type = typ
    m.max_difference = max_diff
    m.mean_difference = mean_diff
    m.dispatch_index = dispatch
    return m


def _result(is_valid=True, mismatches=None, matched=None, error=None):
    r = MagicMock()
    r.is_valid = is_valid
    r.num_arrays_validated = 2
    r.num_mismatches = len(mismatches or [])
    r.summary.return_value = "PASS" if is_valid else "FAIL"
    r.error_message = error
    r.matched_arrays = matched if matched is not None else {"output": True}
    r.mismatches = mismatches
    return r


def _args(**over):
    base = dict(
        kernel_name="reduce_sum",
        ref_binary="./ref",
        opt_binary="./opt",
        tolerance=None,
        atol=1e-08,
        rtol=1e-05,
        equal_nan=False,
        timeout=30,
        working_dir=".",
        kernel_args=None,
        log_level="WARNING",
    )
    base.update(over)
    return argparse.Namespace(**base)


def _drive(args, result=None, accordo_side_effect=None):
    """Run _run_validate with fds and process exit neutralised; return (code, payload)."""
    buf = io.StringIO()
    validator = MagicMock()
    validator.capture_snapshot.return_value = MagicMock()
    if result is not None:
        validator.compare_snapshots.return_value = result

    accordo_cls = MagicMock(return_value=validator)
    if accordo_side_effect is not None:
        accordo_cls.side_effect = accordo_side_effect

    with (
        patch("accordo.validator.Accordo", accordo_cls),
        patch("accordo.cli.os.dup", return_value=99),
        patch("accordo.cli.os.dup2"),
        patch("accordo.cli.os.close"),
        patch("accordo.cli.os.fdopen", return_value=buf),
        patch("accordo.cli.os._exit", side_effect=_exit_raiser),
    ):
        with pytest.raises(_Exited) as excinfo:
            cli._run_validate(args)

    return excinfo.value.code, json.loads(buf.getvalue()), validator


class TestParseKernelArgs:
    def test_single_pair(self):
        assert cli._parse_kernel_args("input:const float*") == [("input", "const float*")]

    def test_multiple_pairs(self):
        assert cli._parse_kernel_args("a:int,b:float*") == [("a", "int"), ("b", "float*")]

    def test_whitespace_is_stripped(self):
        assert cli._parse_kernel_args("  a : int , b : float*  ") == [("a", "int"), ("b", "float*")]

    def test_empty_items_skipped(self):
        assert cli._parse_kernel_args("a:int,,b:float*,") == [("a", "int"), ("b", "float*")]

    def test_empty_string_yields_nothing(self):
        assert cli._parse_kernel_args("") == []

    def test_type_may_contain_colons(self):
        # split(":", 1) keeps the remainder intact
        assert cli._parse_kernel_args("v:std::vector<int>") == [("v", "std::vector<int>")]

    def test_missing_colon_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid kernel-arg format"):
            cli._parse_kernel_args("noseparator")


class TestRunValidate:
    def test_success_payload(self):
        code, out, _ = _drive(_args(), result=_result())
        assert code == 0
        assert out["is_valid"] is True
        assert out["num_arrays_validated"] == 2
        assert out["num_mismatches"] == 0
        assert out["summary"] == "PASS"
        assert out["mismatches"] == []
        assert "error" not in out

    def test_failure_still_exits_zero(self):
        """A failed validation is a result, not a tool error."""
        code, out, _ = _drive(_args(), result=_result(is_valid=False, mismatches=[_mismatch()]))
        assert code == 0
        assert out["is_valid"] is False
        assert out["num_mismatches"] == 1

    def test_mismatch_fields_serialized(self):
        _, out, _ = _drive(
            _args(),
            result=_result(
                is_valid=False, mismatches=[_mismatch(idx=3, name="acc", typ="double*")]
            ),
        )
        m = out["mismatches"][0]
        assert m["arg_index"] == 3
        assert m["arg_name"] == "acc"
        assert m["arg_type"] == "double*"
        assert m["max_difference"] == 1.0
        assert m["mean_difference"] == 0.5

    def test_dispatch_index_omitted_when_none(self):
        _, out, _ = _drive(
            _args(), result=_result(is_valid=False, mismatches=[_mismatch(dispatch=None)])
        )
        assert "dispatch_index" not in out["mismatches"][0]

    def test_dispatch_index_included_when_set(self):
        _, out, _ = _drive(
            _args(), result=_result(is_valid=False, mismatches=[_mismatch(dispatch=7)])
        )
        assert out["mismatches"][0]["dispatch_index"] == 7

    def test_none_mismatches_treated_as_empty(self):
        _, out, _ = _drive(_args(), result=_result(mismatches=None))
        assert out["mismatches"] == []

    def test_none_matched_arrays_becomes_empty_dict(self):
        _, out, _ = _drive(_args(), result=_result(matched=None))
        assert isinstance(out["matched_arrays"], dict)

    def test_tolerance_and_flags_forwarded(self):
        args = _args(tolerance=1e-3, atol=1e-9, rtol=1e-4, equal_nan=True)
        _, _, validator = _drive(args, result=_result())
        kwargs = validator.compare_snapshots.call_args.kwargs
        assert kwargs["tolerance"] == 1e-3
        assert kwargs["atol"] == 1e-9
        assert kwargs["rtol"] == 1e-4
        assert kwargs["equal_nan"] is True

    def test_both_binaries_captured_with_timeout(self):
        _, _, validator = _drive(_args(timeout=45), result=_result())
        binaries = [c.kwargs["binary"] for c in validator.capture_snapshot.call_args_list]
        assert binaries == ["./ref", "./opt"]
        for call in validator.capture_snapshot.call_args_list:
            assert call.kwargs["timeout_seconds"] == 45

    def test_accordo_error_reported_and_exits_one(self):
        from accordo.exceptions import AccordoError

        code, out, _ = _drive(_args(), accordo_side_effect=AccordoError("kernel never dispatched"))
        assert code == 1
        assert out["error"] == "kernel never dispatched"

    def test_unexpected_exception_is_labelled(self):
        code, out, _ = _drive(_args(), accordo_side_effect=RuntimeError("boom"))
        assert code == 1
        assert out["error"].startswith("Unexpected error:")
        assert "boom" in out["error"]

    def test_stdout_is_restored_before_writing(self):
        """The saved fd must be duplicated back over fd 1 before the JSON is emitted."""
        validator = MagicMock()
        validator.capture_snapshot.return_value = MagicMock()
        validator.compare_snapshots.return_value = _result()

        with (
            patch("accordo.validator.Accordo", MagicMock(return_value=validator)),
            patch("accordo.cli.os.dup", return_value=99) as dup,
            patch("accordo.cli.os.dup2") as dup2,
            patch("accordo.cli.os.close") as close,
            patch("accordo.cli.os.fdopen", return_value=io.StringIO()),
            patch("accordo.cli.os._exit", side_effect=_exit_raiser),
        ):
            with pytest.raises(_Exited):
                cli._run_validate(_args())

        dup.assert_called_once_with(1)
        assert dup2.call_args_list[0].args == (2, 1)  # stdout -> stderr during run
        assert dup2.call_args_list[-1].args == (99, 1)  # restored afterwards
        close.assert_called_once_with(99)  # saved fd released


class TestMain:
    def test_no_subcommand_prints_help_and_exits_one(self, capsys):
        with patch("sys.argv", ["accordo"]):
            with pytest.raises(SystemExit) as exc:
                cli.main()
        assert exc.value.code == 1
        assert "usage" in capsys.readouterr().out.lower()

    def test_validate_dispatches_to_run_validate(self):
        argv = [
            "accordo",
            "validate",
            "--kernel-name",
            "k",
            "--ref-binary",
            "r",
            "--opt-binary",
            "o",
        ]
        with patch("sys.argv", argv), patch.object(cli, "_run_validate", return_value=0) as run:
            with pytest.raises(SystemExit) as exc:
                cli.main()
        assert exc.value.code == 0
        run.assert_called_once()

    def test_required_arguments_enforced(self):
        with patch("sys.argv", ["accordo", "validate", "--kernel-name", "k"]):
            with pytest.raises(SystemExit) as exc:
                cli.main()
        assert exc.value.code != 0

    def test_parser_defaults(self):
        argv = [
            "accordo",
            "validate",
            "--kernel-name",
            "k",
            "--ref-binary",
            "r",
            "--opt-binary",
            "o",
        ]
        captured = {}
        with (
            patch("sys.argv", argv),
            patch.object(cli, "_run_validate", side_effect=lambda a: captured.update(vars(a)) or 0),
        ):
            with pytest.raises(SystemExit):
                cli.main()
        assert captured["atol"] == 1e-08
        assert captured["rtol"] == 1e-05
        assert captured["timeout"] == 30
        assert captured["working_dir"] == "."
        assert captured["log_level"] == "WARNING"
        assert captured["equal_nan"] is False
        assert captured["tolerance"] is None

    def test_kernel_args_parsed_through_argparse(self):
        argv = [
            "accordo",
            "validate",
            "--kernel-name",
            "k",
            "--ref-binary",
            "r",
            "--opt-binary",
            "o",
            "--kernel-args",
            "input:const float*,n:int",
        ]
        captured = {}
        with (
            patch("sys.argv", argv),
            patch.object(cli, "_run_validate", side_effect=lambda a: captured.update(vars(a)) or 0),
        ):
            with pytest.raises(SystemExit):
                cli.main()
        assert captured["kernel_args"] == [("input", "const float*"), ("n", "int")]

    def test_invalid_log_level_rejected(self):
        argv = [
            "accordo",
            "validate",
            "--kernel-name",
            "k",
            "--ref-binary",
            "r",
            "--opt-binary",
            "o",
            "--log-level",
            "TRACE",
        ]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit) as exc:
                cli.main()
        assert exc.value.code != 0
