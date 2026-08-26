# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for capture_snapshot's error translation and CLI import guard.

``capture_snapshot`` wraps the instrumented run in a SIGALRM timeout and maps
three distinct failures onto Accordo's own exception types. That mapping is the
contract callers depend on, and it is reachable without a GPU by substituting
``_run_instrumented_app``. The instance is built with ``object.__new__`` to skip
the hardware setup in ``__init__``.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from accordo.exceptions import AccordoProcessError, AccordoTimeoutError
from accordo.validator import Accordo, _TimeoutException


def _validator():
    v = object.__new__(Accordo)
    v.kernel_args = [("out", "float*")]
    v.working_directory = "."
    v.kernel_name = "k"
    return v


def _capture(side_effect=None, arrays=None, timeout=5):
    v = _validator()
    run = MagicMock()
    if side_effect is not None:
        run.side_effect = side_effect
    else:
        run.return_value = arrays if arrays is not None else [[]]
    with patch.object(Accordo, "_run_instrumented_app", run):
        return v, run, v.capture_snapshot(binary=["./app"], timeout_seconds=timeout)


class TestErrorTranslation:
    def test_internal_timeout_becomes_accordo_timeout(self):
        v = _validator()
        with patch.object(Accordo, "_run_instrumented_app", side_effect=_TimeoutException("boom")):
            with pytest.raises(AccordoTimeoutError) as exc:
                v.capture_snapshot(binary=["./app"], timeout_seconds=7)
        assert "timed out after 7s" in str(exc.value)
        assert exc.value.timeout_seconds == 7

    def test_timeout_error_becomes_accordo_timeout(self):
        v = _validator()
        with patch.object(Accordo, "_run_instrumented_app", side_effect=TimeoutError("slow")):
            with pytest.raises(AccordoTimeoutError) as exc:
                v.capture_snapshot(binary=["./app"], timeout_seconds=3)
        assert "slow" in str(exc.value)
        assert exc.value.timeout_seconds == 3

    def test_runtime_error_becomes_process_error(self):
        v = _validator()
        with patch.object(Accordo, "_run_instrumented_app", side_effect=RuntimeError("segfault")):
            with pytest.raises(AccordoProcessError) as exc:
                v.capture_snapshot(binary=["./app"], timeout_seconds=5)
        assert "segfault" in str(exc.value)

    def test_timeout_mentions_gpu_crash_as_likely_cause(self):
        """The hint is the actionable part of the message for a hung dispatch."""
        v = _validator()
        with patch.object(Accordo, "_run_instrumented_app", side_effect=_TimeoutException("x")):
            with pytest.raises(AccordoTimeoutError, match="GPU crash or hung process"):
                v.capture_snapshot(binary=["./app"], timeout_seconds=1)


class TestSnapshotConstruction:
    def test_returns_snapshot_with_dispatch_arrays(self):
        _, _, snap = _capture(arrays=[["A"], ["B"]])
        assert snap.dispatch_arrays == [["A"], ["B"]]
        assert snap.arrays == ["A"]  # first dispatch for backward compatibility

    def test_empty_result_yields_empty_arrays(self):
        _, _, snap = _capture(arrays=[])
        assert snap.arrays == []

    def test_binary_and_working_directory_recorded(self):
        _, _, snap = _capture()
        assert snap.binary == ["./app"]
        assert snap.working_directory == "."

    def test_execution_time_is_populated(self):
        _, _, snap = _capture()
        assert isinstance(snap.execution_time_ms, float)
        assert snap.execution_time_ms >= 0.0

    def test_missing_metadata_file_is_not_fatal(self):
        """Grid/block come from a metadata file the C++ side may not have written."""
        _, _, snap = _capture()
        assert snap.grid_size is None
        assert snap.block_size is None

    def test_grid_and_block_read_from_metadata(self, tmp_path):
        v = _validator()
        meta = {"grid": {"x": 8, "y": 1, "z": 1}, "block": {"x": 64, "y": 1, "z": 1}}

        real_open = open

        def fake_open(path, *a, **kw):
            if "accordo_dispatch_" in str(path):
                import io

                return io.StringIO(json.dumps(meta))
            return real_open(path, *a, **kw)

        with (
            patch.object(Accordo, "_run_instrumented_app", return_value=[[]]),
            patch("builtins.open", side_effect=fake_open),
            patch("os.path.exists", return_value=True),
        ):
            snap = v.capture_snapshot(binary=["./app"], timeout_seconds=5)
        assert snap.grid_size == {"x": 8, "y": 1, "z": 1}
        assert snap.block_size == {"x": 64, "y": 1, "z": 1}

    def test_malformed_metadata_is_swallowed(self, tmp_path):
        """A corrupt metadata file must not fail an otherwise good capture."""
        v = _validator()
        real_open = open

        def fake_open(path, *a, **kw):
            if "accordo_dispatch_" in str(path):
                import io

                return io.StringIO("{not json")
            return real_open(path, *a, **kw)

        with (
            patch.object(Accordo, "_run_instrumented_app", return_value=[[]]),
            patch("builtins.open", side_effect=fake_open),
            patch("os.path.exists", return_value=True),
        ):
            snap = v.capture_snapshot(binary=["./app"], timeout_seconds=5)
        assert snap.grid_size is None


class TestDispatchId:
    def test_dispatch_id_forwarded_when_given(self):
        v = _validator()
        run = MagicMock(return_value=[[]])
        with patch.object(Accordo, "_run_instrumented_app", run):
            v.capture_snapshot(binary=["./app"], timeout_seconds=5, dispatch_id=3)
        env = run.call_args.kwargs.get("extra_env", {})
        assert env.get("ACCORDO_DISPATCH_ID") == "3"

    def test_dispatch_id_absent_by_default(self):
        v = _validator()
        run = MagicMock(return_value=[[]])
        with patch.object(Accordo, "_run_instrumented_app", run):
            v.capture_snapshot(binary=["./app"], timeout_seconds=5)
        env = run.call_args.kwargs.get("extra_env", {})
        assert "ACCORDO_DISPATCH_ID" not in env

    def test_metadata_file_path_always_provided(self):
        v = _validator()
        run = MagicMock(return_value=[[]])
        with patch.object(Accordo, "_run_instrumented_app", run):
            v.capture_snapshot(binary=["./app"], timeout_seconds=5)
        env = run.call_args.kwargs.get("extra_env", {})
        assert "ACCORDO_DISPATCH_METADATA_FILE" in env


class TestCliImportGuard:
    def test_import_failure_reported_as_json_error(self, capsys):
        """cli._run_validate must emit JSON, not a traceback, if the package is broken."""
        from accordo import cli

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

        def boom(name, *args, **kwargs):
            if name.endswith("validator") or name == "accordo.validator":
                raise ImportError("libaccordo.so not found")
            return real_import(name, *args, **kwargs)

        args = MagicMock()
        args.log_level = "WARNING"
        with patch("builtins.__import__", side_effect=boom):
            rc = cli._run_validate(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "Failed to import accordo" in out
