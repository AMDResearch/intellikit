# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for run_analyzer's result object (no GPU)."""

from omniprobe_ik import AnalyzerRun


def _run(output="", stdout="", stderr=""):
    return AnalyzerRun("Heatmap", "./app", 0, stdout, stderr, output)


def test_json_lines_and_arrays_parse_the_same():
    lines = '{"a": 1}\n{"a": 2}\n'
    array = '[ {"a": 1},\n {"a": 2} ]'
    assert (
        list(_run(lines).json_records()) == list(_run(array).json_records()) == [{"a": 1}, {"a": 2}]
    )


def test_empty_output_has_no_records():
    assert list(_run("").json_records()) == []


def test_log_includes_both_streams():
    run = _run(stdout="ACCESS_COUNTER {}", stderr="omniprobe: warning")
    assert "ACCESS_COUNTER" in run.log and "warning" in run.log


def _fake_omniprobe(monkeypatch, tmp_path):
    """Replace subprocess.run in raw.py; return the list of (args, env) it was called with."""
    import subprocess

    from omniprobe_ik import raw

    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs.get("env")))
        return subprocess.CompletedProcess(args, 0, "ADDRESS_COUNTER {}\n", "omniprobe log\n")

    monkeypatch.setattr(raw.subprocess, "run", fake_run)
    exe = tmp_path / "omniprobe"
    exe.write_text("")
    return calls, str(exe)


def test_py_path_runs_python_handler(monkeypatch, tmp_path):
    from omniprobe_ik import run_analyzer

    calls, exe = _fake_omniprobe(monkeypatch, tmp_path)
    handler = tmp_path / "h.py"
    handler.write_text("")
    run = run_analyzer(str(handler), "./app", omniprobe_path=exe)
    args, env = calls[0]
    assert args[args.index("-a") + 1] == "PythonHandler"
    assert env["OMNIPROBE_PYTHON_HANDLER"] == str(handler.resolve())
    assert run.stdout.startswith("ADDRESS_COUNTER") and "ADDRESS_COUNTER" not in run.stderr


def test_missing_python_handler_is_an_error(monkeypatch, tmp_path):
    import pytest

    from omniprobe_ik import run_analyzer

    calls, exe = _fake_omniprobe(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        run_analyzer("Heatmap", "./app", omniprobe_path=exe, python_handler=str(tmp_path / "x.py"))
    assert calls == []


def test_named_analyzer_keeps_the_environment(monkeypatch, tmp_path):
    from omniprobe_ik import run_analyzer

    calls, exe = _fake_omniprobe(monkeypatch, tmp_path)
    run_analyzer("Heatmap", "./app", omniprobe_path=exe)
    assert calls[0][1] is None


def test_unusable_python_handler_is_an_error(monkeypatch, tmp_path):
    import subprocess

    import pytest

    from omniprobe_ik import raw, run_analyzer

    def fake_run(args, **kwargs):
        err = "PythonHandler: module 'h' has no callable handleTimeMessage; no Python handler installed\n"
        return subprocess.CompletedProcess(args, 0, "heatmap report\n", err)

    monkeypatch.setattr(raw.subprocess, "run", fake_run)
    exe = tmp_path / "omniprobe"
    exe.write_text("")
    handler = tmp_path / "h.py"
    handler.write_text("")
    with pytest.raises(RuntimeError, match="handleTimeMessage"):
        run_analyzer(str(handler), "./app", omniprobe_path=str(exe))
