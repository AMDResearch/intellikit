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


def test_log_includes_stderr_where_handlers_print():
    run = _run(stdout="banner", stderr="ACCESS_COUNTER {}")
    assert "ACCESS_COUNTER" in run.log and "banner" in run.log
