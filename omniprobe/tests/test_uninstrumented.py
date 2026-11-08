# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Detecting kernels that were dispatched without instrumentation."""

from unittest.mock import MagicMock

import pytest

from omniprobe_ik import NoInstrumentedKernelsError, Omniprobe, parse_uninstrumented_kernels

LOG = """\
Found instrumented alternative for coalesced_copy(float const*, float*, int) [clone .kd]
No instrumented alternative found for __amd_rocclr_copyBuffer.kd
No instrumented alternative found for other_kernel(float*) [clone .kd]
No instrumented alternative found for other_kernel(float*) [clone .kd]
No instrumented alternative found for _ZL6helperPf.intern.e6bbddf.kd
"""


def test_parse_skips_runtime_kernels_and_duplicates():
    assert parse_uninstrumented_kernels(LOG) == [
        "other_kernel(float*) [clone .kd]",
        "_ZL6helperPf.intern.e6bbddf.kd",
    ]


def test_parse_empty_log():
    assert parse_uninstrumented_kernels("") == []


def _probe_with_log(monkeypatch, upstream_result, log):
    """A wrapper whose upstream call 'runs' omniprobe and produces *log*."""
    import omniprobe_ik.api as api

    probe = Omniprobe.__new__(Omniprobe)
    probe._upstream = MagicMock()
    probe._upstream.analyze_memory.return_value = upstream_result
    monkeypatch.setattr(api, "_call_capturing_log", lambda func, **kwargs: (func(**kwargs), log))
    return probe


def test_partial_instrumentation_is_reported(monkeypatch):
    access = MagicMock(
        source_file="a.hip",
        line=3,
        column=1,
        excess_cache_lines=0,
        bank_conflicts=0,
        memory_space="global",
        access_type="read",
        execution_count=1,
    )
    kernel = MagicMock(kernel_name="coalesced_copy", dispatch_id=1, accesses=[access])
    probe = _probe_with_log(monkeypatch, MagicMock(kernels=[kernel]), LOG)
    result = probe.analyze_memory("./app")
    assert result.uninstrumented_kernels == [
        "other_kernel(float*) [clone .kd]",
        "_ZL6helperPf.intern.e6bbddf.kd",
    ]


def test_unknown_log_is_none_not_empty(monkeypatch):
    kernel = MagicMock(kernel_name="k", dispatch_id=1, accesses=[])
    probe = _probe_with_log(monkeypatch, MagicMock(kernels=[kernel]), None)
    assert probe.analyze_memory("./app").uninstrumented_kernels is None


def test_zero_kernels_error_names_the_uninstrumented_ones(monkeypatch):
    monkeypatch.setenv("OMNIPROBE_PLUGIN_DIR", "/nonexistent")
    probe = _probe_with_log(monkeypatch, MagicMock(kernels=[]), LOG)
    with pytest.raises(NoInstrumentedKernelsError, match="other_kernel"):
        probe.analyze_memory("./app")
