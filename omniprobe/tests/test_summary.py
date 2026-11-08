# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for summarize_memory."""

from omniprobe_ik import summarize_memory
from omniprobe_ik.api import KernelMemoryResult, MemoryAccess, MemoryAnalysisResult


def _access(line, space, kind, excess=0, conflicts=0, executions=64, column=5):
    return MemoryAccess(
        source_file="k.hip",
        line=line,
        column=column,
        excess_cache_lines=excess,
        bank_conflicts=conflicts,
        memory_space=space,
        access_type=kind,
        execution_count=executions,
    )


def _result():
    return MemoryAnalysisResult(
        kernels=[
            KernelMemoryResult(
                kernel_name="strided",
                dispatch_id=1,
                accesses=[
                    _access(10, "global", "read", excess=64 * 30),
                    _access(11, "global", "write", excess=0),
                    # one LDS location split in two entries, as Omniprobe sometimes emits
                    _access(20, "lds", "read", conflicts=63 * 62, executions=63),
                    _access(20, "lds", "read", conflicts=62, executions=1),
                ],
            ),
            KernelMemoryResult(
                kernel_name="mild",
                dispatch_id=2,
                accesses=[_access(30, "global", "read", excess=128 * 2, executions=128)],
            ),
        ]
    )


def test_global_sites_ranked_by_total_and_zero_sites_omitted():
    sites = summarize_memory(_result())["global_sites"]
    assert [(s["kernel"], s["line"]) for s in sites] == [("strided", 10), ("mild", 30)]
    assert sites[0]["excess_per_execution"] == 30.0
    assert sites[1]["excess_per_execution"] == 2.0
    assert "bank_conflicts" not in sites[0]


def test_split_entries_are_merged():
    sites = summarize_memory(_result())["lds_sites"]
    assert len(sites) == 1
    assert sites[0]["bank_conflicts"] == 64 * 62
    assert sites[0]["execution_count"] == 64
    assert sites[0]["conflicts_per_execution"] == 62.0


def test_kernel_totals_worst_first():
    kernels = summarize_memory(_result())["kernels"]
    assert kernels[0] == {
        "kernel": "strided",
        "excess_cache_lines": 64 * 30,
        "bank_conflicts": 64 * 62,
    }
    assert kernels[1]["kernel"] == "mild"


def test_top_limits_each_list():
    summary = summarize_memory(_result(), top=1)
    assert len(summary["global_sites"]) == 1
    assert summary["global_sites"][0]["kernel"] == "strided"


def test_empty_result_and_how_to_read():
    summary = summarize_memory(MemoryAnalysisResult())
    assert summary["global_sites"] == [] and summary["lds_sites"] == []
    assert "per wave execution" in summary["how_to_read"]


def test_uninstrumented_kernels_become_a_warning():
    result = _result()
    result.uninstrumented_kernels = ["other_kernel(float*) [clone .kd]"]
    summary = summarize_memory(result)
    assert summary["uninstrumented_kernels"] == ["other_kernel(float*) [clone .kd]"]
    assert len(summary["warnings"]) == 1
    assert "WITHOUT instrumentation" in summary["warnings"][0]
    assert "other_kernel" in summary["warnings"][0]


def test_no_warning_when_everything_was_instrumented():
    summary = summarize_memory(_result())
    assert summary["warnings"] == []
