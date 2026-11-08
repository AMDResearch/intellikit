# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end basic-block timing against a real Omniprobe installation.

Companion to ``test_memory_analysis.py``, covering the second of the two
analyses the package exposes. The main tests use a single-kernel program, and
``test_multi_kernel_program`` covers a program with two.
"""

from .conftest import (
    skip_no_gpu,
    skip_no_hipcc,
    skip_no_omniprobe,
    skip_no_upstream_api,
)

pytestmark = [skip_no_omniprobe, skip_no_upstream_api, skip_no_gpu, skip_no_hipcc]


def test_basic_block_analysis_reports_blocks(probe, branching_bb_binary):
    """A real run reports at least one kernel carrying at least one basic block."""
    result = probe.analyze_basic_blocks(command=str(branching_bb_binary))

    assert result.kernels, "analysis returned no kernels for a binary that launched one"
    assert any(kernel.basic_blocks for kernel in result.kernels), (
        "kernels were reported but none carried any basic blocks"
    )


def test_cycle_percentiles_are_ordered(probe, branching_bb_binary):
    """Percentiles must be monotonic and bounded by min/max.

    A distribution whose p50 sits outside [min, max], or whose p99 falls below
    its p50, is not a distribution -- it is a conversion bug. This is the
    cheapest assertion that would catch one, and it needs no knowledge of what
    the true cycle counts should be.
    """
    result = probe.analyze_basic_blocks(command=str(branching_bb_binary))

    checked = 0
    for kernel in result.kernels:
        for bb in kernel.basic_blocks:
            assert bb.min_cycles <= bb.max_cycles, (
                f"block {bb.bb_id}: min_cycles {bb.min_cycles} > max_cycles {bb.max_cycles}"
            )
            percentiles = [bb.p25, bb.p50, bb.p75, bb.p95, bb.p99]
            assert percentiles == sorted(percentiles), (
                f"block {bb.bb_id}: percentiles not monotonic: {percentiles}"
            )
            assert bb.min_cycles <= bb.p50 <= bb.max_cycles, (
                f"block {bb.bb_id}: p50 {bb.p50} outside [{bb.min_cycles}, {bb.max_cycles}]"
            )
            checked += 1

    assert checked > 0, "no basic blocks were available to check"


def test_wave_counts_are_positive(probe, branching_bb_binary):
    """Any block that was timed must have been executed by at least one wave."""
    result = probe.analyze_basic_blocks(command=str(branching_bb_binary))

    timed = [bb for kernel in result.kernels for bb in kernel.basic_blocks if bb.max_cycles > 0]

    assert timed, "no basic block reported a nonzero cycle count"
    assert all(bb.wave_count > 0 for bb in timed), (
        "a block reported timing data but a wave_count of zero"
    )


def test_multi_kernel_program(probe, strided_bb_binary):
    """Both kernels of a two-kernel program are reported."""
    result = probe.analyze_basic_blocks(command=str(strided_bb_binary))

    names = {kernel.kernel_name for kernel in result.kernels}
    assert any("coalesced" in name for name in names), sorted(names)
    assert any("strided" in name for name in names), sorted(names)
