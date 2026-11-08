# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end basic-block analysis against a real Omniprobe installation.

Basic-block analysis reports exact execution counts per block -- how many times
a wave entered it (``wave_visits``) and how many lanes were active over those
entries (``active_lanes``) -- with the block's source line. It does not time
blocks. Every expectation below is computed by hand from the test kernels.
"""

from .conftest import (
    BRANCHING_KERNEL,
    STRIDED_KERNEL,
    skip_no_gpu,
    skip_no_hipcc,
    skip_no_omniprobe,
    skip_no_upstream_api,
)

pytestmark = [skip_no_omniprobe, skip_no_upstream_api, skip_no_gpu, skip_no_hipcc]

# BRANCHING_KERNEL: 256 * 1024 threads in blocks of 256 = 4096 waves of 64 lanes.
BRANCH_THREADS = 256 * 1024
BRANCH_WAVES = BRANCH_THREADS // 64


def _line(source, text, after=None):
    """1-based line of *text* in a kernel source string (as written to the .hip file),
    optionally the first one after the line containing *after*."""
    seen_after = after is None
    for number, line in enumerate(source.splitlines(), start=1):
        if not seen_after:
            seen_after = after in line
            continue
        if text in line:
            return number
    raise AssertionError(text)


def _blocks_at(kernel, line):
    return [b for b in kernel.basic_blocks if b.line == line]


def test_basic_block_analysis_reports_blocks_with_source_lines(probe, branching_bb_binary):
    result = probe.analyze_basic_blocks(command=str(branching_bb_binary))

    assert len(result.kernels) == 1
    blocks = result.kernels[0].basic_blocks
    assert blocks, "kernel reported no basic blocks"
    assert all(b.source_file.endswith(".hip") and b.line > 0 for b in blocks)


def test_every_lane_passes_through_entry_and_exit(probe, branching_bb_binary):
    kernel = probe.analyze_basic_blocks(command=str(branching_bb_binary)).kernels[0]

    for text in ("int idx = blockIdx.x", "out[idx] = acc;"):
        blocks = _blocks_at(kernel, _line(BRANCHING_KERNEL, text))
        assert blocks, text
        assert all(b.wave_visits == BRANCH_WAVES for b in blocks), text
        assert all(b.active_lanes == BRANCH_THREADS for b in blocks), text


def test_branch_sides_split_lanes_exactly(probe, branching_bb_binary):
    """Every wave has even and odd lanes, so it enters both sides with half its lanes."""
    kernel = probe.analyze_basic_blocks(command=str(branching_bb_binary)).kernels[0]

    for text in ("acc = acc * 2.0f + 1.0f;", "acc = __sinf(acc) + __cosf(acc);"):
        blocks = _blocks_at(kernel, _line(BRANCHING_KERNEL, text))
        assert len(blocks) == 1, (text, blocks)
        assert blocks[0].wave_visits == BRANCH_WAVES
        assert blocks[0].active_lanes == BRANCH_THREADS // 2


def test_multi_kernel_program(probe, strided_bb_binary):
    """Both kernels are reported; the strided guard admits only 1 thread in 16."""
    result = probe.analyze_basic_blocks(command=str(strided_bb_binary))
    kernels = {k.kernel_name.split("(")[0]: k for k in result.kernels}
    assert set(kernels) == {"coalesced_copy", "strided_copy"}

    threads = 1024 * 1024
    for block in kernels["coalesced_copy"].basic_blocks:
        assert (block.wave_visits, block.active_lanes) == (threads // 64, threads)

    # strided_copy: idx = gid * 16 < N holds for gid < N / 16 = 65536 threads = 1024 waves.
    guarded = _blocks_at(
        kernels["strided_copy"],
        _line(STRIDED_KERNEL, "out[idx] = in[idx] * 2.0f;", after="strided_copy"),
    )
    assert [(b.wave_visits, b.active_lanes) for b in guarded] == [(1024, 65536)]
