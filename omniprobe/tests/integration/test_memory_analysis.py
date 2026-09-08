# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end memory coalescing analysis against a real Omniprobe installation.

The unit tests in ``omniprobe/tests/test_api.py`` mock the upstream subprocess,
so they establish that results are converted correctly and nothing about whether
Omniprobe produces them. These tests close that gap on a machine that has the
tool: they compile a kernel, analyze it, and assert on what comes back.
"""

import pytest

from .conftest import (
    skip_no_gpu,
    skip_no_hipcc,
    skip_no_omniprobe,
    skip_no_upstream_api,
)

pytestmark = [skip_no_omniprobe, skip_no_upstream_api, skip_no_gpu, skip_no_hipcc]


def test_memory_analysis_reports_kernels(probe, strided_binary):
    """A real run reports at least one kernel with at least one access."""
    result = probe.analyze_memory(command=str(strided_binary))

    assert result.kernels, "analysis returned no kernels for a binary that launched two"

    names = {kernel.kernel_name for kernel in result.kernels}
    assert any("copy" in name for name in names), (
        f"expected coalesced_copy/strided_copy in the results, got {sorted(names)}"
    )

    assert any(kernel.accesses for kernel in result.kernels), (
        "kernels were reported but none carried any memory accesses"
    )


def test_memory_analysis_attributes_to_source(probe, strided_binary):
    """With -g, findings carry a real source location rather than empty fields."""
    result = probe.analyze_memory(command=str(strided_binary))

    located = [
        access
        for kernel in result.kernels
        for access in kernel.accesses
        if access.source_file and access.line > 0
    ]

    assert located, (
        "no access carried a source location; the binary was built with -g, so "
        "either debug info is not reaching Omniprobe or attribution is broken"
    )
    assert any(access.source_file.endswith(".hip") for access in located)


def test_strided_kernel_wastes_more_bandwidth(probe, strided_binary):
    """The strided kernel should fetch more excess cache lines than the coalesced one.

    This is the assertion that makes the test worth running: it checks that the
    numbers mean what the documentation says they mean, rather than merely that
    fields are populated. Both kernels do identical arithmetic over the same
    buffers and differ only in stride.
    """
    result = probe.analyze_memory(command=str(strided_binary))

    excess = {}
    for kernel in result.kernels:
        total = sum(access.excess_cache_lines for access in kernel.accesses)
        excess[kernel.kernel_name] = excess.get(kernel.kernel_name, 0) + total

    coalesced = next((v for k, v in excess.items() if "coalesced" in k), None)
    strided = next((v for k, v in excess.items() if "strided" in k), None)

    if coalesced is None or strided is None:
        pytest.skip(f"expected both kernels in the results, got {sorted(excess)}")

    assert strided > coalesced, (
        f"strided_copy ({strided}) should waste more cache lines than coalesced_copy ({coalesced})"
    )


def test_kernel_filter_narrows_results(probe, strided_binary):
    """The kernel_filter regex actually restricts which kernels are analyzed."""
    result = probe.analyze_memory(command=str(strided_binary), kernel_filter="strided.*")

    assert result.kernels, "filtered analysis returned nothing for a matching pattern"
    assert all("strided" in kernel.kernel_name for kernel in result.kernels), (
        f"filter leaked non-matching kernels: {sorted(k.kernel_name for k in result.kernels)}"
    )
