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


def test_uninstrumented_binary_raises(probe, uninstrumented_binary):
    """A binary built without the plugin is an explanatory error, not an empty result.

    This is the mistake users will make most often, and the one an empty result
    hides best: Omniprobe exits 0 and reports nothing.
    """
    from omniprobe_ik import NoInstrumentedKernelsError

    with pytest.raises(NoInstrumentedKernelsError, match="-fpass-plugin"):
        probe.analyze_memory(command=str(uninstrumented_binary))


def test_kernel_filter_matching_nothing_is_its_own_error(probe, tmp_path_factory):
    """'^gstride' misses template kernels ('void gstride<1>(...)'): Omniprobe b71927b+ says the
    filter matched nothing; the error must say so, list the kernels, and not ask for a rebuild."""
    import asyncio

    from fastmcp import Client

    from omniprobe_ik import KernelFilterMatchedNothingError
    from omniprobe_ik.mcp.server import mcp

    from .conftest import _compile
    from .test_calibration import CALIBRATION_SOURCE

    binary = _compile(tmp_path_factory, CALIBRATION_SOURCE, "filter", "memory")
    with pytest.raises(KernelFilterMatchedNothingError) as err:
        probe.analyze_memory(command=str(binary), kernel_filter="^gstride")
    text = str(err.value)
    assert "'^gstride'" in text and "do not rebuild" in text and "searched" in text
    assert any(n.startswith("void gstride<16>(") for n in err.value.dispatched)
    assert "via_table" in text

    # The template kernel is matched by a search without the anchor.
    assert probe.analyze_memory(command=str(binary), kernel_filter="gstride<16>").kernels

    async def call():
        async with Client(mcp) as client:
            return await client.call_tool(
                "analyze_memory",
                {"command": str(binary), "kernel_filter": "^gstride"},
                raise_on_error=False,
            )

    result = asyncio.run(call())
    assert result.is_error
    assert "matched none of the instrumented kernels" in result.content[0].text


def test_random_dispatch_with_filter_explains_an_empty_capture(probe, strided_binary):
    """Omniprobe 22c4d95 draws the random dispatch from all dispatches, including ones the
    filter excludes, so most runs capture nothing. Each run must either return the filtered
    kernel or say that random-with-filter is why (not ask for a rebuild)."""
    from omniprobe_ik import NoInstrumentedKernelsError

    empty = 0
    for _ in range(6):
        try:
            result = probe.analyze_memory(
                command=str(strided_binary), kernel_filter="strided", dispatches="random"
            )
            assert all("strided" in k.kernel_name for k in result.kernels)
        except NoInstrumentedKernelsError as err:
            empty += 1
            assert "dispatches='random'" in str(err) and "do not rebuild" in str(err), str(err)
    assert empty >= 1, "expected at least one empty capture in 6 runs (measured: 4 of 4 here)"


def test_omniprobe_messages_are_passed_through(probe, strided_binary):
    """Omniprobe's own warning reaches the caller verbatim: in the error when nothing was
    captured, and as an empty list on a clean run."""
    from omniprobe_ik import NoInstrumentedKernelsError, summarize_memory

    with pytest.raises(NoInstrumentedKernelsError) as err:
        probe.analyze_memory(command=str(strided_binary), kernel_filter="nomatch_xyz")
    assert "omniprobe: WARNING: no dispatched kernel matched the kernel filter" in str(err.value)

    result = probe.analyze_memory(command=str(strided_binary))
    assert result.omniprobe_messages == []
    assert summarize_memory(result)["omniprobe_messages"] == []
