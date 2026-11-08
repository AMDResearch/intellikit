# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Raw analyzers, typed traces, a bespoke handler, and a CMake build, on real hardware.

Every expectation is computed by hand from the test kernel: 1000 threads in
blocks of 256 run as 16 waves, the last with 40 active lanes; each active
thread does one global load and one global store, and every wave also loads its
kernel arguments (constant address space, all 64 lanes).
"""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

from .conftest import (
    skip_no_gpu,
    skip_no_hipcc,
    skip_no_omniprobe,
    skip_no_upstream_api,
)

pytestmark = [skip_no_omniprobe, skip_no_upstream_api, skip_no_gpu, skip_no_hipcc]

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _example_module():
    path = EXAMPLES / "04_custom_handler" / "analyze_custom.py"
    spec = importlib.util.spec_from_file_location("analyze_custom", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The example's handler and count kernel (memory build), plus a basic-block build."""
    from omniprobe_ik import instrumentation_flags

    module = _example_module()
    tmp = tmp_path_factory.mktemp("bespoke")
    handler, app = module.build(tmp)
    bb_app = tmp / "copy_n_bb"
    subprocess.run(
        [
            "hipcc",
            "-O2",
            "-g",
            *instrumentation_flags("basic_blocks"),
            str(tmp / "copy_n.hip"),
            "-o",
            str(bb_app),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return module, handler, app, bb_app


def test_bespoke_handler_counts_match_hand_computed_values(built):
    from omniprobe_ik import run_analyzer

    module, handler, app, _ = built
    run = run_analyzer(str(handler), str(app))
    rows = {(r["kernel"].split("(")[0], r["space"]): r for r in module.counts_from(run.stdout)}
    assert "ACCESS_COUNTER" not in run.stderr

    glob = rows[("copy_n", "global")]
    assert {k: glob[k] for k in module.EXPECTED_GLOBAL} == module.EXPECTED_GLOBAL
    kernarg = rows[("copy_n", "constant")]
    assert (kernarg["load_executions"], kernarg["lanes"]) == (16, 16 * 64)


def test_python_handler_counts_match_hand_computed_values(built):
    """A .py handler runs through PythonHandler; its report is on stdout, not stderr."""
    from omniprobe_ik import run_analyzer

    _, _, app, _ = built
    path = EXAMPLES / "06_python_handler" / "analyze_python_handler.py"
    spec = importlib.util.spec_from_file_location("analyze_python_handler", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    run = run_analyzer(str(EXAMPLES / "06_python_handler" / "address_counter.py"), str(app))
    rows = example.counts_from(run.stdout)
    assert len(rows) == 1, run.stdout[-2000:] + run.stderr[-2000:]
    row = rows[0]
    assert row["kernel"].startswith("copy_n")
    # 2000 global lane accesses + 16 waves x 64 lanes loading a kernel argument
    assert row["lane_accesses"] == 2000 + 16 * 64
    assert row["distinct_addresses"] >= 2000
    assert "ADDRESS_COUNTER" not in run.stderr


def test_handler_build_flags_point_at_existing_paths():
    from omniprobe_ik import handler_build_flags

    flags = handler_build_flags()
    for flag in flags["cflags"]:
        assert Path(flag[2:]).is_dir(), flag
    lib = next(f[2:] for f in flags["ldflags"] if f.startswith("-L"))
    assert (Path(lib) / "libdh_comms.so").is_file()


def test_heatmap(built):
    from omniprobe_ik import analyze_heatmap

    _, _, app, _ = built
    result = analyze_heatmap(str(app))
    assert len(result.kernels) == 1
    kernel = result.kernels[0]
    assert kernel.page_size == 1 << 20
    assert sorted(p.accesses for p in kernel.pages) == [1024, 2000]


def test_address_trace_and_truncation(built):
    from omniprobe_ik import log_addresses

    _, _, app, _ = built
    trace = log_addresses(str(app))
    assert not trace.truncated
    assert len(trace.records) == 48  # 16 waves x (load, store, kernel-argument load)
    assert sum(r.active_lane_count for r in trace.records) == 2000 + 16 * 64
    assert all(len(r.addresses) == r.active_lane_count for r in trace.records)

    short = log_addresses(str(app), max_records=5)
    assert short.truncated and len(short.records) == 5


def test_basic_block_trace(built):
    from omniprobe_ik import log_basic_blocks

    _, _, _, bb_app = built
    trace = log_basic_blocks(str(bb_app))
    assert len(trace.records) == 16 * 3  # every wave enters each of the kernel's 3 blocks
    assert sorted({r.basic_block_id for r in trace.records}) == [0, 1, 2]


@pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake not on PATH")
def test_cmake_example_builds_instrumented_and_finds_the_strided_kernel(tmp_path, probe):
    from omniprobe_ik import instrumentation_flags, summarize_memory

    flags = instrumentation_flags("memory")
    arch = next(f.split("=", 1)[1] for f in flags if f.startswith("--offload-arch="))
    hipcc = Path(shutil.which("hipcc")).resolve()
    clang = hipcc.parent.parent / "llvm" / "bin" / "clang++"
    configure = [
        "cmake",
        "-S",
        str(EXAMPLES / "05_cmake_app"),
        "-B",
        str(tmp_path),
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        f"-DCMAKE_HIP_ARCHITECTURES={arch}",
        f"-DCMAKE_HIP_FLAGS={' '.join(flags)}",
    ]
    if clang.is_file():
        configure += [f"-DCMAKE_HIP_COMPILER={clang}", f"-DCMAKE_PREFIX_PATH={hipcc.parent.parent}"]
    for step in (configure, ["cmake", "--build", str(tmp_path)]):
        done = subprocess.run(step, capture_output=True, text=True)
        assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]

    summary = summarize_memory(probe.analyze_memory(command=str(tmp_path / "app")))
    assert summary["warnings"] == []
    kernels = {s["kernel"].split("(")[0] for s in summary["global_sites"]}
    assert kernels == {"scale_columns"}
    assert all(s["source_file"].endswith("kernels.hip") for s in summary["global_sites"])
