# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the Nexus Python API.

These need neither a GPU nor a built ``libnexus.so``: the library lookup and
the traced subprocess are replaced at their call sites, which is what makes
the error paths reachable. Trace payloads are real dicts in the shape the
tracer emits, so a change to the JSON contract breaks these tests.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import nexus as nx
from nexus import Kernel, Nexus, Trace

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

KERNEL_PAYLOAD = {
    "assembly": ["v_add_f32 v0, v1, v2", "s_endpgm"],
    "hip": ["__global__ void vector_add(float* a) {}"],
    "files": ["/src/vector_add.hip"],
    "lines": [12, 13],
    "signature": "vector_add(float*)",
}

TRACE_PAYLOAD = {"kernels": {"vector_add": KERNEL_PAYLOAD, "reduce": {"assembly": ["s_endpgm"]}}}


def _completed(rc=0, stderr=""):
    return subprocess.CompletedProcess(args=["x"], returncode=rc, stdout="", stderr=stderr)


@pytest.fixture
def tracer(tmp_path):
    """A Nexus instance with the shared-library lookup satisfied."""
    lib = tmp_path / "libnexus.so"
    lib.write_bytes(b"")
    with patch.object(nx, "_find_nexus_lib", return_value=lib):
        yield Nexus(log_level=3)


# --------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------


def test_kernel_exposes_payload_fields():
    k = Kernel("vector_add", KERNEL_PAYLOAD)
    assert k.name == "vector_add"
    assert k.assembly == KERNEL_PAYLOAD["assembly"]
    assert k.hip == KERNEL_PAYLOAD["hip"]
    assert k.files == KERNEL_PAYLOAD["files"]
    assert k.lines == KERNEL_PAYLOAD["lines"]
    assert k.signature == "vector_add(float*)"


def test_kernel_fields_default_when_absent():
    # A kernel the tracer emitted with no source info must not raise.
    k = Kernel("bare", {})
    assert k.assembly == []
    assert k.hip == []
    assert k.files == []
    assert k.lines == []
    assert k.signature == ""


def test_kernel_repr_reports_instruction_count():
    assert "2 instructions" in repr(Kernel("vector_add", KERNEL_PAYLOAD))
    assert "vector_add" in repr(Kernel("vector_add", KERNEL_PAYLOAD))


# --------------------------------------------------------------------------
# Trace
# --------------------------------------------------------------------------


def test_trace_len_counts_kernels():
    assert len(Trace(TRACE_PAYLOAD)) == 2


def test_trace_empty_payload_is_empty():
    assert len(Trace({})) == 0
    assert Trace({}).kernels == []


def test_trace_iterates_kernel_objects():
    names = [k.name for k in Trace(TRACE_PAYLOAD)]
    assert sorted(names) == ["reduce", "vector_add"]
    assert all(isinstance(k, Kernel) for k in Trace(TRACE_PAYLOAD))


def test_trace_getitem_returns_named_kernel():
    k = Trace(TRACE_PAYLOAD)["vector_add"]
    assert k.signature == "vector_add(float*)"


def test_trace_getitem_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="not found in trace"):
        Trace(TRACE_PAYLOAD)["nope"]


def test_trace_contains():
    t = Trace(TRACE_PAYLOAD)
    assert "vector_add" in t
    assert "nope" not in t


def test_trace_kernels_and_names_agree():
    t = Trace(TRACE_PAYLOAD)
    assert sorted(t.kernel_names) == sorted(k.name for k in t.kernels)


def test_trace_repr_reports_count():
    assert repr(Trace(TRACE_PAYLOAD)) == "Trace(2 kernels)"


def test_trace_save_roundtrips_through_load(tmp_path):
    out = tmp_path / "t.json"
    Trace(TRACE_PAYLOAD).save(str(out))
    # save() must write the raw payload, so load() reconstructs an equal trace.
    assert json.loads(out.read_text()) == TRACE_PAYLOAD
    assert len(Nexus.load(str(out))) == 2


# --------------------------------------------------------------------------
# _find_nexus_lib
# --------------------------------------------------------------------------


def test_find_lib_returns_none_when_absent():
    with patch.object(Path, "exists", return_value=False):
        assert nx._find_nexus_lib() is None


def test_find_lib_returns_resolved_path_when_present():
    with patch.object(Path, "exists", return_value=True):
        found = nx._find_nexus_lib()
    assert found is not None
    assert found.name == "libnexus.so"
    assert found.is_absolute()


# --------------------------------------------------------------------------
# Nexus construction
# --------------------------------------------------------------------------


def test_missing_library_raises_with_build_instructions():
    with patch.object(nx, "_find_nexus_lib", return_value=None):
        with pytest.raises(RuntimeError, match="Could not find libnexus.so"):
            Nexus()


def test_constructor_records_settings(tmp_path):
    lib = tmp_path / "libnexus.so"
    lib.write_bytes(b"")
    with patch.object(nx, "_find_nexus_lib", return_value=lib):
        n = Nexus(log_level=4, extra_search_prefix="/a:/b")
    assert n.log_level == 4
    assert n.extra_search_prefix == "/a:/b"


# --------------------------------------------------------------------------
# Nexus.run — environment contract
# --------------------------------------------------------------------------


def test_run_sets_tracer_environment(tracer, tmp_path):
    out = tmp_path / "t.json"
    out.write_text(json.dumps(TRACE_PAYLOAD))
    with patch.object(subprocess, "run", return_value=_completed()) as run:
        tracer.run(["python", "app.py"], output=str(out))
    env = run.call_args.kwargs["env"]
    # HSA_TOOLS_LIB is how the tracer gets injected; without it nothing traces.
    assert env["HSA_TOOLS_LIB"] == str(tracer._lib_path)
    assert env["NEXUS_LOG_LEVEL"] == "3"
    assert env["NEXUS_OUTPUT_FILE"] == str(out)
    assert env["TRITON_DISABLE_LINE_INFO"] == "0"


def test_run_omits_search_prefix_when_unset(tracer, tmp_path):
    out = tmp_path / "t.json"
    out.write_text("{}")
    with patch.object(subprocess, "run", return_value=_completed()) as run:
        tracer.run(["true"], output=str(out))
    assert "NEXUS_EXTRA_SEARCH_PREFIX" not in run.call_args.kwargs["env"]


def test_run_passes_search_prefix_when_set(tmp_path):
    lib = tmp_path / "libnexus.so"
    lib.write_bytes(b"")
    out = tmp_path / "t.json"
    out.write_text("{}")
    with patch.object(nx, "_find_nexus_lib", return_value=lib):
        n = Nexus(extra_search_prefix="/src:/inc")
    with patch.object(subprocess, "run", return_value=_completed()) as run:
        n.run(["true"], output=str(out))
    assert run.call_args.kwargs["env"]["NEXUS_EXTRA_SEARCH_PREFIX"] == "/src:/inc"


def test_run_merges_caller_env_last(tracer, tmp_path):
    out = tmp_path / "t.json"
    out.write_text("{}")
    with patch.object(subprocess, "run", return_value=_completed()) as run:
        tracer.run(["true"], output=str(out), env={"MY_VAR": "1", "NEXUS_LOG_LEVEL": "9"})
    env = run.call_args.kwargs["env"]
    assert env["MY_VAR"] == "1"
    # Caller-supplied values override the defaults, not the other way round.
    assert env["NEXUS_LOG_LEVEL"] == "9"


def test_run_forwards_cwd(tracer, tmp_path):
    out = tmp_path / "t.json"
    out.write_text("{}")
    with patch.object(subprocess, "run", return_value=_completed()) as run:
        tracer.run(["true"], output=str(out), cwd="/tmp")
    assert run.call_args.kwargs["cwd"] == "/tmp"


def test_run_generates_output_path_when_omitted(tracer):
    with patch.object(subprocess, "run", return_value=_completed()) as run:
        tracer.run(["true"])
    generated = run.call_args.kwargs["env"]["NEXUS_OUTPUT_FILE"]
    assert generated.endswith(".json")
    assert "nexus_trace_" in generated


# --------------------------------------------------------------------------
# Nexus.run — result handling
# --------------------------------------------------------------------------


def test_run_parses_trace(tracer, tmp_path):
    out = tmp_path / "t.json"
    out.write_text(json.dumps(TRACE_PAYLOAD))
    with patch.object(subprocess, "run", return_value=_completed()):
        trace = tracer.run(["true"], output=str(out))
    assert len(trace) == 2
    assert trace["vector_add"].signature == "vector_add(float*)"


def test_run_nonzero_exit_raises_with_stderr(tracer, tmp_path):
    out = tmp_path / "t.json"
    with patch.object(subprocess, "run", return_value=_completed(rc=2, stderr="segfault")):
        with pytest.raises(RuntimeError, match="segfault"):
            tracer.run(["false"], output=str(out))


def test_run_empty_trace_file_yields_empty_trace(tracer, tmp_path):
    # No kernels executed: the tracer writes an empty file rather than JSON.
    out = tmp_path / "t.json"
    out.write_text("   \n")
    with patch.object(subprocess, "run", return_value=_completed()):
        assert len(tracer.run(["true"], output=str(out))) == 0


def test_run_missing_trace_file_yields_empty_trace(tracer, tmp_path):
    out = tmp_path / "never_written.json"
    with patch.object(subprocess, "run", return_value=_completed()):
        assert len(tracer.run(["true"], output=str(out))) == 0


def test_run_malformed_trace_file_raises(tracer, tmp_path):
    out = tmp_path / "t.json"
    out.write_text("{not json")
    with patch.object(subprocess, "run", return_value=_completed()):
        with pytest.raises(RuntimeError, match="Failed to parse trace file"):
            tracer.run(["true"], output=str(out))


# --------------------------------------------------------------------------
# Nexus.load
# --------------------------------------------------------------------------


def test_load_reads_saved_trace(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(TRACE_PAYLOAD))
    trace = Nexus.load(str(p))
    assert sorted(trace.kernel_names) == ["reduce", "vector_add"]


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        Nexus.load("/nonexistent/trace.json")


def test_version_is_exposed():
    assert isinstance(nx.__version__, str)
    assert nx.__version__
