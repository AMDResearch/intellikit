"""Unit tests for kerncap.validator's format routing and build/run pipelines.

``validate_reproducer`` dispatches on which artifacts are present on disk, then
each ``_validate_*`` pipeline shells out to build and run the reproducer. The
existing ``test_validator.py`` covers the replay-comparison maths; this file
covers the routing and the three build/run pipelines around it.

``subprocess.run`` and ``_get_replay_path`` are stubbed at their call sites, so
none of this needs ROCm, a GPU, or a compiler. Reference and output buffers are
real files written with numpy, so the dtype inference and shape comparison run
for real rather than being mocked away.
"""

import json
import os
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from kerncap.validator import (
    _compare_outputs,
    _infer_numpy_dtype,
    _infer_numpy_dtype_from_torch,
    _validate_hip,
    _validate_hsaco,
    _validate_triton,
    validate_reproducer,
)


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


def sequence(*results):
    """Return a ``subprocess.run`` stub yielding *results* in order."""
    calls = []

    def _run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return results[min(len(calls) - 1, len(results) - 1)]

    _run.calls = calls
    return _run


@pytest.fixture
def repro(tmp_path):
    """An empty reproducer directory with a capture/ subdir."""
    (tmp_path / "capture").mkdir()
    return tmp_path


def write_buffers(repro_dir, idx, ref, out, ref_name="arg_0.bin"):
    """Write a reference buffer and a reproducer output buffer to disk."""
    (repro_dir / "capture").mkdir(exist_ok=True)
    (repro_dir / "reference_output").mkdir(exist_ok=True)
    ref.tofile(repro_dir / "capture" / ref_name)
    out.tofile(repro_dir / "reference_output" / f"output_{idx}.bin")


def out_arg(idx=0, **overrides):
    """A metadata entry for an output (non-const pointer) argument."""
    arg = {
        "index": idx,
        "is_pointer": True,
        "is_const": False,
        "file": "arg_0.bin",
        "type": "float*",
    }
    arg.update(overrides)
    return arg


# --------------------------------------------------------------------------
# validate_reproducer — format routing
# --------------------------------------------------------------------------


class TestFormatRouting:
    def test_dispatch_json_routes_to_the_replay_path(self, repro):
        """The VA-faithful format wins whenever dispatch.json is present."""
        (repro / "capture" / "dispatch.json").write_text("{}")

        with patch("kerncap.validator._validate_replay") as replay:
            validate_reproducer(str(repro), tolerance=1e-4, rtol=1e-3, hsaco="/x.hsaco")

        replay.assert_called_once_with(str(repro), 1e-4, 1e-3, hsaco="/x.hsaco")

    @pytest.mark.parametrize(
        ("artifact", "target"),
        [
            ("harness.hip", "_validate_hsaco"),
            ("reproducer.hip", "_validate_hip"),
            ("reproducer.py", "_validate_triton"),
        ],
    )
    def test_legacy_metadata_routes_by_artifact(self, repro, artifact, target):
        (repro / "capture" / "metadata.json").write_text('{"args": []}')
        (repro / artifact).write_text("")

        with patch(f"kerncap.validator.{target}") as fn:
            validate_reproducer(str(repro))

        fn.assert_called_once()

    def test_harness_wins_over_the_other_two(self, repro):
        """All three can coexist; the branch order decides."""
        (repro / "capture" / "metadata.json").write_text('{"args": []}')
        for name in ("harness.hip", "reproducer.hip", "reproducer.py"):
            (repro / name).write_text("")

        with (
            patch("kerncap.validator._validate_hsaco") as hsaco,
            patch("kerncap.validator._validate_hip") as hip,
            patch("kerncap.validator._validate_triton") as triton,
        ):
            validate_reproducer(str(repro))

        hsaco.assert_called_once()
        hip.assert_not_called()
        triton.assert_not_called()

    def test_metadata_is_parsed_and_forwarded(self, repro):
        (repro / "capture" / "metadata.json").write_text('{"args": [{"index": 3}]}')
        (repro / "reproducer.hip").write_text("")

        with patch("kerncap.validator._validate_hip") as hip:
            validate_reproducer(str(repro))

        assert hip.call_args.args[1] == {"args": [{"index": 3}]}

    def test_bare_triton_reproducer_needs_no_capture_dir(self, repro):
        """A Triton reproducer can be validated with no metadata at all."""
        (repro / "reproducer.py").write_text("")

        with patch("kerncap.validator._validate_triton") as triton:
            validate_reproducer(str(repro))

        assert triton.call_args.args[1] == {}

    def test_bare_triton_reproducer_still_reads_metadata_when_present(self, tmp_path):
        """No dispatch.json, but capture/metadata.json exists and is used."""
        (tmp_path / "capture").mkdir()
        (tmp_path / "reproducer.py").write_text("")
        # Written *after* the legacy branch would have matched, so this
        # exercises the second metadata read rather than the first.
        (tmp_path / "capture" / "metadata.json").write_text('{"args": [{"index": 7}]}')

        with patch("kerncap.validator._validate_triton") as triton:
            validate_reproducer(str(tmp_path))

        triton.assert_called_once()

    def test_nothing_recognisable_fails_clearly(self, repro):
        result = validate_reproducer(str(repro))

        assert result.passed is False
        assert "No dispatch.json, metadata.json, or reproducer.py found" in result.details[0]


# --------------------------------------------------------------------------
# _validate_hsaco
# --------------------------------------------------------------------------


class TestValidateHsaco:
    def test_build_failure_reports_stderr(self, repro):
        with patch("subprocess.run", sequence(completed(2, stderr="undefined reference"))):
            result = _validate_hsaco(str(repro), {"args": []}, 1e-6, 1e-5)

        assert result.passed is False
        assert result.mode == "numeric"
        assert "Build failed" in result.details[0]
        assert "undefined reference" in result.details[0]

    def test_builds_with_make_in_the_reproducer_dir(self, repro):
        (repro / "kernel.hsaco").write_bytes(b"\x7fELF")
        stub = sequence(completed(), completed())
        with patch("subprocess.run", stub):
            _validate_hsaco(str(repro), {"args": []}, 1e-6, 1e-5)

        build_argv = stub.calls[0][0]
        assert build_argv == ["make", "-C", str(repro), "all"]

    def test_missing_hsaco_after_a_successful_build(self, repro):
        """Build can succeed while producing nothing runnable."""
        with patch("subprocess.run", sequence(completed())):
            result = _validate_hsaco(str(repro), {"args": []}, 1e-6, 1e-5)

        assert result.passed is False
        assert result.details[0] == "Build: OK"
        assert "kernel.hsaco not found" in result.details[1]

    def test_run_failure_keeps_the_build_detail(self, repro):
        (repro / "kernel.hsaco").write_bytes(b"\x7fELF")
        with patch("subprocess.run", sequence(completed(), completed(134, stderr="SIGABRT"))):
            result = _validate_hsaco(str(repro), {"args": []}, 1e-6, 1e-5)

        assert result.passed is False
        assert result.details[0] == "Build: OK"
        assert "Run failed (exit 134)" in result.details[1]
        assert "SIGABRT" in result.details[1]

    def test_runs_the_harness_against_the_hsaco(self, repro):
        (repro / "kernel.hsaco").write_bytes(b"\x7fELF")
        stub = sequence(completed(), completed(stdout="ok"))
        with patch("subprocess.run", stub):
            _validate_hsaco(str(repro), {"args": []}, 1e-6, 1e-5)

        run_argv, run_kwargs = stub.calls[1]
        assert run_argv == ["./harness", "kernel.hsaco"]
        assert run_kwargs["cwd"] == str(repro)

    def test_success_flows_into_output_comparison(self, repro):
        (repro / "kernel.hsaco").write_bytes(b"\x7fELF")
        with (
            patch("subprocess.run", sequence(completed(), completed(stdout="ran 1 iter"))),
            patch("kerncap.validator._compare_outputs") as cmp,
        ):
            _validate_hsaco(str(repro), {"args": []}, 1e-6, 1e-5)

        details = cmp.call_args.args[4]
        assert details[0] == "Build: OK"
        assert "ran 1 iter" in details[1]


# --------------------------------------------------------------------------
# _validate_hip
# --------------------------------------------------------------------------


class TestValidateHip:
    def test_build_failure(self, repro):
        with patch("subprocess.run", sequence(completed(1, stderr="hipcc: error"))):
            result = _validate_hip(str(repro), {"args": []}, 1e-6, 1e-5)

        assert result.passed is False
        assert "hipcc: error" in result.details[0]

    def test_runs_the_reproducer_binary(self, repro):
        stub = sequence(completed(), completed(stdout="done"))
        with patch("subprocess.run", stub):
            _validate_hip(str(repro), {"args": []}, 1e-6, 1e-5)

        run_argv, run_kwargs = stub.calls[1]
        assert run_argv == ["./reproducer"]
        assert run_kwargs["cwd"] == str(repro)

    def test_run_failure(self, repro):
        with patch("subprocess.run", sequence(completed(), completed(1, stderr="segfault"))):
            result = _validate_hip(str(repro), {"args": []}, 1e-6, 1e-5)

        assert result.passed is False
        assert "Run failed (exit 1)" in result.details[1]

    def test_no_hsaco_check_on_this_path(self, repro):
        """Unlike the HSACO path, a HIP reproducer needs no separate artifact."""
        with (
            patch("subprocess.run", sequence(completed(), completed())),
            patch("kerncap.validator._compare_outputs") as cmp,
        ):
            _validate_hip(str(repro), {"args": []}, 1e-6, 1e-5)

        cmp.assert_called_once()


# --------------------------------------------------------------------------
# _validate_triton
# --------------------------------------------------------------------------


class TestValidateTriton:
    def test_runs_the_python_reproducer_with_no_build_step(self, repro):
        """Triton re-JITs itself, so there is nothing to make."""
        stub = sequence(completed(stdout="jitted"))
        with patch("subprocess.run", stub):
            _validate_triton(str(repro), {"args": []}, 1e-6, 1e-5)

        assert len(stub.calls) == 1
        run_argv, run_kwargs = stub.calls[0]
        assert run_argv == ["python3", "reproducer.py"]
        assert run_kwargs["cwd"] == str(repro)

    def test_run_failure_has_no_build_detail_to_keep(self, repro):
        with patch("subprocess.run", sequence(completed(1, stderr="ImportError: triton"))):
            result = _validate_triton(str(repro), {"args": []}, 1e-6, 1e-5)

        assert result.passed is False
        assert len(result.details) == 1
        assert "ImportError: triton" in result.details[0]

    def test_success_flows_into_output_comparison(self, repro):
        with (
            patch("subprocess.run", sequence(completed(stdout="ok"))),
            patch("kerncap.validator._compare_outputs") as cmp,
        ):
            _validate_triton(str(repro), {"args": []}, 1e-6, 1e-5)

        assert "ok" in cmp.call_args.args[4][0]


# --------------------------------------------------------------------------
# _compare_outputs
# --------------------------------------------------------------------------


class TestCompareOutputs:
    def test_matching_buffers_pass(self, repro):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        write_buffers(repro, 0, data, data.copy())

        result = _compare_outputs(str(repro), {"args": [out_arg()]}, 1e-6, 1e-5, [])

        assert result.passed is True

    def test_const_pointers_are_not_treated_as_outputs(self, repro):
        """Inputs should not be compared — only non-const pointers are outputs."""
        meta = {"args": [out_arg(is_const=True), {"index": 1, "is_pointer": False}]}

        result = _compare_outputs(str(repro), meta, 1e-6, 1e-5, [])

        assert result.passed is True

    def test_missing_output_file_fails(self, repro):
        data = np.array([1.0], dtype=np.float32)
        (repro / "capture").mkdir(exist_ok=True)
        data.tofile(repro / "capture" / "arg_0.bin")

        result = _compare_outputs(str(repro), {"args": [out_arg()]}, 1e-6, 1e-5, [])

        assert result.passed is False
        assert any("MISSING output file" in d for d in result.details)

    def test_missing_reference_file_fails(self, repro):
        (repro / "reference_output").mkdir()
        np.array([1.0], dtype=np.float32).tofile(repro / "reference_output" / "output_0.bin")

        result = _compare_outputs(str(repro), {"args": [out_arg()]}, 1e-6, 1e-5, [])

        assert result.passed is False
        assert any("MISSING reference file" in d for d in result.details)

    def test_prefers_the_post_kernel_reference(self, repro):
        """ref_output_file is the post-kernel snapshot; file is pre-kernel."""
        (repro / "capture").mkdir(exist_ok=True)
        (repro / "reference_output").mkdir()
        np.array([9.0], dtype=np.float32).tofile(repro / "capture" / "arg_0.bin")
        np.array([1.0], dtype=np.float32).tofile(repro / "capture" / "arg_0_out.bin")
        np.array([1.0], dtype=np.float32).tofile(repro / "reference_output" / "output_0.bin")

        meta = {"args": [out_arg(ref_output_file="arg_0_out.bin")]}
        result = _compare_outputs(str(repro), meta, 1e-6, 1e-5, [])

        assert result.passed is True

    def test_falls_back_to_the_pre_kernel_capture(self, repro):
        """When ref_output_file names a file that was never written."""
        data = np.array([4.0], dtype=np.float32)
        write_buffers(repro, 0, data, data.copy())

        meta = {"args": [out_arg(ref_output_file="absent.bin")]}
        result = _compare_outputs(str(repro), meta, 1e-6, 1e-5, [])

        assert result.passed is True

    def test_shape_mismatch_fails(self, repro):
        write_buffers(
            repro,
            0,
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        )

        result = _compare_outputs(str(repro), {"args": [out_arg()]}, 1e-6, 1e-5, [])

        assert result.passed is False
        assert any("SHAPE MISMATCH" in d for d in result.details)

    def test_empty_buffers_are_reported_but_not_a_failure(self, repro):
        """A zero-element output is degenerate, not wrong."""
        empty = np.array([], dtype=np.float32)
        write_buffers(repro, 0, empty, empty)

        result = _compare_outputs(str(repro), {"args": [out_arg()]}, 1e-6, 1e-5, [])

        assert any("EMPTY (0 elements)" in d for d in result.details)
        assert result.passed is True

    def test_torch_dtype_takes_priority_over_c_type(self, repro):
        """Triton captures carry torch_dtype; misreading it changes the shape."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
        write_buffers(repro, 0, data, data.copy())

        meta = {"args": [out_arg(torch_dtype="torch.float16", type="float*")]}
        result = _compare_outputs(str(repro), meta, 1e-6, 1e-5, [])

        assert result.passed is True

    def test_a_wrong_dtype_shows_up_as_a_shape_mismatch(self, repro):
        """8 float16 bytes read as float32 is 2 elements, not 4."""
        (repro / "capture").mkdir(exist_ok=True)
        (repro / "reference_output").mkdir()
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16).tofile(repro / "capture" / "arg_0.bin")
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tofile(
            repro / "reference_output" / "output_0.bin"
        )

        result = _compare_outputs(str(repro), {"args": [out_arg()]}, 1e-6, 1e-5, [])

        assert result.passed is False
        assert any("SHAPE MISMATCH" in d for d in result.details)


# --------------------------------------------------------------------------
# dtype inference
# --------------------------------------------------------------------------


class TestDtypeInference:
    @pytest.mark.parametrize(
        ("torch_str", "expected"),
        [
            ("torch.float16", np.float16),
            ("torch.float32", np.float32),
            ("torch.float64", np.float64),
            ("torch.int8", np.int8),
            ("torch.int16", np.int16),
            ("torch.int32", np.int32),
            ("torch.int64", np.int64),
            ("torch.bool", np.bool_),
            ("torch.uint8", np.uint8),
        ],
    )
    def test_torch_dtypes_map_directly(self, torch_str, expected):
        assert _infer_numpy_dtype_from_torch(torch_str) == np.dtype(expected)

    def test_bfloat16_is_carried_as_raw_bits(self):
        """numpy has no native bf16, so the bytes are compared as uint16."""
        assert _infer_numpy_dtype_from_torch("torch.bfloat16") == np.dtype(np.uint16)

    def test_unknown_torch_dtype_falls_back_to_float32(self):
        assert _infer_numpy_dtype_from_torch("torch.complex128") == np.dtype(np.float32)

    def test_unknown_c_type_falls_back_to_float32(self):
        assert _infer_numpy_dtype("struct Foo*") == np.dtype(np.float32)


# --------------------------------------------------------------------------
# _validate_replay — routing, auto-detect, and failure paths
# --------------------------------------------------------------------------


class TestValidateReplay:
    def test_missing_replay_binary_is_reported(self, repro):
        (repro / "capture" / "dispatch.json").write_text("{}")

        with patch(
            "kerncap._get_replay_path", side_effect=FileNotFoundError("kerncap-replay missing")
        ):
            result = validate_reproducer(str(repro))

        assert result.passed is False
        assert "kerncap-replay missing" in result.details[0]

    @pytest.mark.parametrize("name", ["candidate.hsaco", "optimized.hsaco"])
    def test_auto_detects_a_rebuilt_hsaco(self, repro, name):
        """The documented edit loop: rebuild then `kerncap validate <dir>`.

        Triton's reproducer.py writes candidate.hsaco; `make recompile`
        writes optimized.hsaco. Either must be picked up without a flag.
        """
        (repro / "capture" / "dispatch.json").write_text("{}")
        (repro / name).write_bytes(b"\x7fELF")

        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("kerncap.validator._validate_replay_variant") as variant,
        ):
            validate_reproducer(str(repro))

        assert variant.call_args.args[2] == str(repro / name)
        details = variant.call_args.args[3]
        assert any("Auto-detected rebuilt HSACO" in d for d in details)

    def test_candidate_wins_over_optimized(self, repro):
        (repro / "capture" / "dispatch.json").write_text("{}")
        (repro / "candidate.hsaco").write_bytes(b"\x7fELF")
        (repro / "optimized.hsaco").write_bytes(b"\x7fELF")

        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("kerncap.validator._validate_replay_variant") as variant,
        ):
            validate_reproducer(str(repro))

        assert variant.call_args.args[2].endswith("candidate.hsaco")

    def test_an_explicit_hsaco_skips_auto_detection(self, repro):
        (repro / "capture" / "dispatch.json").write_text("{}")
        (repro / "candidate.hsaco").write_bytes(b"\x7fELF")

        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("kerncap.validator._validate_replay_variant") as variant,
        ):
            validate_reproducer(str(repro), hsaco="/explicit.hsaco")

        assert variant.call_args.args[2] == "/explicit.hsaco"
        assert not any("Auto-detected" in d for d in variant.call_args.args[3])

    def test_no_rebuilt_hsaco_runs_the_smoke_test(self, repro):
        (repro / "capture" / "dispatch.json").write_text("{}")

        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("kerncap.validator._validate_replay_baseline") as baseline,
        ):
            validate_reproducer(str(repro))

        baseline.assert_called_once()


class TestRunReplay:
    def test_reports_ok_without_a_timing_line(self, repro):
        """Not every replay prints timing; the header must still appear."""
        from kerncap.validator import _run_replay

        details = []
        with patch("subprocess.run", sequence(completed(stdout="Replaying\nDone\n"))):
            proc = _run_replay("/bin/replay", str(repro), details, label="Replay")

        assert proc is not None
        assert details[0] == "Replay: OK"

    def test_reports_ok_with_a_timing_line(self, repro):
        from kerncap.validator import _run_replay

        details = []
        stdout = "Replaying\nAverage GPU time: 12.5 us\n"
        with patch("subprocess.run", sequence(completed(stdout=stdout))):
            _run_replay("/bin/replay", str(repro), details, label="Baseline")

        assert details[0] == "Baseline: OK (Average GPU time: 12.5 us)"

    def test_builds_the_command_with_flags(self, repro):
        from kerncap.validator import _run_replay

        stub = sequence(completed())
        with patch("subprocess.run", stub):
            _run_replay("/bin/replay", str(repro), [], hsaco="/v.hsaco", dump_output=True)

        argv = stub.calls[0][0]
        assert argv[:2] == ["/bin/replay", str(repro)]
        assert "--dump-output" in argv
        assert argv[argv.index("--hsaco") + 1] == "/v.hsaco"


class TestValidateReplayVariantFailures:
    def test_baseline_producing_no_output_dir_fails(self, repro):
        from kerncap.validator import _validate_replay_variant

        details = []
        with patch("subprocess.run", sequence(completed())):
            result = _validate_replay_variant(
                "/bin/replay", str(repro / "capture"), "/v.hsaco", details
            )

        assert result.passed is False
        assert result.mode == "byte-exact"
        assert "Baseline produced no output directory" in details

    def test_variant_replay_failure_is_reported(self, repro):
        """Baseline succeeds and dumps output; the variant run then fails."""
        from kerncap.validator import _validate_replay_variant

        capture = repro / "capture"

        def _run(argv, **kwargs):
            if "--hsaco" in argv:
                return completed(1, stderr="variant crashed")
            (capture / "output").mkdir(exist_ok=True)
            (capture / "output" / "region_0.bin").write_bytes(b"\x00")
            return completed()

        details = []
        with patch("subprocess.run", _run):
            result = _validate_replay_variant("/bin/replay", str(capture), "/v.hsaco", details)

        assert result.passed is False
        assert any("variant crashed" in d for d in details)

    def test_variant_producing_no_output_dir_fails(self, repro):
        from kerncap.validator import _validate_replay_variant

        capture = repro / "capture"
        calls = []

        def _run(argv, **kwargs):
            calls.append(argv)
            if len(calls) == 1:
                (capture / "output").mkdir(exist_ok=True)
                (capture / "output" / "region_0.bin").write_bytes(b"\x00")
            return completed()

        details = []
        with patch("subprocess.run", _run):
            result = _validate_replay_variant("/bin/replay", str(capture), "/v.hsaco", details)

        assert result.passed is False
        assert "Variant produced no output directory" in details


class TestCompareReplayOutputExtras:
    def test_a_region_only_in_the_variant_is_reported(self, tmp_path):
        """Asymmetric: extra regions matter as much as missing ones."""
        from kerncap.validator import _compare_replay_outputs

        base = tmp_path / "base"
        var = tmp_path / "var"
        base.mkdir()
        var.mkdir()
        (base / "region_a.bin").write_bytes(b"\x01")
        (var / "region_a.bin").write_bytes(b"\x01")
        (var / "region_b.bin").write_bytes(b"\x02")

        details = []
        result = _compare_replay_outputs(str(base), str(var), details)

        assert result.passed is False
        assert any("region_b.bin: MISSING in baseline output" in d for d in details)

    def test_empty_regions_count_as_identical(self, tmp_path):
        from kerncap.validator import _compare_replay_outputs

        base = tmp_path / "base"
        var = tmp_path / "var"
        base.mkdir()
        var.mkdir()
        (base / "region_a.bin").write_bytes(b"")
        (var / "region_a.bin").write_bytes(b"")

        result = _compare_replay_outputs(str(base), str(var), [])

        assert result.passed is True
        assert result.regions_identical == 1
        assert any("PASS (empty)" in line for line in result.region_lines)


class TestUnreachableMetadataRead:
    def test_the_second_metadata_read_cannot_run(self, repro):
        """``validate_reproducer``'s bare-Triton metadata read is dead code.

        Reaching it needs ``reproducer.py`` present *and* the legacy branch
        not to have returned — but the legacy branch returns whenever
        ``metadata.json`` exists and ``reproducer.py`` is present. So when
        control reaches the second block, ``metadata.json`` is always absent
        and the inner ``if`` is always False.

        Pinned so the dead branch is visible rather than looking like
        untested behaviour.
        """
        (repro / "capture" / "metadata.json").write_text('{"args": [{"index": 7}]}')
        (repro / "reproducer.py").write_text("")

        with patch("kerncap.validator._validate_triton") as triton:
            validate_reproducer(str(repro))

        # Routed by the legacy branch, which passes the parsed metadata.
        assert triton.call_args.args[1] == {"args": [{"index": 7}]}
