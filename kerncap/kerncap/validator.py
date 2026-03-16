"""Reproducer correctness validator.

For VA-faithful captures (HIP kernels), validates using kerncap-replay:
  - Baseline (no --hsaco): smoke test only — confirms replay runs without
    crashing but does not check numerical correctness.
  - Variant (--hsaco provided): runs replay twice (captured HSACO vs
    variant HSACO), compares post-execution memory byte-for-byte, and
    fails on any difference.

For Triton reproducers, runs the Python reproducer and compares outputs
against captured reference data using numpy allclose.
"""

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class ValidationResult:
    """Result of validating a reproducer."""
    passed: bool
    details: List[str]
    max_error: float = 0.0


def validate_reproducer(
    reproducer_dir: str,
    tolerance: float = 1e-6,
    rtol: float = 1e-5,
    hsaco: Optional[str] = None,
) -> ValidationResult:
    """Run a reproducer and compare outputs to captured reference.

    Parameters
    ----------
    reproducer_dir : str
        Path to the reproducer project directory.
    tolerance : float
        Absolute tolerance for np.allclose.
    rtol : float
        Relative tolerance for np.allclose.
    hsaco : str, optional
        Path to an alternative HSACO to replay instead of the captured
        one.  Used to validate recompiled / optimized kernels.

    Returns
    -------
    ValidationResult
    """
    capture_dir = os.path.join(reproducer_dir, "capture")

    # New VA-faithful format: dispatch.json + memory_regions.json
    if os.path.isfile(os.path.join(capture_dir, "dispatch.json")):
        return _validate_replay(reproducer_dir, tolerance, rtol, hsaco=hsaco)

    # Legacy format: metadata.json
    meta_path = os.path.join(capture_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        if os.path.exists(os.path.join(reproducer_dir, "harness.hip")):
            return _validate_hsaco(reproducer_dir, metadata, tolerance, rtol)
        elif os.path.exists(os.path.join(reproducer_dir, "reproducer.hip")):
            return _validate_hip(reproducer_dir, metadata, tolerance, rtol)
        elif os.path.exists(os.path.join(reproducer_dir, "reproducer.py")):
            return _validate_triton(reproducer_dir, metadata, tolerance, rtol)

    # Triton reproducer (no capture dir needed)
    if os.path.exists(os.path.join(reproducer_dir, "reproducer.py")):
        meta_path = os.path.join(capture_dir, "metadata.json")
        metadata = {}
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
        return _validate_triton(reproducer_dir, metadata, tolerance, rtol)

    return ValidationResult(
        passed=False,
        details=["No dispatch.json, metadata.json, or reproducer.py found"],
    )


def _validate_replay(
    reproducer_dir: str,
    atol: float,
    rtol: float,
    hsaco: Optional[str] = None,
) -> ValidationResult:
    """Validate using kerncap-replay.

    Without ``--hsaco``: smoke test only — confirms the captured kernel
    replays without crashing.

    With ``--hsaco``: runs replay twice (captured HSACO, then variant
    HSACO), compares post-execution memory regions byte-for-byte, and
    fails on any difference.
    """
    from kerncap import _get_replay_path

    details: List[str] = []
    capture_dir = os.path.join(reproducer_dir, "capture")

    try:
        replay_bin = _get_replay_path()
    except FileNotFoundError as e:
        return ValidationResult(passed=False, details=[str(e)])

    if hsaco:
        return _validate_replay_variant(
            replay_bin, capture_dir, hsaco, details,
        )
    return _validate_replay_baseline(replay_bin, capture_dir, details)


def _run_replay(
    replay_bin: str,
    capture_dir: str,
    details: List[str],
    hsaco: Optional[str] = None,
    dump_output: bool = False,
) -> Optional[subprocess.CompletedProcess]:
    """Run kerncap-replay and append stdout to *details*.

    Returns the CompletedProcess on success, or ``None`` after
    appending failure info to *details*.
    """
    cmd = [replay_bin, capture_dir]
    if dump_output:
        cmd.append("--dump-output")
    if hsaco:
        cmd.extend(["--hsaco", hsaco])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        details.append(
            f"Replay failed (exit {proc.returncode}):\n{proc.stderr}"
        )
        return None

    timing_line = ""
    for line in proc.stdout.splitlines():
        if "Average GPU time:" in line:
            timing_line = line.strip()
        details.append(line.rstrip())

    label = "Replay"
    if timing_line:
        details.insert(0, f"{label}: OK ({timing_line})")
    else:
        details.insert(0, f"{label}: OK")

    return proc


def _validate_replay_baseline(
    replay_bin: str,
    capture_dir: str,
    details: List[str],
) -> ValidationResult:
    """Smoke test: confirm the captured kernel replays without crashing."""
    proc = _run_replay(replay_bin, capture_dir, details)
    if proc is None:
        return ValidationResult(passed=False, details=details)

    details.append(
        "Baseline replay OK (smoke test — no variant HSACO to compare against)"
    )
    return ValidationResult(passed=True, details=details)


def _validate_replay_variant(
    replay_bin: str,
    capture_dir: str,
    hsaco: str,
    details: List[str],
) -> ValidationResult:
    """Two-replay comparison: captured HSACO vs variant HSACO.

    Runs the baseline replay with --dump-output, stashes the output
    directory, runs the variant replay, then compares every output
    region byte-for-byte.
    """
    # --- baseline replay ---
    baseline_proc = _run_replay(
        replay_bin, capture_dir, details, dump_output=True,
    )
    if baseline_proc is None:
        return ValidationResult(passed=False, details=details)

    output_dir = os.path.join(capture_dir, "output")
    if not os.path.isdir(output_dir):
        details.append("Baseline produced no output directory")
        return ValidationResult(passed=False, details=details)

    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = os.path.join(tmpdir, "output_baseline")
        shutil.move(output_dir, baseline_dir)

        # --- variant replay ---
        details.append("")
        details.append(f"Variant HSACO: {hsaco}")
        variant_proc = _run_replay(
            replay_bin, capture_dir, details,
            hsaco=hsaco, dump_output=True,
        )
        if variant_proc is None:
            return ValidationResult(passed=False, details=details)

        if not os.path.isdir(output_dir):
            details.append("Variant produced no output directory")
            return ValidationResult(passed=False, details=details)

        return _compare_replay_outputs(baseline_dir, output_dir, details)


def _compare_replay_outputs(
    baseline_dir: str,
    variant_dir: str,
    details: List[str],
) -> ValidationResult:
    """Byte-exact comparison of two replay output directories."""
    baseline_files = sorted(
        f for f in os.listdir(baseline_dir)
        if f.startswith("region_") and f.endswith(".bin")
    )
    variant_files = sorted(
        f for f in os.listdir(variant_dir)
        if f.startswith("region_") and f.endswith(".bin")
    )

    if not baseline_files:
        details.append("No output region files to compare")
        return ValidationResult(passed=True, details=details)

    matched = set(baseline_files) & set(variant_files)
    only_baseline = set(baseline_files) - set(variant_files)
    only_variant = set(variant_files) - set(baseline_files)

    all_passed = True

    for f in sorted(only_baseline):
        details.append(f"  {f}: MISSING in variant output")
        all_passed = False
    for f in sorted(only_variant):
        details.append(f"  {f}: MISSING in baseline output")
        all_passed = False

    for region_file in sorted(matched):
        base_data = np.fromfile(
            os.path.join(baseline_dir, region_file), dtype=np.uint8,
        )
        var_data = np.fromfile(
            os.path.join(variant_dir, region_file), dtype=np.uint8,
        )

        if base_data.size != var_data.size:
            details.append(
                f"  {region_file}: SIZE MISMATCH "
                f"baseline={base_data.size:,} variant={var_data.size:,}"
            )
            all_passed = False
            continue

        if base_data.size == 0:
            details.append(f"  {region_file}: PASS (empty)")
            continue

        diff_count = int(np.sum(base_data != var_data))
        if diff_count == 0:
            details.append(f"  {region_file}: PASS (identical)")
        else:
            pct = 100.0 * diff_count / base_data.size
            details.append(
                f"  {region_file}: FAIL — {diff_count:,} bytes differ "
                f"({pct:.1f}% of {base_data.size:,} bytes)"
            )
            all_passed = False

    return ValidationResult(passed=all_passed, details=details)


def _validate_hsaco(
    reproducer_dir: str,
    metadata: dict,
    atol: float,
    rtol: float,
) -> ValidationResult:
    """Build and run an HSACO-based reproducer."""
    details: List[str] = []

    # Build the harness (and optionally the .hsaco from kernel.hip)
    build_proc = subprocess.run(
        ["make", "-C", reproducer_dir, "all"],
        capture_output=True, text=True,
    )
    if build_proc.returncode != 0:
        return ValidationResult(
            passed=False,
            details=[f"Build failed:\n{build_proc.stderr}"],
        )
    details.append("Build: OK")

    # Check that kernel.hsaco exists
    hsaco_path = os.path.join(reproducer_dir, "kernel.hsaco")
    if not os.path.exists(hsaco_path):
        return ValidationResult(
            passed=False,
            details=details + ["kernel.hsaco not found — cannot run harness"],
        )

    # Run
    run_proc = subprocess.run(
        ["./harness", "kernel.hsaco"],
        cwd=reproducer_dir,
        capture_output=True, text=True,
    )
    if run_proc.returncode != 0:
        return ValidationResult(
            passed=False,
            details=details + [
                f"Run failed (exit {run_proc.returncode}):\n{run_proc.stderr}"
            ],
        )
    details.append(f"Run: OK\n{run_proc.stdout.strip()}")

    return _compare_outputs(reproducer_dir, metadata, atol, rtol, details)


def _validate_hip(
    reproducer_dir: str,
    metadata: dict,
    atol: float,
    rtol: float,
) -> ValidationResult:
    """Build and run a HIP reproducer."""
    details: List[str] = []

    # Build
    build_proc = subprocess.run(
        ["make", "-C", reproducer_dir, "all"],
        capture_output=True, text=True,
    )
    if build_proc.returncode != 0:
        return ValidationResult(
            passed=False,
            details=[f"Build failed:\n{build_proc.stderr}"],
        )
    details.append("Build: OK")

    # Run
    run_proc = subprocess.run(
        ["./reproducer"],
        cwd=reproducer_dir,
        capture_output=True, text=True,
    )
    if run_proc.returncode != 0:
        return ValidationResult(
            passed=False,
            details=details + [f"Run failed (exit {run_proc.returncode}):\n{run_proc.stderr}"],
        )
    details.append(f"Run: OK\n{run_proc.stdout.strip()}")

    return _compare_outputs(reproducer_dir, metadata, atol, rtol, details)


def _validate_triton(
    reproducer_dir: str,
    metadata: dict,
    atol: float,
    rtol: float,
) -> ValidationResult:
    """Run a Triton reproducer."""
    details: List[str] = []

    run_proc = subprocess.run(
        ["python3", "reproducer.py"],
        cwd=reproducer_dir,
        capture_output=True, text=True,
    )
    if run_proc.returncode != 0:
        return ValidationResult(
            passed=False,
            details=[f"Run failed (exit {run_proc.returncode}):\n{run_proc.stderr}"],
        )
    details.append(f"Run: OK\n{run_proc.stdout.strip()}")

    return _compare_outputs(reproducer_dir, metadata, atol, rtol, details)


def _compare_outputs(
    reproducer_dir: str,
    metadata: dict,
    atol: float,
    rtol: float,
    details: List[str],
) -> ValidationResult:
    """Compare reproducer outputs to captured reference data."""
    args = metadata.get("args", [])
    output_args = [a for a in args if a.get("is_pointer") and not a.get("is_const")]

    if not output_args:
        details.append("No output arguments to validate")
        return ValidationResult(passed=True, details=details)

    all_passed = True
    max_error = 0.0

    for arg in output_args:
        idx = arg["index"]
        # Prefer post-kernel reference output (ref_output_file) when
        # available; fall back to the pre-kernel capture (file).
        ref_file = arg.get("ref_output_file", arg["file"])
        ref_path = os.path.join(reproducer_dir, "capture", ref_file)
        if not os.path.exists(ref_path):
            ref_path = os.path.join(reproducer_dir, "capture", arg["file"])
        out_path = os.path.join(reproducer_dir, "reference_output", f"output_{idx}.bin")

        if not os.path.exists(out_path):
            details.append(f"  arg_{idx}: MISSING output file")
            all_passed = False
            continue

        if not os.path.exists(ref_path):
            details.append(f"  arg_{idx}: MISSING reference file")
            all_passed = False
            continue

        # Infer dtype — Triton captures use "torch_dtype", HIP uses "type"
        torch_dtype_str = arg.get("torch_dtype")
        if torch_dtype_str:
            dtype = _infer_numpy_dtype_from_torch(torch_dtype_str)
        else:
            dtype = _infer_numpy_dtype(arg.get("type", "float*"))

        ref_data = np.fromfile(ref_path, dtype=dtype)
        out_data = np.fromfile(out_path, dtype=dtype)

        if ref_data.shape != out_data.shape:
            details.append(
                f"  arg_{idx}: SHAPE MISMATCH ref={ref_data.shape} out={out_data.shape}"
            )
            all_passed = False
            continue

        if ref_data.size == 0:
            details.append(f"  arg_{idx}: EMPTY (0 elements)")
            continue

        diff = np.abs(ref_data.astype(np.float64) - out_data.astype(np.float64))
        nan_mask = np.isnan(diff)
        nan_count = int(np.sum(nan_mask))

        if nan_count > 0 and nan_count == diff.size:
            error = float("nan")
        elif nan_count > 0:
            error = float(np.nanmax(diff))
        else:
            error = float(np.max(diff))

        if nan_count > 0:
            max_error = float("nan")
        elif not math.isnan(max_error):
            max_error = max(max_error, error)

        close = np.allclose(ref_data, out_data, atol=atol, rtol=rtol, equal_nan=False)

        status = "PASS" if close and nan_count == 0 else "FAIL"
        if nan_count > 0:
            details.append(
                f"  arg_{idx}: {status} ({nan_count:,} NaN element(s) in diff "
                f"out of {diff.size:,})"
            )
            details.append(
                f"           Note: NaN values typically indicate uninitialized "
                f"device memory, half-precision overflow, or buffer size "
                f"misinterpretation (wrong dtype)."
            )
        else:
            details.append(f"  arg_{idx}: {status} (max_error={error:.2e}, atol={atol})")
        if status == "FAIL":
            all_passed = False

    return ValidationResult(
        passed=all_passed,
        details=details,
        max_error=max_error,
    )


def _infer_numpy_dtype(type_str: str) -> np.dtype:
    """Infer numpy dtype from a C/C++ type string."""
    t = type_str.replace("const ", "").replace("*", "").strip()
    mapping = {
        "float": np.float32,
        "double": np.float64,
        "int": np.int32,
        "unsigned int": np.uint32,
        "long": np.int64,
        "unsigned long": np.uint64,
        "short": np.int16,
        "unsigned short": np.uint16,
        "char": np.int8,
        "unsigned char": np.uint8,
        "__half": np.float16,
        "half": np.float16,
        "_Float16": np.float16,
        "__hip_bfloat16": np.float16,  # approximate
        "uint8_t": np.uint8,
        "int8_t": np.int8,
        "uint16_t": np.uint16,
        "int16_t": np.int16,
        "uint32_t": np.uint32,
        "int32_t": np.int32,
        "uint64_t": np.uint64,
        "int64_t": np.int64,
        "size_t": np.uint64,
    }
    return np.dtype(mapping.get(t, np.float32))


def _infer_numpy_dtype_from_torch(torch_dtype_str: str) -> np.dtype:
    """Infer numpy dtype from a torch dtype string (e.g. 'torch.float16')."""
    mapping = {
        "torch.float16": np.float16,
        "torch.float32": np.float32,
        "torch.float64": np.float64,
        "torch.bfloat16": np.uint16,   # raw bits; no native numpy bf16
        "torch.int8": np.int8,
        "torch.int16": np.int16,
        "torch.int32": np.int32,
        "torch.int64": np.int64,
        "torch.bool": np.bool_,
        "torch.uint8": np.uint8,
    }
    return np.dtype(mapping.get(torch_dtype_str, np.float32))
