# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Accordo tests: inline HIP kernels, compile, capture, and compare.
- Reduction: baseline vs optimized (parametrized tolerance); vector_add self-compare.
- Template kernels: scale_values<float> and scale_values<double> (parametrized).
- Mismatch detection: reference vs wrong output (assert validation fails).
- Validator/snapshot: kernel_args extraction, snapshot attributes, summary().
- IPC robustness: kernel never dispatched, Python dies before "done".
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from accordo import Accordo
from accordo.exceptions import AccordoKernelNeverDispatched

# -----------------------------------------------------------------------------
# Reduction: shared main(), kernel body varies (baseline, optimized, wrong)
# -----------------------------------------------------------------------------

REDUCE_KERNEL_BASELINE = """
__global__ void reduce_sum(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(output, input[idx]);
    }
}
"""

REDUCE_KERNEL_OPTIMIZED = """
__global__ void reduce_sum(const float* input, float* output, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}
"""

REDUCE_KERNEL_WRONG = """
__global__ void reduce_sum(const float* input, float* output, int N) {
    (void)input;
    (void)N;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *output = 0.0f;
    }
}
"""

REDUCE_MAIN = """
int main() {
    const int N = 1024;
    float *d_input, *d_output;
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, sizeof(float));
    float* h_input = (float*)malloc(N * sizeof(float));
    float h_output = 0.0f;
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;
    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_output, &h_output, sizeof(float), hipMemcpyHostToDevice);
    int gridSize = (N + 255) / 256;
    hipLaunchKernelGGL(reduce_sum, dim3(gridSize), dim3(256), 0, 0, d_input, d_output, N);
    hipDeviceSynchronize();
    hipMemcpy(&h_output, d_output, sizeof(float), hipMemcpyDeviceToHost);
    printf("%.2f\\n", h_output);
    free(h_input);
    hipFree(d_input);
    hipFree(d_output);
    return 0;
}
"""

# Main that never dispatches reduce_sum (binary still has the symbol for kernelDB).
REDUCE_MAIN_NO_DISPATCH = """
int main() {
    hipInit(0);
    return 0;
}
"""


def _reduce_source(kernel_body: str) -> str:
    """Build full HIP source: includes + kernel body + shared main()."""
    return (
        "#include <hip/hip_runtime.h>\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n" + kernel_body.strip() + "\n" + REDUCE_MAIN
    )


def _reduce_source_no_dispatch(kernel_body: str) -> str:
    """HIP source with kernel defined but main() never dispatches it."""
    return (
        "#include <hip/hip_runtime.h>\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n" + kernel_body.strip() + "\n" + REDUCE_MAIN_NO_DISPATCH
    )


# -----------------------------------------------------------------------------
# Template kernel: scale_values<T> — one template, instantiate for float/double
# -----------------------------------------------------------------------------

SCALE_VALUES_TEMPLATE = """
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

template<typename T>
__global__ void scale_values(T* input, T* output, T factor, int N) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) output[idx] = input[idx] * factor;
}}

int main() {{
    const int N = 1024;
    {dtype} *d_in, *d_out;
    hipMalloc(&d_in, N * sizeof({dtype}));
    hipMalloc(&d_out, N * sizeof({dtype}));
    {dtype}* h_in = ({dtype}*)malloc(N * sizeof({dtype}));
    for (int i = 0; i < N; i++) h_in[i] = ({dtype})i;
    hipMemcpy(d_in, h_in, N * sizeof({dtype}), hipMemcpyHostToDevice);
    hipLaunchKernelGGL((scale_values<{dtype}>), dim3(4), dim3(256), 0, 0, d_in, d_out, {factor}, N);
    hipDeviceSynchronize();
    free(h_in);
    hipFree(d_in);
    hipFree(d_out);
    return 0;
}}
"""


def _scale_values_source(dtype: str) -> str:
    """Build scale_values HIP source for dtype ('float' or 'double')."""
    factor = "2.0f" if dtype == "float" else "2.0"
    return SCALE_VALUES_TEMPLATE.format(dtype=dtype, factor=factor)


# -----------------------------------------------------------------------------
# Vector add: same kernel twice (identity comparison)
# -----------------------------------------------------------------------------

VECTOR_ADD_KERNEL = """
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vector_add(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }
    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(vector_add, dim3((N + 255) / 256), dim3(256), 0, 0, d_a, d_b, d_c, N);
    hipDeviceSynchronize();
    hipMemcpy(h_a, d_c, bytes, hipMemcpyDeviceToHost);
    printf("%.2f\\n", h_a[0]);
    free(h_a);
    free(h_b);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    return 0;
}
"""


def _compile_hip(kernel_code: str, name: str, tmp_dir: Path) -> Path:
    """Write HIP source, compile with hipcc, return path to binary."""
    src = tmp_dir / f"{name}.hip"
    bin_path = tmp_dir / name
    src.write_text(kernel_code)
    r = subprocess.run(
        ["hipcc", str(src), "-o", str(bin_path), "-O2", "-g"],
        capture_output=True,
        text=True,
        cwd=tmp_dir,
        timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"hipcc failed for {name}:\n{r.stderr}")
    return bin_path


@pytest.mark.parametrize("tolerance", [1e-3, 1e-4, 1e-5, 1e-6])
def test_reduce_sum_baseline_vs_optimized(tolerance):
    """Reference and optimized reduce_sum match within tolerance."""
    with tempfile.TemporaryDirectory(prefix="accordo_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        baseline_bin = _compile_hip(_reduce_source(REDUCE_KERNEL_BASELINE), "baseline", tmp_path)
        optimized_bin = _compile_hip(_reduce_source(REDUCE_KERNEL_OPTIMIZED), "optimized", tmp_path)
        validator = Accordo(
            binary=str(baseline_bin),
            kernel_name="reduce_sum",
            working_directory=str(tmp_path),
        )
        ref_snap = validator.capture_snapshot(binary=str(baseline_bin), timeout_seconds=30)
        opt_snap = validator.capture_snapshot(binary=str(optimized_bin), timeout_seconds=30)
        result = validator.compare_snapshots(ref_snap, opt_snap, tolerance=tolerance)
    assert result.is_valid, result.summary()
    assert result.num_arrays_validated >= 1


def test_vector_add_self_compare():
    """Same vector_add binary vs itself (identity) always validates."""
    with tempfile.TemporaryDirectory(prefix="accordo_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_KERNEL, "vector_add", tmp_path)
        validator = Accordo(
            binary=str(bin_path),
            kernel_name="vector_add",
            working_directory=str(tmp_path),
        )
        snap1 = validator.capture_snapshot(binary=str(bin_path), timeout_seconds=30)
        snap2 = validator.capture_snapshot(binary=str(bin_path), timeout_seconds=30)
        result = validator.compare_snapshots(snap1, snap2, tolerance=1e-6)
    assert result.is_valid, result.summary()
    assert result.num_arrays_validated >= 1


# -----------------------------------------------------------------------------
# Template kernels: scale_values<float> and scale_values<double>
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float", "double"], ids=["float", "double"])
def test_template_scale_values_self_compare(dtype):
    """Template scale_values<T> (float and double): same binary vs itself validates."""
    with tempfile.TemporaryDirectory(prefix="accordo_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_name = f"scale_{dtype}"
        bin_path = _compile_hip(_scale_values_source(dtype), bin_name, tmp_path)
        validator = Accordo(
            binary=str(bin_path),
            kernel_name="scale_values",
            working_directory=str(tmp_path),
        )
        snap1 = validator.capture_snapshot(binary=str(bin_path), timeout_seconds=30)
        snap2 = validator.capture_snapshot(binary=str(bin_path), timeout_seconds=30)
        result = validator.compare_snapshots(snap1, snap2, tolerance=1e-5)
    assert result.is_valid, result.summary()
    assert result.num_arrays_validated >= 1


# -----------------------------------------------------------------------------
# Mismatch detection: reference vs wrong output must fail
# -----------------------------------------------------------------------------


def test_reduce_sum_mismatch_detected():
    """Reference reduce_sum vs wrong kernel (output=0) fails validation."""
    with tempfile.TemporaryDirectory(prefix="accordo_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        ref_bin = _compile_hip(_reduce_source(REDUCE_KERNEL_BASELINE), "ref", tmp_path)
        wrong_bin = _compile_hip(_reduce_source(REDUCE_KERNEL_WRONG), "wrong", tmp_path)
        validator = Accordo(
            binary=str(ref_bin),
            kernel_name="reduce_sum",
            working_directory=str(tmp_path),
        )
        ref_snap = validator.capture_snapshot(binary=str(ref_bin), timeout_seconds=30)
        wrong_snap = validator.capture_snapshot(binary=str(wrong_bin), timeout_seconds=30)
        result = validator.compare_snapshots(ref_snap, wrong_snap, tolerance=1e-6)
    assert not result.is_valid
    assert result.num_mismatches >= 1
    assert len(result.mismatches) >= 1
    mismatch = result.mismatches[0]
    assert mismatch.arg_name
    assert mismatch.max_difference >= 0
    assert result.summary() and "failed" in result.summary().lower()


# -----------------------------------------------------------------------------
# Validator and snapshot structure
# -----------------------------------------------------------------------------


def test_validator_extracts_kernel_args():
    """Accordo extracts kernel arguments from binary (reduce_sum: input, output, N)."""
    with tempfile.TemporaryDirectory(prefix="accordo_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(_reduce_source(REDUCE_KERNEL_BASELINE), "baseline", tmp_path)
        validator = Accordo(
            binary=str(bin_path),
            kernel_name="reduce_sum",
            working_directory=str(tmp_path),
        )
    assert len(validator.kernel_args) >= 2
    arg_names = [name for name, _ in validator.kernel_args]
    assert "input" in arg_names or "N" in arg_names or len(arg_names) == 3


def test_capture_snapshot_returns_valid_snapshot():
    """capture_snapshot returns Snapshot with arrays and execution_time_ms."""
    with tempfile.TemporaryDirectory(prefix="accordo_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(_reduce_source(REDUCE_KERNEL_BASELINE), "baseline", tmp_path)
        validator = Accordo(
            binary=str(bin_path),
            kernel_name="reduce_sum",
            working_directory=str(tmp_path),
        )
        snap = validator.capture_snapshot(binary=str(bin_path), timeout_seconds=30)
    assert hasattr(snap, "arrays")
    assert isinstance(snap.arrays, list)
    assert len(snap.arrays) >= 1
    assert hasattr(snap, "execution_time_ms")
    assert isinstance(snap.execution_time_ms, (int, float))
    assert snap.execution_time_ms >= 0


def test_validation_result_summary():
    """ValidationResult.summary() returns non-empty string for pass and fail."""
    with tempfile.TemporaryDirectory(prefix="accordo_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        ref_bin = _compile_hip(_reduce_source(REDUCE_KERNEL_BASELINE), "ref", tmp_path)
        wrong_bin = _compile_hip(_reduce_source(REDUCE_KERNEL_WRONG), "wrong", tmp_path)
        validator = Accordo(
            binary=str(ref_bin),
            kernel_name="reduce_sum",
            working_directory=str(tmp_path),
        )
        ref_snap = validator.capture_snapshot(binary=str(ref_bin), timeout_seconds=30)
        wrong_snap = validator.capture_snapshot(binary=str(wrong_bin), timeout_seconds=30)
        fail_result = validator.compare_snapshots(ref_snap, wrong_snap, tolerance=1e-6)
        pass_result = validator.compare_snapshots(ref_snap, ref_snap, tolerance=1e-6)
    assert not fail_result.is_valid
    assert fail_result.summary()
    assert "fail" in fail_result.summary().lower() or "mismatch" in fail_result.summary().lower()
    assert pass_result.is_valid
    assert pass_result.summary()
    assert "pass" in pass_result.summary().lower() or "match" in pass_result.summary().lower()


# -----------------------------------------------------------------------------
# IPC robustness: kernel never dispatched, Python dies before "done"
# -----------------------------------------------------------------------------


def test_kernel_never_dispatched_fails_fast():
    """When the target kernel is never dispatched, we should get a clear
    error quickly (AccordoKernelNeverDispatched), not a generic TimeoutError after 30s.

    Binary has reduce_sum symbol but main() never launches it. Without sentinel behavior,
    Python waits for IPC file until timeout. With sentinel behavior, C++ writes a sentinel and
    Python raises AccordoKernelNeverDispatched quickly.
    """
    with tempfile.TemporaryDirectory(prefix="accordo_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        no_dispatch_bin = _compile_hip(
            _reduce_source_no_dispatch(REDUCE_KERNEL_BASELINE), "no_dispatch", tmp_path
        )
        validator = Accordo(
            binary=str(no_dispatch_bin),
            kernel_name="reduce_sum",
            working_directory=str(tmp_path),
        )
        t0 = time.perf_counter()
        with pytest.raises(AccordoKernelNeverDispatched):
            validator.capture_snapshot(binary=str(no_dispatch_bin), timeout_seconds=15)
        elapsed = time.perf_counter() - t0
    # Should fail fast (< 10s), not after full timeout
    assert elapsed < 10.0, (
        "Expected AccordoKernelNeverDispatched quickly; got it after %.1fs. "
        "Without sentinel behavior this can wait for timeout." % elapsed
    )


def _get_child_pids(pid: int) -> list[int]:
    """Return list of child PIDs of the given process (Linux)."""
    # Prefer /proc (Linux) so we don't depend on ps syntax
    proc_children = Path(f"/proc/{pid}/task/{pid}/children")
    if not proc_children.exists():
        proc_children = Path(f"/proc/{pid}/children")
    if proc_children.exists():
        try:
            return [int(p) for p in proc_children.read_text().split()]
        except (OSError, ValueError):
            pass
    try:
        out = subprocess.run(
            ["ps", "-o", "pid=", "--ppid", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode != 0:
            return []
        return [int(p) for p in out.stdout.strip().split() if p.strip()]
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        return []


def test_app_exits_when_python_dies_before_sending_done():
    """When Python (driver) dies before sending 'done', the instrumented
    app blocks in C++ read() forever. With the PID watchdog fix, C++ should notice the
    parent is dead and exit, so the app process should terminate.

    We run capture_snapshot in a subprocess, wait for the kernel to run (C++ to block),
    kill the Python subprocess, then assert the instrumented app (child) exits within 10s.
    """
    driver_script = """
import sys
from pathlib import Path
from accordo import Accordo

tmp_dir = sys.argv[1]
pid_file = sys.argv[2]
bin_path = sys.argv[3]

Path(pid_file).write_text(str(__import__("os").getpid()))
validator = Accordo(
    binary=bin_path,
    kernel_name="reduce_sum",
    working_directory=tmp_dir,
)
validator.capture_snapshot(binary=bin_path, timeout_seconds=60)
"""
    with tempfile.TemporaryDirectory(prefix="accordo_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(_reduce_source(REDUCE_KERNEL_BASELINE), "reduction", tmp_path)
        pid_file = tmp_path / "driver_pid.txt"
        driver = subprocess.Popen(
            [sys.executable, "-c", driver_script, str(tmp_path), str(pid_file), str(bin_path)],
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for driver to write its PID and enter capture_snapshot (binary started)
        for _ in range(50):
            if pid_file.exists():
                break
            time.sleep(0.1)
        else:
            driver.kill()
            pytest.fail("Driver did not write pid_file")
        driver_pid = int(pid_file.read_text().strip())
        # Wait for driver to spawn the instrumented binary (validator init + capture_snapshot)
        child_pids = []
        for _ in range(120):  # up to 12s
            child_pids = _get_child_pids(driver_pid)
            if child_pids:
                break
            time.sleep(0.1)
        if not child_pids:
            driver.kill()
            pytest.skip("Could not get child PIDs (e.g. non-Linux or ps not available)")
        # Give time for kernel to run and C++ to block on read()
        time.sleep(2.0)
        driver.kill()
        driver.wait(timeout=2)
        # With PID watchdog, the instrumented app should exit when parent died
        app_pid = child_pids[0]
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                os.kill(app_pid, 0)
            except OSError:
                break
            time.sleep(0.2)
        else:
            try:
                os.kill(app_pid, 9)
            except OSError:
                pass
            pytest.fail(
                "Instrumented app did not exit within 10s after Python died. "
                "C++ may be blocked on read() when Python never sends 'done'."
            )
