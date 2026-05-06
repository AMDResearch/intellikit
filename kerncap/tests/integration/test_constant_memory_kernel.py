"""Integration test for kernels that read from `__constant__` memory.

This is the mechanical analogue of what fails today on LAMMPS
TagPairEAMKernelC (Kokkos's hip_parallel_launch_constant_memory<...>):
the kernel reads state populated via hipMemcpyToSymbol that lives in
the HSACO module variable, not in any tracked allocation. Without
the module-variable snapshot/restore (libkerncap snapshot_module_variables +
replay STAGE 4.5), the constant buffer is zero on replay and the
kernel either faults or returns garbage.

Exercises: capture -> source find -> reproducer -> replay -> validate
with the constant-memory restore path enabled.

Requires ROCm installation with hipcc and an AMD GPU.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.integration.conftest import skip_no_gpu, skip_no_rocprof


CONSTANT_KERNEL = """\
#pragma once
#include <hip/hip_runtime.h>

// 256 weights packed into __constant__ memory and consumed by the kernel.
// The kernel reads exclusively from c_weights, so the captured-vs-replayed
// output matches only if the constant buffer is restored by replay.
__constant__ float c_weights[256];

__global__ void weighted_add(const float* a, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Pull a deterministic weight from the constant table so any
        // zero-initialized replay buffer produces a clearly different
        // (in fact: zero) result.
        float w = c_weights[idx % 256];
        out[idx] = a[idx] * w + w;
    }
}
"""

CONSTANT_DRIVER = """\
#include <cstdio>
#include <cstdlib>
#include "weighted_add.hpp"

int main() {
    const int N = 65536;
    const size_t bytes = N * sizeof(float);

    float h_weights[256];
    for (int i = 0; i < 256; ++i) {
        h_weights[i] = 1.0f + 0.01f * static_cast<float>(i);
    }

    // Populate the device-side __constant__ buffer the same way Kokkos
    // populates kokkos_impl_hip_constant_memory_buffer for its launchers.
    hipMemcpyToSymbol(HIP_SYMBOL(c_weights), h_weights, sizeof(h_weights));

    float* h_a = new float[N];
    float* h_out = new float[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i % 1024);
    }

    float *d_a, *d_out;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_out, bytes);
    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);

    int block = 256;
    int grid = (N + block - 1) / block;

    weighted_add<<<grid, block>>>(d_a, d_out, N);
    hipDeviceSynchronize();

    hipMemcpy(h_out, d_out, bytes, hipMemcpyDeviceToHost);

    bool pass = true;
    for (int i = 0; i < N; ++i) {
        float w = h_weights[i % 256];
        float expected = h_a[i] * w + w;
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            fprintf(stderr, "Mismatch at %d: %f != %f\\n", i, h_out[i], expected);
            pass = false;
            break;
        }
    }

    hipFree(d_a);
    hipFree(d_out);
    delete[] h_a;
    delete[] h_out;

    if (pass) {
        printf("weighted_add: PASS\\n");
        return 0;
    } else {
        printf("weighted_add: FAIL\\n");
        return 1;
    }
}
"""


@pytest.fixture
def weighted_add_app(tmp_path):
    """Compile a HIP program with __constant__ memory populated via
    hipMemcpyToSymbol, plus a kernel that reads from it."""
    if not shutil.which("hipcc"):
        pytest.skip("hipcc not available")

    header = tmp_path / "weighted_add.hpp"
    header.write_text(CONSTANT_KERNEL)

    src = tmp_path / "weighted_add.hip"
    src.write_text(CONSTANT_DRIVER)

    binary = tmp_path / "weighted_add"
    result = subprocess.run(
        ["hipcc", "-O2", "-g", "-o", str(binary), str(src)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"hipcc failed:\n{result.stderr}")

    return str(binary), str(tmp_path)


@skip_no_gpu
@skip_no_rocprof
class TestConstantMemoryKernel:
    """End-to-end coverage for kernels that read __constant__ memory.

    Without the module-variable snapshot/restore plumbing
    (libkerncap.so snapshot_module_variables + replay.cpp STAGE 4.5),
    the replay binary would dispatch against a zero-initialized constant
    buffer and validation would fail (output bytes would be ~zero
    instead of the captured non-zero results).
    """

    def test_capture_writes_module_variables_manifest(self, weighted_add_app, tmp_path):
        """A capture of a kernel using __constant__ memory must produce
        module_variables.json and at least one blob."""
        binary, _ = weighted_add_app
        from kerncap.capturer import run_capture

        capture_dir = tmp_path / "capture"
        run_capture(
            kernel_name="weighted_add",
            cmd=[binary],
            output_dir=str(capture_dir),
        )

        manifest_path = capture_dir / "module_variables.json"
        assert manifest_path.exists(), (
            "Expected module_variables.json from libkerncap's snapshot_module_variables() pass"
        )

        manifest = json.loads(manifest_path.read_text())
        assert "variables" in manifest
        names = [v["name"] for v in manifest["variables"]]
        # The HIP frontend mangles c_weights into a hidden symbol; we just
        # require that *some* variable was captured (the kernel uses one).
        assert len(names) >= 1, f"No module variables captured: {manifest}"

        # Each entry must have all required fields.
        required = {"executable_sha256", "name", "size", "blob"}
        for v in manifest["variables"]:
            assert required.issubset(v.keys()), v
            blob_path = capture_dir / v["blob"]
            assert blob_path.exists(), f"Missing blob: {blob_path}"
            assert blob_path.stat().st_size == v["size"], (
                f"Blob size {blob_path.stat().st_size} != manifest size {v['size']} for {v['name']}"
            )

    def test_full_pipeline_validates_byte_exact(self, weighted_add_app, tmp_path):
        """Full capture -> reproducer -> validate. With STAGE 4.5 active
        the replay output should match the captured output."""
        binary, workdir = weighted_add_app

        from kerncap.capturer import run_capture
        from kerncap.source_finder import find_kernel_source
        from kerncap.reproducer import generate_hsaco_reproducer
        from kerncap.validator import validate_reproducer

        capture_dir = str(tmp_path / "capture")
        run_capture(
            kernel_name="weighted_add",
            cmd=[binary],
            output_dir=capture_dir,
        )

        kernel_src = find_kernel_source(
            kernel_name="weighted_add",
            source_dir=workdir,
            language="hip",
        )
        assert kernel_src is not None

        repro_dir = str(tmp_path / "reproducer")
        generate_hsaco_reproducer(
            capture_dir,
            repro_dir,
            kernel_source=kernel_src,
        )

        assert Path(repro_dir, "Makefile").exists()

        result = validate_reproducer(repro_dir, tolerance=1e-4)
        assert result.passed, (
            "Validation failed — likely the constant buffer was not restored "
            f"on replay: {result.details}"
        )

    def test_disable_variable_snapshot_skips_manifest(
        self, weighted_add_app, tmp_path, monkeypatch
    ):
        """KERNCAP_DISABLE_VARIABLE_SNAPSHOT=1 must short-circuit the
        snapshot pass; the manifest may exist as `{"variables": []}` or
        not exist at all (depending on capture wiring), but it must not
        contain any captured blobs."""
        binary, _ = weighted_add_app
        from kerncap.capturer import run_capture

        monkeypatch.setenv("KERNCAP_DISABLE_VARIABLE_SNAPSHOT", "1")

        capture_dir = tmp_path / "capture_disabled"
        run_capture(
            kernel_name="weighted_add",
            cmd=[binary],
            output_dir=str(capture_dir),
        )

        manifest_path = capture_dir / "module_variables.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            assert manifest.get("variables", []) == [], (
                "KERNCAP_DISABLE_VARIABLE_SNAPSHOT=1 should not produce "
                f"variable entries; got {manifest}"
            )
        var_dir = capture_dir / "module_variables"
        if var_dir.exists():
            assert list(var_dir.iterdir()) == [], (
                "KERNCAP_DISABLE_VARIABLE_SNAPSHOT=1 should not write blob files"
            )
