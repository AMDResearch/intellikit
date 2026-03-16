"""Integration test for a batched kernel with embedded device pointers.

This test verifies the full pipeline for kernels with T** (double-pointer)
arguments. The VA-faithful capture snapshots all tracked device memory at
original virtual addresses, so embedded pointers are inherently valid on
replay without any pointer chasing or relocation.

Exercises: capture -> source find -> reproducer -> replay -> validate

Requires ROCm installation with hipcc and an AMD GPU.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.integration.conftest import skip_no_gpu, skip_no_rocprof


BATCHED_SCALE_KERNEL = """\
#pragma once
#include <hip/hip_runtime.h>

// Batched vector scale: out[batch][i] = in[batch][i] * scale
// Uses T** pattern with embedded device pointers.
__global__ void batched_scale(
    const float* const* inputs,
    float** outputs,
    float scale,
    int n,
    int batch_count
) {
    int batch = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_count && idx < n) {
        outputs[batch][idx] = inputs[batch][idx] * scale;
    }
}
"""

BATCHED_SCALE_DRIVER = """\
#include <cstdio>
#include <cstdlib>
#include "batched_scale.hpp"

int main() {
    const int N = 1024;
    const int BATCH = 4;
    const float SCALE = 2.0f;
    const size_t elem_bytes = N * sizeof(float);

    // Host arrays of pointers
    float* h_inputs[BATCH];
    float* h_outputs[BATCH];
    float* d_inputs[BATCH];
    float* d_outputs[BATCH];

    // Allocate and initialize each batch on host, then copy to device
    for (int b = 0; b < BATCH; b++) {
        h_inputs[b] = new float[N];
        h_outputs[b] = new float[N];
        for (int i = 0; i < N; i++) {
            h_inputs[b][i] = static_cast<float>(b * N + i);
            h_outputs[b][i] = 0.0f;
        }
        hipMalloc(&d_inputs[b], elem_bytes);
        hipMalloc(&d_outputs[b], elem_bytes);
        hipMemcpy(d_inputs[b], h_inputs[b], elem_bytes, hipMemcpyHostToDevice);
    }

    // Device array of pointers (the T** args)
    float** d_input_ptrs;
    float** d_output_ptrs;
    hipMalloc(&d_input_ptrs, BATCH * sizeof(float*));
    hipMalloc(&d_output_ptrs, BATCH * sizeof(float*));
    hipMemcpy(d_input_ptrs, d_inputs, BATCH * sizeof(float*), hipMemcpyHostToDevice);
    hipMemcpy(d_output_ptrs, d_outputs, BATCH * sizeof(float*), hipMemcpyHostToDevice);

    // Launch
    dim3 block(256);
    dim3 grid((N + 255) / 256, BATCH);
    batched_scale<<<grid, block>>>(d_input_ptrs, d_output_ptrs, SCALE, N, BATCH);
    hipDeviceSynchronize();

    // Copy back and verify
    bool pass = true;
    for (int b = 0; b < BATCH; b++) {
        hipMemcpy(h_outputs[b], d_outputs[b], elem_bytes, hipMemcpyDeviceToHost);
        for (int i = 0; i < N; i++) {
            float expected = h_inputs[b][i] * SCALE;
            if (fabsf(h_outputs[b][i] - expected) > 1e-5f) {
                fprintf(stderr, "Mismatch at batch %d, idx %d: %f != %f\\n",
                        b, i, h_outputs[b][i], expected);
                pass = false;
            }
        }
    }

    // Cleanup
    for (int b = 0; b < BATCH; b++) {
        hipFree(d_inputs[b]);
        hipFree(d_outputs[b]);
        delete[] h_inputs[b];
        delete[] h_outputs[b];
    }
    hipFree(d_input_ptrs);
    hipFree(d_output_ptrs);

    if (pass) {
        printf("batched_scale: PASS\\n");
        return 0;
    } else {
        printf("batched_scale: FAIL\\n");
        return 1;
    }
}
"""


@pytest.fixture
def batched_scale_app(tmp_path):
    """Compile the batched_scale test application."""
    if not shutil.which("hipcc"):
        pytest.skip("hipcc not available")

    header = tmp_path / "batched_scale.hpp"
    header.write_text(BATCHED_SCALE_KERNEL)

    src = tmp_path / "batched_scale.hip"
    src.write_text(BATCHED_SCALE_DRIVER)

    binary = tmp_path / "batched_scale"
    result = subprocess.run(
        ["hipcc", "-O2", "-g", "-o", str(binary), str(src)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"hipcc failed:\n{result.stderr}")

    return str(binary), str(tmp_path)


@skip_no_gpu
@skip_no_rocprof
class TestBatchedScaleEmbeddedPointers:
    """Integration tests for a kernel with T** (double-pointer) arguments.

    The VA-faithful capture snapshots all tracked device memory at
    original virtual addresses, so embedded pointers are inherently
    valid on replay — no pointer chasing or relocation needed.
    """

    def test_capture(self, batched_scale_app, tmp_path):
        """Capture the batched_scale kernel and verify dispatch metadata."""
        binary, workdir = batched_scale_app
        from kerncap.capturer import run_capture

        output_dir = str(tmp_path / "capture")
        run_capture(
            kernel_name="batched_scale",
            cmd=[binary],
            output_dir=output_dir,
        )

        meta_path = os.path.join(output_dir, "dispatch.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(output_dir, "metadata.json")
        assert os.path.exists(meta_path), \
            f"Expected dispatch.json or metadata.json in {output_dir}"

        with open(meta_path) as f:
            meta = json.load(f)

        kernel_name = meta.get("kernel_name") or meta.get("demangled_name", "")
        assert "batched_scale" in kernel_name

        grid = meta.get("grid", {})
        grid_y = grid.get("y") if isinstance(grid, dict) else (grid[1] if grid else None)
        assert grid_y == 4, f"Expected grid_y=4 (BATCH), got {grid_y}"

    def test_source_finder(self, batched_scale_app):
        """Find the batched_scale kernel source in the header."""
        _, workdir = batched_scale_app
        from kerncap.source_finder import find_kernel_source

        result = find_kernel_source(
            kernel_name="batched_scale",
            source_dir=workdir,
            language="hip",
        )
        assert result is not None
        assert result.language == "hip"
        assert result.kernel_function == "batched_scale"
        assert result.main_file.endswith("batched_scale.hpp")

    def test_full_pipeline(self, batched_scale_app, tmp_path):
        """Full capture -> source -> reproducer -> validate pipeline.

        The key value of this test: the kernel dereferences T** arguments
        (device arrays of device pointers). VA-faithful replay restores all
        memory at original VAs, so the embedded pointers remain valid.
        """
        binary, workdir = batched_scale_app

        from kerncap.capturer import run_capture
        from kerncap.source_finder import find_kernel_source
        from kerncap.reproducer import generate_hsaco_reproducer
        from kerncap.validator import validate_reproducer

        capture_dir = str(tmp_path / "capture")
        run_capture(
            kernel_name="batched_scale",
            cmd=[binary],
            output_dir=capture_dir,
        )

        kernel_src = find_kernel_source(
            kernel_name="batched_scale",
            source_dir=workdir,
            language="hip",
        )
        assert kernel_src is not None
        assert kernel_src.kernel_function == "batched_scale"

        repro_dir = str(tmp_path / "reproducer")
        generate_hsaco_reproducer(
            capture_dir, repro_dir,
            kernel_source=kernel_src,
        )

        assert Path(repro_dir, "Makefile").exists()
        makefile = Path(repro_dir, "Makefile").read_text()
        assert "kerncap-replay" in makefile

        result = validate_reproducer(repro_dir, tolerance=1e-4)
        assert result.passed, f"Validation failed: {result.details}"
