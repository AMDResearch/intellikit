"""Basic end-to-end integration test with a simple HIP vector_add kernel.

This test does NOT require Docker — just a ROCm installation with hipcc,
rocprofv3, and an AMD GPU. It exercises the full pipeline:
  profile -> extract (capture + source find + reproducer) -> validate
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import skip markers from conftest
from tests.integration.conftest import skip_no_gpu, skip_no_rocprof


VECTOR_ADD_KERNEL = """\
#pragma once
#include <hip/hip_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

VECTOR_ADD_DRIVER = """\
#include <cstdio>
#include <cstdlib>
#include "vector_add.hpp"

int main() {
    const int N = 65536;
    const size_t bytes = N * sizeof(float);

    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    hipDeviceSynchronize();

    hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            fprintf(stderr, "Mismatch at %d: %f != %f\\n",
                    i, h_c[i], h_a[i] + h_b[i]);
            return 1;
        }
    }
    printf("vector_add: PASS\\n");

    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
    return 0;
}
"""


@pytest.fixture
def vector_add_app(tmp_path):
    """Compile the vector_add test application.

    Uses a realistic layout: kernel in a header (vector_add.hpp),
    driver with main() in a separate source file (vector_add.hip).
    """
    if not shutil.which("hipcc"):
        pytest.skip("hipcc not available")

    header = tmp_path / "vector_add.hpp"
    header.write_text(VECTOR_ADD_KERNEL)

    src = tmp_path / "vector_add.hip"
    src.write_text(VECTOR_ADD_DRIVER)

    binary = tmp_path / "vector_add"
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
class TestVectorAddE2E:
    """End-to-end test with a simple vector_add kernel."""

    def test_profile(self, vector_add_app):
        """Profile the vector_add app and find the kernel."""
        binary, workdir = vector_add_app
        from kerncap.profiler import run_profile

        kernels = run_profile([binary])
        assert len(kernels) > 0

        # The vector_add kernel should be present
        names = [k.name for k in kernels]
        assert any("vector_add" in n for n in names), (
            f"vector_add not found in kernel list: {names}"
        )

    def test_capture(self, vector_add_app, tmp_path):
        """Capture the vector_add kernel dispatch."""
        binary, workdir = vector_add_app
        from kerncap.capturer import run_capture

        output_dir = str(tmp_path / "capture")
        run_capture(
            kernel_name="vector_add",
            cmd=[binary],
            output_dir=output_dir,
        )

        meta_path = os.path.join(output_dir, "dispatch.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(output_dir, "metadata.json")
        assert os.path.exists(meta_path), f"Expected dispatch.json or metadata.json in {output_dir}"

        with open(meta_path) as f:
            meta = json.load(f)

        kernel_name = meta.get("kernel_name") or meta.get("demangled_name", "")
        assert "vector_add" in kernel_name
        grid = meta.get("grid", {})
        grid_x = grid.get("x") if isinstance(grid, dict) else (grid[0] if grid else None)
        assert grid_x is not None and grid_x > 0
        block = meta.get("block", {})
        block_x = block.get("x") if isinstance(block, dict) else (block[0] if block else None)
        assert block_x == 256

    def test_source_finder(self, vector_add_app):
        """Find the vector_add kernel source in the header."""
        binary, workdir = vector_add_app
        from kerncap.source_finder import find_kernel_source

        result = find_kernel_source(
            kernel_name="vector_add",
            source_dir=workdir,
            language="hip",
        )
        assert result is not None
        assert result.language == "hip"
        assert result.kernel_function == "vector_add"
        # Source finder should identify the header, not the driver
        assert result.main_file.endswith("vector_add.hpp")

    def test_full_pipeline(self, vector_add_app, tmp_path):
        """Full profile -> capture -> source -> reproducer -> validate."""
        binary, workdir = vector_add_app

        from kerncap.profiler import run_profile
        from kerncap.capturer import run_capture
        from kerncap.source_finder import find_kernel_source
        from kerncap.reproducer import generate_hsaco_reproducer
        from kerncap.validator import validate_reproducer

        # 1. Profile
        kernels = run_profile([binary])
        assert any("vector_add" in k.name for k in kernels)

        # 2. Capture
        capture_dir = str(tmp_path / "capture")
        run_capture(
            kernel_name="vector_add",
            cmd=[binary],
            output_dir=capture_dir,
        )

        # 3. Find source
        kernel_src = find_kernel_source(
            kernel_name="vector_add",
            source_dir=workdir,
            language="hip",
        )
        assert kernel_src is not None

        # 4. Generate reproducer (HSACO-based)
        repro_dir = str(tmp_path / "reproducer")
        generate_hsaco_reproducer(
            capture_dir,
            repro_dir,
            kernel_source=kernel_src,
        )

        assert os.path.exists(os.path.join(repro_dir, "Makefile"))
        makefile = Path(os.path.join(repro_dir, "Makefile")).read_text()
        assert "kerncap-replay" in makefile

        # 5. Validate
        result = validate_reproducer(repro_dir, tolerance=1e-4)
        assert result.passed, f"Validation failed: {result.details}"
