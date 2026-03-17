"""Integration test with a single HIP file containing multiple kernels.

This tests kerncap's ability to handle a single-TU application that defines
multiple GPU kernels — a common real-world pattern not covered by the other
integration tests (which use one kernel per file).

The app runs a five-kernel pipeline (vector_add, vector_scale,
vector_bias_relu, vector_shift, histogram_atomic) iteratively and validates
results on the host.

Requires ROCm installation with hipcc, rocprofv3, and an AMD GPU.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.integration.conftest import skip_no_gpu, skip_no_rocprof


MINI_PIPELINE_SOURCE = """\
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <limits>
#include <utility>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_scale(float* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

__global__ void vector_bias_relu(float* data, float bias, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = data[idx] + bias;
        data[idx] = (v > 0.0f) ? v : 0.0f;
    }
}

__global__ void vector_shift(const float* in, float* out, float shift, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] + shift;
    }
}

__global__ void histogram_atomic(const int* keys, int n, unsigned int* bins,
                                 int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int key = keys[idx];
        atomicAdd(&bins[key % num_bins], 1u);
    }
}

#define HIP_CHECK(cmd)                                                         \\
    do {                                                                       \\
        hipError_t err = (cmd);                                                \\
        if (err != hipSuccess) {                                               \\
            std::fprintf(stderr, "HIP error at %s:%d: %s\\n", __FILE__,        \\
                         __LINE__, hipGetErrorString(err));                     \\
            return 1;                                                          \\
        }                                                                      \\
    } while (0)

static void fill_inputs(std::vector<float>& a, std::vector<float>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        float x = static_cast<float>(i % 1024) * 0.001f;
        a[i] = std::sinf(x);
        b[i] = std::cosf(x) - 0.5f;
    }
}

static void fill_hot_keys(std::vector<int>& keys, int hot_bins) {
    for (size_t i = 0; i < keys.size(); ++i) {
        if ((i & 15u) != 0u) {
            keys[i] = 0;
        } else {
            keys[i] = static_cast<int>((i >> 4) % static_cast<size_t>(hot_bins));
        }
    }
}

int main() {
    const int n = 1 << 18;
    const int iters = 20;

    const float scale = 1.15f;
    const float bias = -0.10f;
    const float shift = 0.03f;
    const int histogram_multiplier = 4;
    const int num_bins = 32;
    const int hot_bins = 4;
    const long long hist_n_ll = static_cast<long long>(n) * histogram_multiplier;
    if (hist_n_ll <= 0 || hist_n_ll > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "Invalid histogram size for n=%d\\n", n);
        return 2;
    }
    const int hist_n = static_cast<int>(hist_n_ll);

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t hist_bytes = static_cast<size_t>(hist_n) * sizeof(int);
    const size_t bins_bytes = static_cast<size_t>(num_bins) * sizeof(unsigned int);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_out(n);
    std::vector<float> h_ref(n);
    std::vector<float> h_tmp(n);

    std::vector<int> h_keys(hist_n);
    std::vector<unsigned int> h_bins(num_bins, 0u);
    std::vector<unsigned int> h_bins_ref(num_bins, 0u);

    fill_inputs(h_a, h_b);
    fill_hot_keys(h_keys, hot_bins);
    h_ref = h_a;

    for (int key : h_keys) {
        h_bins_ref[key % num_bins] += 1u;
    }
    for (int i = 0; i < num_bins; ++i) {
        h_bins_ref[i] *= static_cast<unsigned int>(iters);
    }

    float *d_a = nullptr, *d_b = nullptr, *d_tmp = nullptr, *d_out = nullptr;
    int* d_keys = nullptr;
    unsigned int* d_bins = nullptr;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_tmp, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMalloc(&d_keys, hist_bytes));
    HIP_CHECK(hipMalloc(&d_bins, bins_bytes));

    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_keys, h_keys.data(), hist_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_bins, 0, bins_bytes));

    const dim3 block(256);
    const dim3 grid((n + block.x - 1) / block.x);
    const dim3 hist_grid((hist_n + block.x - 1) / block.x);

    for (int iter = 0; iter < iters; ++iter) {
        vector_add<<<grid, block>>>(d_a, d_b, d_tmp, n);
        vector_scale<<<grid, block>>>(d_tmp, scale, n);
        vector_bias_relu<<<grid, block>>>(d_tmp, bias, n);
        vector_shift<<<grid, block>>>(d_tmp, d_out, shift, n);
        histogram_atomic<<<hist_grid, block>>>(d_keys, hist_n, d_bins, num_bins);
        std::swap(d_a, d_out);
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_a, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_bins.data(), d_bins, bins_bytes, hipMemcpyDeviceToHost));

    for (int iter = 0; iter < iters; ++iter) {
        for (int i = 0; i < n; ++i) {
            float v = h_ref[i] + h_b[i];
            v *= scale;
            v += bias;
            v = (v > 0.0f) ? v : 0.0f;
            h_tmp[i] = v + shift;
        }
        std::swap(h_ref, h_tmp);
    }

    float max_abs_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_abs_err = std::max(max_abs_err, std::fabs(h_out[i] - h_ref[i]));
    }

    int bad_bins = 0;
    unsigned long long total_counts = 0;
    for (int i = 0; i < num_bins; ++i) {
        total_counts += static_cast<unsigned long long>(h_bins[i]);
        if (h_bins[i] != h_bins_ref[i]) {
            ++bad_bins;
        }
    }
    const unsigned long long expected_counts =
        static_cast<unsigned long long>(hist_n) * static_cast<unsigned long long>(iters);

    if (max_abs_err > 5e-5f || bad_bins != 0 || total_counts != expected_counts) {
        std::fprintf(stderr,
                     "mini_pipeline: FAIL (max_abs_err=%g, bad_bins=%d, total=%llu, expected=%llu)\\n",
                     max_abs_err, bad_bins, total_counts, expected_counts);
        return 1;
    }

    double checksum = 0.0;
    for (int i = 0; i < std::min(n, 16); ++i) {
        checksum += h_out[i];
    }
    std::printf(
        "mini_pipeline: PASS (n=%d, iters=%d, hist_n=%d, max_abs_err=%g, hot_bin0=%u, checksum16=%.6f)\\n",
        n, iters, hist_n, max_abs_err, h_bins[0], checksum);

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_tmp));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_keys));
    HIP_CHECK(hipFree(d_bins));
    return 0;
}
"""

ALL_KERNEL_NAMES = [
    "vector_add",
    "vector_scale",
    "vector_bias_relu",
    "vector_shift",
    "histogram_atomic",
]


@pytest.fixture
def mini_pipeline_app(tmp_path):
    """Compile the mini_pipeline application from a single .hip file."""
    if not shutil.which("hipcc"):
        pytest.skip("hipcc not available")

    src = tmp_path / "mini_pipeline.hip"
    src.write_text(MINI_PIPELINE_SOURCE)

    binary = tmp_path / "mini_pipeline"
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
class TestMiniPipelineMultiKernel:
    """End-to-end tests for a single HIP file containing multiple kernels."""

    def test_profile_finds_all_kernels(self, mini_pipeline_app):
        """Profiling should discover all five kernels in the single TU."""
        binary, workdir = mini_pipeline_app
        from kerncap.profiler import run_profile

        kernels = run_profile([binary])
        assert len(kernels) > 0

        found = {k.name for k in kernels}
        for name in ALL_KERNEL_NAMES:
            assert any(name in f for f in found), f"{name} not found in profiled kernels: {found}"

    def test_capture_vector_scale(self, mini_pipeline_app, tmp_path):
        """Capture a kernel that is NOT the first one defined in the file."""
        binary, workdir = mini_pipeline_app
        from kerncap.capturer import run_capture

        output_dir = str(tmp_path / "capture")
        run_capture(
            kernel_name="vector_scale",
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
        assert "vector_scale" in kernel_name
        block = meta.get("block", {})
        block_x = block.get("x") if isinstance(block, dict) else (block[0] if block else None)
        assert block_x == 256

    def test_capture_histogram_atomic(self, mini_pipeline_app, tmp_path):
        """Capture the histogram kernel that uses atomics and a different grid."""
        binary, workdir = mini_pipeline_app
        from kerncap.capturer import run_capture

        output_dir = str(tmp_path / "capture")
        run_capture(
            kernel_name="histogram_atomic",
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
        assert "histogram_atomic" in kernel_name
        block = meta.get("block", {})
        block_x = block.get("x") if isinstance(block, dict) else (block[0] if block else None)
        assert block_x == 256
        # histogram uses a larger grid than the vector kernels
        grid = meta.get("grid", {})
        grid_x = grid.get("x") if isinstance(grid, dict) else (grid[0] if grid else None)
        assert grid_x is not None and grid_x > 0

    def test_source_finder_single_file(self, mini_pipeline_app):
        """Source finder should locate kernels in the single .hip file."""
        binary, workdir = mini_pipeline_app
        from kerncap.source_finder import find_kernel_source

        for name in ALL_KERNEL_NAMES:
            result = find_kernel_source(
                kernel_name=name,
                source_dir=workdir,
                language="hip",
            )
            assert result is not None, f"source_finder failed for {name}"
            assert result.language == "hip"
            assert result.kernel_function == name
            assert result.main_file.endswith("mini_pipeline.hip")

    def test_full_pipeline_vector_scale(self, mini_pipeline_app, tmp_path):
        """Full pipeline for vector_scale: profile -> capture -> source -> reproducer -> validate."""
        binary, workdir = mini_pipeline_app

        from kerncap.profiler import run_profile
        from kerncap.capturer import run_capture
        from kerncap.source_finder import find_kernel_source
        from kerncap.reproducer import generate_hsaco_reproducer
        from kerncap.validator import validate_reproducer

        kernels = run_profile([binary])
        assert any("vector_scale" in k.name for k in kernels)

        capture_dir = str(tmp_path / "capture")
        run_capture(
            kernel_name="vector_scale",
            cmd=[binary],
            output_dir=capture_dir,
        )

        kernel_src = find_kernel_source(
            kernel_name="vector_scale",
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

        assert os.path.exists(os.path.join(repro_dir, "Makefile"))
        makefile = Path(os.path.join(repro_dir, "Makefile")).read_text()
        assert "kerncap-replay" in makefile

        result = validate_reproducer(repro_dir, tolerance=1e-4)
        assert result.passed, f"Validation failed: {result.details}"

    def test_full_pipeline_histogram_atomic(self, mini_pipeline_app, tmp_path):
        """Full pipeline for histogram_atomic — exercises atomics and int/uint args."""
        binary, workdir = mini_pipeline_app

        from kerncap.profiler import run_profile
        from kerncap.capturer import run_capture
        from kerncap.source_finder import find_kernel_source
        from kerncap.reproducer import generate_hsaco_reproducer
        from kerncap.validator import validate_reproducer

        kernels = run_profile([binary])
        assert any("histogram_atomic" in k.name for k in kernels)

        capture_dir = str(tmp_path / "capture")
        run_capture(
            kernel_name="histogram_atomic",
            cmd=[binary],
            output_dir=capture_dir,
        )

        kernel_src = find_kernel_source(
            kernel_name="histogram_atomic",
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

        assert os.path.exists(os.path.join(repro_dir, "Makefile"))
        makefile = Path(os.path.join(repro_dir, "Makefile")).read_text()
        assert "kerncap-replay" in makefile

        result = validate_reproducer(repro_dir, tolerance=1e-4)
        assert result.passed, f"Validation failed: {result.details}"
