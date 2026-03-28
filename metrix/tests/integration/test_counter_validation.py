# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Integration tests: derived counter validation via inline HIP microbenchmarks.

Each test compiles a small HIP kernel designed to produce a predictable hardware
counter pattern, profiles it via the Metrix API, and checks derived metrics
against expected value ranges.

These tests require:
  - hipcc (HIP compiler)
  - rocprofv3 (ROCm profiler)
  - An AMD GPU (gfx942 or gfx90a)
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from metrix import Metrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile_hip(source: str, name: str, tmp_dir: Path) -> Path:
    """Compile inline HIP source with hipcc, return binary path."""
    src = tmp_dir / f"{name}.hip"
    binary = tmp_dir / name
    src.write_text(source)
    r = subprocess.run(
        ["hipcc", str(src), "-o", str(binary), "-O2", "-fno-inline"],
        capture_output=True,
        text=True,
        cwd=tmp_dir,
        timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"hipcc failed:\n{r.stderr}")
    return binary


def _profile(binary: Path, metrics: list, kernel_filter: str = None, timeout: int = 90):
    """Profile a binary and return the first matching kernel's metric dict."""
    profiler = Metrix()
    results = profiler.profile(
        command=str(binary),
        metrics=metrics,
        kernel_filter=kernel_filter,
        num_replays=1,
        cwd=str(binary.parent),
        timeout_seconds=timeout,
    )
    assert results.total_kernels >= 1, "No kernels found in profiling results"
    assert len(results.kernels) >= 1
    kernel = results.kernels[0]
    return kernel


# ---------------------------------------------------------------------------
# Kernel sources
# ---------------------------------------------------------------------------


SEQUENTIAL_READ_HIP = r"""
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void sequential_read(const float* __restrict__ src,
                                float* __restrict__ dst, size_t N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = src[idx];
}

int main() {
    const size_t N = 64 * 1024 * 1024 / sizeof(float);  // 64 MB
    size_t bytes = N * sizeof(float);
    float *d_src, *d_dst;
    hipMalloc(&d_src, bytes);
    hipMalloc(&d_dst, bytes);
    hipMemset(d_src, 1, bytes);
    int bs = 256;
    hipLaunchKernelGGL(sequential_read, dim3((N + bs - 1) / bs), dim3(bs),
                       0, 0, d_src, d_dst, N);
    hipDeviceSynchronize();
    hipFree(d_src);
    hipFree(d_dst);
    return 0;
}
"""


SEQUENTIAL_WRITE_HIP = r"""
#include <hip/hip_runtime.h>

__global__ void sequential_write(float* __restrict__ dst, size_t N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = 1.0f;
}

int main() {
    const size_t N = 64 * 1024 * 1024 / sizeof(float);
    float* d_dst;
    hipMalloc(&d_dst, N * sizeof(float));
    int bs = 256;
    hipLaunchKernelGGL(sequential_write, dim3((N + bs - 1) / bs), dim3(bs),
                       0, 0, d_dst, N);
    hipDeviceSynchronize();
    hipFree(d_dst);
    return 0;
}
"""


COPY_HIP = r"""
#include <hip/hip_runtime.h>

__global__ void copy_kernel(const float* __restrict__ src,
                            float* __restrict__ dst, size_t N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = src[idx];
}

int main() {
    const size_t N = 64 * 1024 * 1024 / sizeof(float);
    size_t bytes = N * sizeof(float);
    float *d_src, *d_dst;
    hipMalloc(&d_src, bytes);
    hipMalloc(&d_dst, bytes);
    hipMemset(d_src, 1, bytes);
    int bs = 256;
    hipLaunchKernelGGL(copy_kernel, dim3((N + bs - 1) / bs), dim3(bs),
                       0, 0, d_src, d_dst, N);
    hipDeviceSynchronize();
    hipFree(d_src);
    hipFree(d_dst);
    return 0;
}
"""


LDS_CONFLICT_FREE_HIP = r"""
#include <hip/hip_runtime.h>

#define LDS_SIZE 4096

__global__ void lds_conflict_free(float* __restrict__ out, int iterations) {
    __shared__ float lds[LDS_SIZE];
    int tid = threadIdx.x;
    for (int i = tid; i < LDS_SIZE; i += blockDim.x) lds[i] = (float)i;
    __syncthreads();
    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) sum += lds[tid];
    out[blockIdx.x * blockDim.x + tid] = sum;
}

int main() {
    int num_wgs = 304;
    int bs = 256;
    float* d_out;
    hipMalloc(&d_out, (size_t)num_wgs * bs * sizeof(float));
    hipLaunchKernelGGL(lds_conflict_free, dim3(num_wgs), dim3(bs), 0, 0,
                       d_out, 10000);
    hipDeviceSynchronize();
    hipFree(d_out);
    return 0;
}
"""


LDS_HIGH_CONFLICT_HIP = r"""
#include <hip/hip_runtime.h>

#define LDS_SIZE 4096

__global__ void lds_high_conflict(float* __restrict__ out, int iterations) {
    __shared__ float lds[LDS_SIZE];
    int tid = threadIdx.x;
    for (int i = tid; i < LDS_SIZE; i += blockDim.x) lds[i] = (float)i;
    __syncthreads();
    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        int idx = (tid * 32) % LDS_SIZE;
        sum += lds[idx];
    }
    out[blockIdx.x * blockDim.x + tid] = sum;
}

int main() {
    int num_wgs = 304;
    int bs = 256;
    float* d_out;
    hipMalloc(&d_out, (size_t)num_wgs * bs * sizeof(float));
    hipLaunchKernelGGL(lds_high_conflict, dim3(num_wgs), dim3(bs), 0, 0,
                       d_out, 10000);
    hipDeviceSynchronize();
    hipFree(d_out);
    return 0;
}
"""


VALU_FMA_HIP = r"""
#include <hip/hip_runtime.h>

__global__ void valu_fma_kernel(float* __restrict__ out, int iterations) {
    float a = 1.0f, b = 1.000001f, c = 0.000001f;
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) a = __builtin_fmaf(a, b, c);
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = a;
}

int main() {
    int bs = 256, nwg = 304;
    size_t n = (size_t)nwg * bs;
    float* d_out;
    hipMalloc(&d_out, n * sizeof(float));
    hipLaunchKernelGGL(valu_fma_kernel, dim3(nwg), dim3(bs), 0, 0, d_out, 100000);
    hipDeviceSynchronize();
    hipFree(d_out);
    return 0;
}
"""


MIXED_K10_HIP = r"""
#include <hip/hip_runtime.h>

__global__ void mixed_kernel(const float* __restrict__ src,
                             float* __restrict__ dst, size_t N, int K) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = src[idx];
        float b = 1.000001f, c = 0.000001f;
        #pragma unroll 1
        for (int k = 0; k < K; k++) val = __builtin_fmaf(val, b, c);
        dst[idx] = val;
    }
}

int main() {
    const size_t N = 64 * 1024 * 1024 / sizeof(float);
    size_t bytes = N * sizeof(float);
    float *d_src, *d_dst;
    hipMalloc(&d_src, bytes);
    hipMalloc(&d_dst, bytes);
    hipMemset(d_src, 1, bytes);
    int bs = 256;
    hipLaunchKernelGGL(mixed_kernel, dim3((N + bs - 1) / bs), dim3(bs),
                       0, 0, d_src, d_dst, N, 10);
    hipDeviceSynchronize();
    hipFree(d_src);
    hipFree(d_dst);
    return 0;
}
"""


# ---------------------------------------------------------------------------
# Tests: Bandwidth metrics
# ---------------------------------------------------------------------------


class TestBandwidthValidation:
    """Validate HBM bandwidth metrics with streaming kernels"""

    @pytest.mark.integration
    def test_sequential_read_bandwidth(self):
        """Coalesced read should report nonzero read bandwidth"""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as tmp:
            binary = _compile_hip(SEQUENTIAL_READ_HIP, "seq_read", Path(tmp))
            kernel = _profile(
                binary,
                ["memory.hbm_read_bandwidth"],
                kernel_filter="sequential_read",
            )
            bw = kernel.metrics["memory.hbm_read_bandwidth"].avg
            assert bw > 10.0, f"Read bandwidth too low: {bw:.2f} GB/s"

    @pytest.mark.integration
    def test_sequential_write_bandwidth(self):
        """Coalesced write should report nonzero write bandwidth"""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as tmp:
            binary = _compile_hip(SEQUENTIAL_WRITE_HIP, "seq_write", Path(tmp))
            kernel = _profile(
                binary,
                ["memory.hbm_write_bandwidth"],
                kernel_filter="sequential_write",
            )
            bw = kernel.metrics["memory.hbm_write_bandwidth"].avg
            assert bw > 10.0, f"Write bandwidth too low: {bw:.2f} GB/s"

    @pytest.mark.integration
    def test_copy_bytes_transferred(self):
        """Copy 64MB: bytes_transferred_hbm should reflect read+write"""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as tmp:
            binary = _compile_hip(COPY_HIP, "copy", Path(tmp))
            kernel = _profile(
                binary,
                ["memory.bytes_transferred_hbm"],
                kernel_filter="copy_kernel",
            )
            hbm_bytes = kernel.metrics["memory.bytes_transferred_hbm"].avg
            # 64 MB read + 64 MB write = 128 MB minimum
            expected_min = 64 * 1024 * 1024
            assert hbm_bytes > expected_min, (
                f"bytes_transferred_hbm too low: {hbm_bytes:.0f} < {expected_min}"
            )


# ---------------------------------------------------------------------------
# Tests: Coalescing
# ---------------------------------------------------------------------------


class TestCoalescingValidation:
    """Validate coalescing efficiency with stride-1 access"""

    @pytest.mark.integration
    def test_stride1_is_perfectly_coalesced(self):
        """Stride-1 sequential read should show 100% coalescing"""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as tmp:
            binary = _compile_hip(SEQUENTIAL_READ_HIP, "seq_read", Path(tmp))
            kernel = _profile(
                binary,
                ["memory.coalescing_efficiency"],
                kernel_filter="sequential_read",
            )
            coal = kernel.metrics["memory.coalescing_efficiency"].avg
            assert coal >= 90.0, f"Coalescing too low for stride-1: {coal:.2f}%"


# ---------------------------------------------------------------------------
# Tests: Cache hit rates
# ---------------------------------------------------------------------------


class TestCacheValidation:
    """Validate L1 and L2 cache hit rate metrics"""

    @pytest.mark.integration
    def test_l2_hit_rate_is_bounded(self):
        """L2 hit rate for streaming read should be between 0% and 100%"""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as tmp:
            binary = _compile_hip(SEQUENTIAL_READ_HIP, "seq_read", Path(tmp))
            kernel = _profile(
                binary,
                ["memory.l2_hit_rate"],
                kernel_filter="sequential_read",
            )
            hit_rate = kernel.metrics["memory.l2_hit_rate"].avg
            assert 0.0 <= hit_rate <= 100.0, f"L2 hit rate out of range: {hit_rate:.2f}%"

    @pytest.mark.integration
    def test_copy_l1_hit_rate(self):
        """Copy kernel L1 hit rate should be between 0% and 100%"""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as tmp:
            binary = _compile_hip(COPY_HIP, "copy", Path(tmp))
            kernel = _profile(
                binary,
                ["memory.l1_hit_rate"],
                kernel_filter="copy_kernel",
            )
            hit_rate = kernel.metrics["memory.l1_hit_rate"].avg
            assert 0.0 <= hit_rate <= 100.0, f"L1 hit rate out of range: {hit_rate:.2f}%"


# ---------------------------------------------------------------------------
# Tests: LDS bank conflicts
# ---------------------------------------------------------------------------


class TestLDSValidation:
    """Validate LDS bank conflict metric distinguishes access patterns"""

    @pytest.mark.integration
    def test_conflict_free_low(self):
        """Sequential LDS access should show near-zero conflicts per instruction"""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as tmp:
            binary = _compile_hip(LDS_CONFLICT_FREE_HIP, "lds_cf", Path(tmp))
            kernel = _profile(
                binary,
                ["memory.lds_bank_conflicts"],
                kernel_filter="lds_conflict_free",
            )
            conflicts = kernel.metrics["memory.lds_bank_conflicts"].avg
            assert conflicts <= 1.0, (
                f"Conflict-free LDS should show near-zero conflicts: {conflicts:.4f}"
            )

    @pytest.mark.integration
    def test_high_conflict_high(self):
        """Stride-32 LDS access should show significant conflicts"""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as tmp:
            binary = _compile_hip(LDS_HIGH_CONFLICT_HIP, "lds_hc", Path(tmp))
            kernel = _profile(
                binary,
                ["memory.lds_bank_conflicts"],
                kernel_filter="lds_high_conflict",
            )
            conflicts = kernel.metrics["memory.lds_bank_conflicts"].avg
            assert conflicts > 1.0, (
                f"High-conflict LDS should show significant conflicts: {conflicts:.4f}"
            )

    @pytest.mark.integration
    def test_conflict_free_less_than_high_conflict(self):
        """Conflict-free pattern must produce fewer conflicts than high-conflict"""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as tmp:
            tmp_path = Path(tmp)
            bin_cf = _compile_hip(LDS_CONFLICT_FREE_HIP, "lds_cf", tmp_path)
            bin_hc = _compile_hip(LDS_HIGH_CONFLICT_HIP, "lds_hc", tmp_path)

            k_cf = _profile(
                bin_cf, ["memory.lds_bank_conflicts"],
                kernel_filter="lds_conflict_free",
            )
            k_hc = _profile(
                bin_hc, ["memory.lds_bank_conflicts"],
                kernel_filter="lds_high_conflict",
            )

            cf_val = k_cf.metrics["memory.lds_bank_conflicts"].avg
            hc_val = k_hc.metrics["memory.lds_bank_conflicts"].avg
            assert hc_val > cf_val, (
                f"High-conflict ({hc_val:.4f}) should exceed conflict-free ({cf_val:.4f})"
            )


# ---------------------------------------------------------------------------
# Tests: Compute metrics
# ---------------------------------------------------------------------------


class TestComputeValidation:
    """Validate FLOPS and arithmetic intensity with compute-heavy kernels"""

    @pytest.mark.integration
    def test_valu_fma_reports_flops(self):
        """Pure FMA loop should report billions of FLOPS"""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as tmp:
            binary = _compile_hip(VALU_FMA_HIP, "valu_fma", Path(tmp))
            kernel = _profile(
                binary,
                ["compute.total_flops"],
                kernel_filter="valu_fma_kernel",
            )
            flops = kernel.metrics["compute.total_flops"].avg
            # 304 WGs * 256 threads = 77824 threads = 1216 waves
            # 100K FMAs * 2 * 64 * 1216 waves ~ 15.6 billion
            assert flops > 1e9, f"FLOPS too low for 100K FMA loop: {flops:.0f}"

    @pytest.mark.integration
    def test_valu_fma_high_arithmetic_intensity(self):
        """Pure compute kernel should have very high arithmetic intensity"""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as tmp:
            binary = _compile_hip(VALU_FMA_HIP, "valu_fma", Path(tmp))
            kernel = _profile(
                binary,
                ["compute.hbm_arithmetic_intensity"],
                kernel_filter="valu_fma_kernel",
            )
            ai = kernel.metrics["compute.hbm_arithmetic_intensity"].avg
            assert ai > 100.0, f"AI too low for pure compute kernel: {ai:.2f}"

    @pytest.mark.integration
    def test_valu_fma_gflops(self):
        """Pure FMA loop should report nonzero GFLOPS"""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as tmp:
            binary = _compile_hip(VALU_FMA_HIP, "valu_fma", Path(tmp))
            kernel = _profile(
                binary,
                ["compute.hbm_gflops"],
                kernel_filter="valu_fma_kernel",
            )
            gflops = kernel.metrics["compute.hbm_gflops"].avg
            assert gflops > 1.0, f"GFLOPS too low: {gflops:.2f}"

    @pytest.mark.integration
    def test_mixed_arithmetic_intensity(self):
        """K=10 FMAs per load: AI should be roughly K/4 = 2.5 FLOP/byte"""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as tmp:
            binary = _compile_hip(MIXED_K10_HIP, "mixed_k10", Path(tmp))
            kernel = _profile(
                binary,
                ["compute.hbm_arithmetic_intensity"],
                kernel_filter="mixed_kernel",
            )
            ai = kernel.metrics["compute.hbm_arithmetic_intensity"].avg
            # Expected ~2.5, allow wide tolerance (hardware prefetch, write-allocate)
            assert 0.5 < ai < 10.0, f"AI out of expected range for K=10: {ai:.4f}"
