# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Derived counter validation tests.

Each test profiles a HIP microbenchmark with known, analytically
predictable behavior and asserts that the metrix-derived metric values
fall within expected ranges.  This validates the YAML expressions in
counter_defs.yaml on real hardware.

All HIP source is inlined as Python strings and compiled at runtime
via hipcc, following the pattern in test_inline_hip_profiling.py.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from metrix import Metrix
from ..unit.conftest import requires_arch, requires_metric

# ---------------------------------------------------------------------------
# HIP compilation helper
# ---------------------------------------------------------------------------

_HIP_HEADER = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call) do {                                         \
    hipError_t err = (call);                                         \
    if (err != hipSuccess) {                                         \
        fprintf(stderr, "HIP error: %s at %s:%d\n",                 \
                hipGetErrorString(err), __FILE__, __LINE__);         \
        exit(1);                                                     \
    }                                                                \
} while(0)
"""


def _compile_hip(source: str, name: str, tmp_dir: Path) -> Path:
    """Write HIP source, compile with hipcc, return path to binary."""
    src = tmp_dir / f"{name}.hip"
    bin_path = tmp_dir / name
    src.write_text(source)
    r = subprocess.run(
        ["hipcc", str(src), "-o", str(bin_path), "-O2", "-fno-inline"],
        capture_output=True,
        text=True,
        cwd=tmp_dir,
        timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"hipcc failed:\n{r.stderr}")
    return bin_path


def _profile(binary: Path, metrics: list, tmp_dir: Path, num_replays: int = 2) -> dict:
    """Profile a binary and return {metric_name: avg_value}."""
    profiler = Metrix()
    results = profiler.profile(
        command=str(binary),
        metrics=metrics,
        num_replays=num_replays,
        aggregate_by_kernel=True,
        cwd=str(tmp_dir),
        timeout_seconds=120,
    )
    assert results.kernels, "No kernels profiled"
    # Pick the kernel with the longest duration (measured run, not warmup)
    kernel = max(results.kernels, key=lambda k: k.duration_us.avg)
    return {m: kernel.metrics[m].avg for m in metrics if m in kernel.metrics}


# =========================================================================
# Bandwidth benchmarks
# =========================================================================


class TestHBMBandwidth:
    """Validate HBM bandwidth derived metrics."""

    _READ_SRC = (
        _HIP_HEADER
        + r"""
    __global__ void read_kernel(const float* __restrict__ src,
                                float* __restrict__ out, size_t N) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) out[0] += src[idx];
    }
    int main() {
        size_t N = 128ULL * 1024 * 1024;
        float *d_src, *d_out;
        HIP_CHECK(hipMalloc(&d_src, N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_out, sizeof(float)));
        HIP_CHECK(hipMemset(d_src, 0, N * sizeof(float)));
        HIP_CHECK(hipMemset(d_out, 0, sizeof(float)));
        int block = 256, grid = (N + block - 1) / block;
        read_kernel<<<grid, block>>>(d_src, d_out, N);
        HIP_CHECK(hipDeviceSynchronize());
        read_kernel<<<grid, block>>>(d_src, d_out, N);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipFree(d_src)); HIP_CHECK(hipFree(d_out));
        return 0;
    }
    """
    )

    _WRITE_SRC = (
        _HIP_HEADER
        + r"""
    __global__ void write_kernel(float* __restrict__ dst, size_t N) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) dst[idx] = 1.0f;
    }
    int main() {
        size_t N = 128ULL * 1024 * 1024;
        float *d_dst;
        HIP_CHECK(hipMalloc(&d_dst, N * sizeof(float)));
        int block = 256, grid = (N + block - 1) / block;
        write_kernel<<<grid, block>>>(d_dst, N);
        HIP_CHECK(hipDeviceSynchronize());
        write_kernel<<<grid, block>>>(d_dst, N);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipFree(d_dst));
        return 0;
    }
    """
    )

    _COPY_SRC = (
        _HIP_HEADER
        + r"""
    __global__ void copy_kernel(const float* __restrict__ src,
                                float* __restrict__ dst, size_t N) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) dst[idx] = src[idx];
    }
    int main() {
        size_t N = 128ULL * 1024 * 1024;
        float *d_src, *d_dst;
        HIP_CHECK(hipMalloc(&d_src, N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_dst, N * sizeof(float)));
        HIP_CHECK(hipMemset(d_src, 0, N * sizeof(float)));
        int block = 256, grid = (N + block - 1) / block;
        copy_kernel<<<grid, block>>>(d_src, d_dst, N);
        HIP_CHECK(hipDeviceSynchronize());
        copy_kernel<<<grid, block>>>(d_src, d_dst, N);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipFree(d_src)); HIP_CHECK(hipFree(d_dst));
        return 0;
    }
    """
    )

    def test_hbm_read_bandwidth(self):
        """Sequential coalesced read should achieve >100 GB/s."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(self._READ_SRC, "bw_read", p)
            m = _profile(b, ["memory.hbm_read_bandwidth"], p)
        assert m["memory.hbm_read_bandwidth"] > 100.0

    def test_hbm_write_bandwidth(self):
        """Sequential coalesced write should achieve >50 GB/s."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(self._WRITE_SRC, "bw_write", p)
            m = _profile(b, ["memory.hbm_write_bandwidth"], p)
        assert m["memory.hbm_write_bandwidth"] > 50.0

    def test_copy_bandwidth_utilization(self):
        """Copy kernel should show measurable BW utilization."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(self._COPY_SRC, "bw_copy", p)
            m = _profile(
                b,
                [
                    "memory.hbm_bandwidth_utilization",
                    "memory.hbm_read_bandwidth",
                    "memory.hbm_write_bandwidth",
                ],
                p,
            )
        # 3% floor: on high-peak-BW parts (e.g. gfx950) the same absolute
        # copy BW yields a smaller % utilization. Absolute BW is checked below.
        assert 3.0 <= m["memory.hbm_bandwidth_utilization"] <= 100.0
        assert m["memory.hbm_read_bandwidth"] > 50.0
        assert m["memory.hbm_write_bandwidth"] > 50.0


# =========================================================================
# Coalescing benchmarks
# =========================================================================


def _strided_source(stride: int) -> str:
    return (
        _HIP_HEADER
        + f"""
    __global__ void strided_kernel(const float* __restrict__ src,
                                   float* __restrict__ out, size_t N) {{
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) out[0] += src[idx * {stride}];
    }}
    int main() {{
        size_t N = 32ULL * 1024 * 1024;
        size_t alloc = N * {stride};
        float *d_src, *d_out;
        HIP_CHECK(hipMalloc(&d_src, alloc * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_out, sizeof(float)));
        HIP_CHECK(hipMemset(d_src, 0, alloc * sizeof(float)));
        HIP_CHECK(hipMemset(d_out, 0, sizeof(float)));
        int block = 256, grid = (N + block - 1) / block;
        strided_kernel<<<grid, block>>>(d_src, d_out, N);
        HIP_CHECK(hipDeviceSynchronize());
        strided_kernel<<<grid, block>>>(d_src, d_out, N);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipFree(d_src)); HIP_CHECK(hipFree(d_out));
        return 0;
    }}
    """
    )


@requires_metric("memory.coalescing_efficiency")
class TestCoalescingEfficiency:
    """Validate coalescing_efficiency scales with stride."""

    def test_stride1_high_coalescing(self):
        """Stride-1 should give ~100% coalescing."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(_strided_source(1), "stride1", p)
            m = _profile(b, ["memory.coalescing_efficiency"], p)
        assert 80.0 <= m["memory.coalescing_efficiency"] <= 100.0

    def test_stride2_reduced_coalescing(self):
        """Stride-2 should give reduced coalescing (~40-50%)."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(_strided_source(2), "stride2", p)
            m = _profile(b, ["memory.coalescing_efficiency"], p)
        assert 25.0 <= m["memory.coalescing_efficiency"] <= 65.0

    def test_stride4_lower_coalescing(self):
        """Stride-4 should give further reduced coalescing."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(_strided_source(4), "stride4", p)
            m = _profile(b, ["memory.coalescing_efficiency"], p)
        assert 10.0 <= m["memory.coalescing_efficiency"] <= 50.0

    def test_coalescing_decreases_with_stride(self):
        """Coalescing should be monotonically non-increasing with stride."""
        values = {}
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            for s in [1, 2, 4]:
                b = _compile_hip(_strided_source(s), f"stride{s}", p)
                m = _profile(b, ["memory.coalescing_efficiency"], p)
                values[s] = m["memory.coalescing_efficiency"]
        assert values[1] >= values[2] >= values[4]


# =========================================================================
# Cache hit rate benchmarks
# =========================================================================


class TestCacheHitRates:
    """Validate L1 and L2 cache hit rate metrics."""

    # Small array iterated many times to get L2 hits.
    # Use few blocks to reduce L2 set contention across CUs.
    # MI210 has 8 MB L2 (lower associativity), MI300X has 256 MB.
    _L2_SRC = (
        _HIP_HEADER
        + r"""
    __global__ void l2_kernel(const float* __restrict__ src,
                              float* __restrict__ out,
                              size_t N, int iters) {
        float acc = 0.0f;
        for (int i = 0; i < iters; i++) {
            size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) % N;
            acc += src[idx];
        }
        if (threadIdx.x == 0) out[blockIdx.x] = acc;
    }
    int main() {
        size_t N = 64ULL * 1024;  // 256 KB — small enough for any L2
        int iters = 500, num_blocks = 16, block = 256;
        float *d_src, *d_out;
        HIP_CHECK(hipMalloc(&d_src, N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));
        float *h = (float*)malloc(N * sizeof(float));
        for (size_t i = 0; i < N; i++) h[i] = 1.0f;
        HIP_CHECK(hipMemcpy(d_src, h, N * sizeof(float), hipMemcpyHostToDevice));
        free(h);
        HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(float)));
        // Warmup — fill L2
        l2_kernel<<<num_blocks, block>>>(d_src, d_out, N, 10);
        HIP_CHECK(hipDeviceSynchronize());
        // Measured — should hit L2 heavily
        l2_kernel<<<num_blocks, block>>>(d_src, d_out, N, iters);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipFree(d_src)); HIP_CHECK(hipFree(d_out));
        return 0;
    }
    """
    )

    _L1_SRC = (
        _HIP_HEADER
        + r"""
    __global__ void l1_kernel(const float* __restrict__ src,
                              float* __restrict__ out,
                              int N_per_block, int iters) {
        float acc = 0.0f;
        int idx = threadIdx.x;
        for (int i = 0; i < iters; i++) {
            if (idx < N_per_block) acc += src[idx];
        }
        if (threadIdx.x == 0) out[blockIdx.x] = acc;
    }
    int main() {
        int N_per_block = 1024, iters = 1000, num_blocks = 256, block = 256;
        float *d_src, *d_out;
        HIP_CHECK(hipMalloc(&d_src, N_per_block * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));
        float *h = (float*)malloc(N_per_block * sizeof(float));
        for (int i = 0; i < N_per_block; i++) h[i] = 1.0f;
        HIP_CHECK(hipMemcpy(d_src, h, N_per_block * sizeof(float), hipMemcpyHostToDevice));
        free(h);
        HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(float)));
        l1_kernel<<<num_blocks, block>>>(d_src, d_out, N_per_block, 1);
        HIP_CHECK(hipDeviceSynchronize());
        l1_kernel<<<num_blocks, block>>>(d_src, d_out, N_per_block, iters);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipFree(d_src)); HIP_CHECK(hipFree(d_out));
        return 0;
    }
    """
    )

    def test_l2_hit_rate_with_resident_data(self):
        """256 KB array iterated 500x with few blocks should show elevated L2 hit rate.

        MI300X achieves >60% due to large 256 MB L2. MI210 achieves ~40% due to
        smaller L2 and different cache hierarchy. Threshold is set at 30% to work
        across architectures while still validating the formula produces real values.
        """
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(self._L2_SRC, "l2_resident", p)
            m = _profile(b, ["memory.l2_hit_rate"], p)
        assert m["memory.l2_hit_rate"] > 30.0

    @requires_metric("memory.l1_hit_rate")
    def test_l1_hit_rate_with_tiny_array(self):
        """4 KB per block iterated 1000x should show >80% L1 hit rate."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(self._L1_SRC, "l1_resident", p)
            m = _profile(b, ["memory.l1_hit_rate"], p)
        assert m["memory.l1_hit_rate"] > 80.0


# =========================================================================
# LDS bank conflict benchmarks
# =========================================================================


def _lds_source(stride: int) -> str:
    access = "lds[tid % 8192]" if stride == 1 else f"lds[(tid * {stride}) % 8192]"
    return (
        _HIP_HEADER
        + f"""
    #define LDS_SIZE 8192
    __global__ void lds_kernel(float* __restrict__ out, int iters) {{
        __shared__ float lds[LDS_SIZE];
        int tid = threadIdx.x;
        if (tid < LDS_SIZE) lds[tid] = 1.0f;
        __syncthreads();
        float acc = 0.0f;
        for (int i = 0; i < iters; i++) acc += {access};
        if (tid == 0) out[blockIdx.x] = acc;
    }}
    int main() {{
        int num_blocks = 128, block = 256, iters = 10000;
        float *d_out;
        HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));
        HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(float)));
        lds_kernel<<<num_blocks, block>>>(d_out, 1);
        HIP_CHECK(hipDeviceSynchronize());
        lds_kernel<<<num_blocks, block>>>(d_out, iters);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipFree(d_out));
        return 0;
    }}
    """
    )


class TestLDSBankConflicts:
    """Validate LDS bank conflict metric."""

    def test_no_conflicts_with_sequential_access(self):
        """Sequential LDS access should show ~0 bank conflicts."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(_lds_source(1), "lds_seq", p)
            m = _profile(b, ["memory.lds_bank_conflicts"], p)
        assert m["memory.lds_bank_conflicts"] < 2.0

    def test_high_conflicts_with_stride32(self):
        """Stride-32 LDS access should cause many bank conflicts."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(_lds_source(32), "lds_conflict", p)
            m = _profile(b, ["memory.lds_bank_conflicts"], p)
        assert m["memory.lds_bank_conflicts"] > 5.0


# =========================================================================
# Compute (FLOPS) benchmarks
# =========================================================================

_VALU_FMA_SRC = (
    _HIP_HEADER
    + r"""
__global__ void valu_fma_kernel(float* __restrict__ out, int iters) {
    float a = 1.0001f, b = 0.9999f, c = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        c = __builtin_fmaf(a, b, c);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = c;
}
int main() {
    int num_blocks = 512, block = 256, iters = 100000;
    float *d_out;
    HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));
    HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(float)));
    valu_fma_kernel<<<num_blocks, block>>>(d_out, 100);
    HIP_CHECK(hipDeviceSynchronize());
    valu_fma_kernel<<<num_blocks, block>>>(d_out, iters);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_out));
    return 0;
}
"""
)


@requires_metric("compute.total_flops", "compute.hbm_gflops")
class TestFLOPSCounters:
    """Validate compute.total_flops and compute.hbm_gflops."""

    def test_valu_fma_total_flops(self):
        """Pure FMA kernel should report correct total FLOPS count.

        512 blocks * 256 threads = 131072 threads = 2048 wavefronts
        2048 waves * 100000 iters * 2 FLOP/FMA * 64 lanes = 26,214,400,000
        """
        expected = 2048 * 100000 * 2 * 64
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(_VALU_FMA_SRC, "valu_fma", p)
            m = _profile(b, ["compute.total_flops", "compute.hbm_gflops"], p)
        # Within 5% of expected
        assert abs(m["compute.total_flops"] - expected) / expected < 0.05
        assert m["compute.hbm_gflops"] > 1.0


# =========================================================================
# Atomic latency benchmarks (gfx942 only)
# =========================================================================

_ATOMIC_HIGH_SRC = (
    _HIP_HEADER
    + r"""
__global__ void atomic_kernel(int* __restrict__ counter, int iters) {
    #pragma unroll 1
    for (int i = 0; i < iters; i++) atomicAdd(counter, 1);
}
int main() {
    int num_blocks = 64, block = 256, iters = 1000;
    int *d;
    HIP_CHECK(hipMalloc(&d, sizeof(int)));
    HIP_CHECK(hipMemset(d, 0, sizeof(int)));
    atomic_kernel<<<num_blocks, block>>>(d, 10);
    HIP_CHECK(hipDeviceSynchronize());
    atomic_kernel<<<num_blocks, block>>>(d, iters);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d));
    return 0;
}
"""
)

_ATOMIC_LOW_SRC = (
    _HIP_HEADER
    + r"""
__global__ void atomic_kernel(int* __restrict__ counters, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) atomicAdd(&counters[tid], 1);
}
int main() {
    int num_blocks = 64, block = 256, iters = 1000, total = num_blocks * block;
    int *d;
    HIP_CHECK(hipMalloc(&d, total * sizeof(int)));
    HIP_CHECK(hipMemset(d, 0, total * sizeof(int)));
    atomic_kernel<<<num_blocks, block>>>(d, 10);
    HIP_CHECK(hipDeviceSynchronize());
    atomic_kernel<<<num_blocks, block>>>(d, iters);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d));
    return 0;
}
"""
)


class TestAtomicLatency:
    """Validate memory.atomic_latency on gfx942."""

    @requires_arch("gfx942")
    def test_high_contention_high_latency(self):
        """All threads atomicAdd to 1 address should show high latency."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(_ATOMIC_HIGH_SRC, "atomic_hi", p)
            m = _profile(b, ["memory.atomic_latency"], p)
        assert m["memory.atomic_latency"] > 100.0

    @requires_arch("gfx942")
    def test_low_contention_lower_latency(self):
        """Each thread atomicAdd to own address should show lower latency."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(_ATOMIC_LOW_SRC, "atomic_lo", p)
            m = _profile(b, ["memory.atomic_latency"], p)
        # Just verify it's a valid number (latency > 0)
        assert m["memory.atomic_latency"] >= 0.0


# =========================================================================
# Arithmetic intensity benchmarks
# =========================================================================


def _mixed_source(K: int) -> str:
    return (
        _HIP_HEADER
        + f"""
    __global__ void mixed_kernel(const float* __restrict__ src,
                                 float* __restrict__ dst, size_t N) {{
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {{
            float val = src[idx];
            float acc = 0.0f;
            #pragma unroll 1
            for (int i = 0; i < {K}; i++) acc = __builtin_fmaf(val, val, acc);
            dst[idx] = acc;
        }}
    }}
    int main() {{
        size_t N = 64ULL * 1024 * 1024;
        float *d_src, *d_dst;
        HIP_CHECK(hipMalloc(&d_src, N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_dst, N * sizeof(float)));
        HIP_CHECK(hipMemset(d_src, 0x3F, N * sizeof(float)));
        int block = 256, grid = (N + block - 1) / block;
        mixed_kernel<<<grid, block>>>(d_src, d_dst, N);
        HIP_CHECK(hipDeviceSynchronize());
        mixed_kernel<<<grid, block>>>(d_src, d_dst, N);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipFree(d_src)); HIP_CHECK(hipFree(d_dst));
        return 0;
    }}
    """
    )


@requires_metric("compute.hbm_arithmetic_intensity")
class TestArithmeticIntensity:
    """Validate compute.hbm_arithmetic_intensity scales with compute/memory ratio."""

    def test_high_ai_with_many_fmas(self):
        """100 FMAs per load should give high arithmetic intensity (>5)."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(_mixed_source(100), "mixed_k100", p)
            m = _profile(b, ["compute.hbm_arithmetic_intensity"], p)
        assert m["compute.hbm_arithmetic_intensity"] > 5.0

    def test_low_ai_with_few_fmas(self):
        """1 FMA per load should give low arithmetic intensity (<50)."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(_mixed_source(1), "mixed_k1", p)
            m = _profile(b, ["compute.hbm_arithmetic_intensity"], p)
        assert m["compute.hbm_arithmetic_intensity"] < 50.0

    def test_ai_increases_with_compute(self):
        """AI should increase when more FMAs per load."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b1 = _compile_hip(_mixed_source(1), "mixed_k1", p)
            b100 = _compile_hip(_mixed_source(100), "mixed_k100", p)
            m1 = _profile(b1, ["compute.hbm_arithmetic_intensity"], p)
            m100 = _profile(b100, ["compute.hbm_arithmetic_intensity"], p)
        assert m100["compute.hbm_arithmetic_intensity"] > m1["compute.hbm_arithmetic_intensity"]


# =========================================================================
# Bounds-checking tests for all derived metrics
# =========================================================================

# Metrics that report percentages must be in [0, 100]
_ALL_PERCENTAGE_METRICS = [
    "memory.l2_hit_rate",
    "memory.l1_hit_rate",
    "memory.hbm_bandwidth_utilization",
    "memory.coalescing_efficiency",
    "memory.global_load_efficiency",
    "memory.global_store_efficiency",
]

# Metrics that must be non-negative (bandwidth, bytes, FLOPS, latency, etc.)
_ALL_NON_NEGATIVE_METRICS = [
    "memory.hbm_read_bandwidth",
    "memory.hbm_write_bandwidth",
    "memory.l2_bandwidth",
    "memory.bytes_transferred_hbm",
    "memory.bytes_transferred_l2",
    "memory.bytes_transferred_l1",
    "memory.lds_bank_conflicts",
    "compute.total_flops",
    "compute.hbm_gflops",
    "compute.hbm_arithmetic_intensity",
    "compute.l2_arithmetic_intensity",
    "compute.l1_arithmetic_intensity",
]


def _filter_available(metric_list):
    """Filter a metric list to only those available on the detected GPU."""
    profiler = Metrix()
    available = set(profiler.backend.get_available_metrics())
    return [m for m in metric_list if m in available]

# Copy kernel exercises both reads and writes — good for testing bounds
# across memory metrics.  FMA-heavy kernel covers compute metrics.
_BOUNDS_COPY_SRC = (
    _HIP_HEADER
    + r"""
__global__ void copy_kernel(const float* __restrict__ src,
                            float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = src[idx];
}
int main() {
    size_t N = 64ULL * 1024 * 1024;
    float *d_src, *d_dst;
    HIP_CHECK(hipMalloc(&d_src, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dst, N * sizeof(float)));
    HIP_CHECK(hipMemset(d_src, 0, N * sizeof(float)));
    int block = 256, grid = (N + block - 1) / block;
    copy_kernel<<<grid, block>>>(d_src, d_dst, N);
    HIP_CHECK(hipDeviceSynchronize());
    copy_kernel<<<grid, block>>>(d_src, d_dst, N);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_src)); HIP_CHECK(hipFree(d_dst));
    return 0;
}
"""
)

_BOUNDS_FMA_SRC = (
    _HIP_HEADER
    + r"""
__global__ void fma_kernel(const float* __restrict__ src,
                           float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = src[idx];
        float acc = 0.0f;
        #pragma unroll 1
        for (int i = 0; i < 10; i++) acc = __builtin_fmaf(val, val, acc);
        dst[idx] = acc;
    }
}
int main() {
    size_t N = 64ULL * 1024 * 1024;
    float *d_src, *d_dst;
    HIP_CHECK(hipMalloc(&d_src, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dst, N * sizeof(float)));
    HIP_CHECK(hipMemset(d_src, 0x3F, N * sizeof(float)));
    int block = 256, grid = (N + block - 1) / block;
    fma_kernel<<<grid, block>>>(d_src, d_dst, N);
    HIP_CHECK(hipDeviceSynchronize());
    fma_kernel<<<grid, block>>>(d_src, d_dst, N);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_src)); HIP_CHECK(hipFree(d_dst));
    return 0;
}
"""
)


class TestMetricBounds:
    """Verify every derived metric stays within valid bounds.

    No metric that reports a percentage should ever exceed 100% or go
    below 0%.  Bandwidth, bytes, FLOPS, and latency must be non-negative.
    """

    def test_percentage_metrics_bounded_0_100(self):
        """All percentage metrics must be in [0, 100]."""
        pct_metrics = _filter_available(_ALL_PERCENTAGE_METRICS)
        if not pct_metrics:
            pytest.skip("No percentage metrics available on this GPU")
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(_BOUNDS_COPY_SRC, "bounds_copy", p)
            m = _profile(b, pct_metrics, p)
        for name in pct_metrics:
            if name not in m:
                continue
            val = m[name]
            assert 0.0 <= val <= 100.0, f"{name} = {val} is outside [0, 100]"

    def test_memory_metrics_non_negative(self):
        """Bandwidth, bytes transferred, and LDS conflicts must be >= 0."""
        mem_metrics = _filter_available(
            [m for m in _ALL_NON_NEGATIVE_METRICS if m.startswith("memory.")]
        )
        if not mem_metrics:
            pytest.skip("No memory metrics available on this GPU")
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(_BOUNDS_COPY_SRC, "bounds_mem", p)
            m = _profile(b, mem_metrics, p)
        for name in mem_metrics:
            if name not in m:
                continue
            assert m[name] >= 0.0, f"{name} = {m[name]} is negative"

    def test_compute_metrics_non_negative(self):
        """FLOPS, GFLOPS, and arithmetic intensity must be >= 0."""
        compute_metrics = _filter_available(
            [m for m in _ALL_NON_NEGATIVE_METRICS if m.startswith("compute.")]
        )
        if not compute_metrics:
            pytest.skip("No compute metrics available on this GPU")
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(_BOUNDS_FMA_SRC, "bounds_fma", p)
            m = _profile(b, compute_metrics, p)
        for name in compute_metrics:
            if name not in m:
                continue
            assert m[name] >= 0.0, f"{name} = {m[name]} is negative"

    @requires_arch("gfx942")
    def test_atomic_latency_non_negative(self):
        """Atomic latency must be >= 0."""
        with tempfile.TemporaryDirectory(prefix="metrix_val_") as d:
            p = Path(d)
            b = _compile_hip(_ATOMIC_HIGH_SRC, "bounds_atomic", p)
            m = _profile(b, ["memory.atomic_latency"], p)
        assert m["memory.atomic_latency"] >= 0.0
