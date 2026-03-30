#!/usr/bin/env python3
"""
validate.py — Derived counter validation for metrix.

Compiles inline HIP microbenchmarks, profiles them with the metrix API,
and checks that derived metric values match analytically expected results.

Validates the YAML expressions in counter_defs.yaml against kernels
with known, analytically predictable behavior.

Usage:
    python validate.py                          # run all benchmarks
    python validate.py --benchmark bw_copy      # run one benchmark
    python validate.py --arch gfx942            # explicit arch
    python validate.py --list                   # list available benchmarks
    python validate.py --report report.md       # save markdown report
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add metrix to path if running from source tree
BENCH_DIR = Path(__file__).resolve().parent
METRIX_SRC = BENCH_DIR.parent / "src"
if METRIX_SRC.is_dir():
    sys.path.insert(0, str(METRIX_SRC))


# ---------------------------------------------------------------------------
# HIP compilation helper (matches test_inline_hip_profiling.py pattern)
# ---------------------------------------------------------------------------

def _compile_hip(source: str, name: str, build_dir: Path) -> Path:
    """Compile inline HIP source to a binary."""
    build_dir.mkdir(parents=True, exist_ok=True)
    src_path = build_dir / f"{name}.hip"
    bin_path = build_dir / name
    src_path.write_text(source)
    result = subprocess.run(
        ["hipcc", str(src_path), "-o", str(bin_path), "-O2", "-fno-inline"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"hipcc failed for {name}:\n{result.stderr}")
    return bin_path


# ---------------------------------------------------------------------------
# Inline HIP kernel sources
# ---------------------------------------------------------------------------

COMMON_HEADER = r"""
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

# --- Benchmark 1: Sequential coalesced read from HBM ---
BW_SEQUENTIAL_READ = COMMON_HEADER + r"""
__global__ void sequential_read_kernel(const float* __restrict__ src,
                                       float* __restrict__ out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[0] += src[idx];
    }
}

int main() {
    size_t N = 128ULL * 1024 * 1024;  // 128M floats = 512 MB
    float *d_src, *d_out;
    HIP_CHECK(hipMalloc(&d_src, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));
    HIP_CHECK(hipMemset(d_src, 0, N * sizeof(float)));
    HIP_CHECK(hipMemset(d_out, 0, sizeof(float)));

    int block = 256, grid = (N + block - 1) / block;
    // Warmup
    sequential_read_kernel<<<grid, block>>>(d_src, d_out, N);
    HIP_CHECK(hipDeviceSynchronize());
    // Measured
    sequential_read_kernel<<<grid, block>>>(d_src, d_out, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
"""

# --- Benchmark 2: Sequential coalesced write to HBM ---
BW_SEQUENTIAL_WRITE = COMMON_HEADER + r"""
__global__ void sequential_write_kernel(float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = 1.0f;
    }
}

int main() {
    size_t N = 128ULL * 1024 * 1024;
    float *d_dst;
    HIP_CHECK(hipMalloc(&d_dst, N * sizeof(float)));
    int block = 256, grid = (N + block - 1) / block;

    sequential_write_kernel<<<grid, block>>>(d_dst, N);
    HIP_CHECK(hipDeviceSynchronize());
    sequential_write_kernel<<<grid, block>>>(d_dst, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_dst));
    return 0;
}
"""

# --- Benchmark 3: Copy kernel (1:1 read/write) ---
BW_COPY = COMMON_HEADER + r"""
__global__ void copy_kernel(const float* __restrict__ src,
                            float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = src[idx];
    }
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

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
    return 0;
}
"""

# --- Benchmark 4: Strided access (parameterized via #define) ---
def _strided_source(stride: int) -> str:
    return COMMON_HEADER + f"""
__global__ void strided_read_kernel(const float* __restrict__ src,
                                    float* __restrict__ out, size_t N) {{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {{
        out[0] += src[idx * {stride}];
    }}
}}

int main() {{
    size_t N = 32ULL * 1024 * 1024;  // 32M threads
    size_t alloc = N * {stride};
    float *d_src, *d_out;
    HIP_CHECK(hipMalloc(&d_src, alloc * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));
    HIP_CHECK(hipMemset(d_src, 0, alloc * sizeof(float)));
    HIP_CHECK(hipMemset(d_out, 0, sizeof(float)));

    int block = 256, grid = (N + block - 1) / block;
    strided_read_kernel<<<grid, block>>>(d_src, d_out, N);
    HIP_CHECK(hipDeviceSynchronize());
    strided_read_kernel<<<grid, block>>>(d_src, d_out, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_out));
    return 0;
}}
"""

# --- Benchmark 5: L2-resident reads ---
CACHE_L2_RESIDENT = COMMON_HEADER + r"""
__global__ void l2_resident_kernel(const float* __restrict__ src,
                                   float* __restrict__ out,
                                   size_t N, int iters) {
    float acc = 0.0f;
    for (int i = 0; i < iters; i++) {
        // All threads read same array repeatedly — but wrap around
        size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) % N;
        acc += src[idx];
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

int main() {
    // 2M floats = 8 MB — fits well inside MI300X 256 MB L2
    size_t N = 2ULL * 1024 * 1024;
    int iters = 100;
    int num_blocks = 256;
    int block = 256;

    float *d_src, *d_out;
    HIP_CHECK(hipMalloc(&d_src, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));

    // Init with nonzero
    float *h = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) h[i] = 1.0f;
    HIP_CHECK(hipMemcpy(d_src, h, N * sizeof(float), hipMemcpyHostToDevice));
    free(h);
    HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(float)));

    // Warmup to populate L2
    l2_resident_kernel<<<num_blocks, block>>>(d_src, d_out, N, 1);
    HIP_CHECK(hipDeviceSynchronize());
    // Measured run
    l2_resident_kernel<<<num_blocks, block>>>(d_src, d_out, N, iters);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
"""

# --- Benchmark 6: L1-resident reads ---
CACHE_L1_RESIDENT = COMMON_HEADER + r"""
__global__ void l1_resident_kernel(const float* __restrict__ src,
                                   float* __restrict__ out,
                                   int N_per_block, int iters) {
    float acc = 0.0f;
    int idx = threadIdx.x;
    for (int i = 0; i < iters; i++) {
        if (idx < N_per_block) {
            acc += src[idx];
        }
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

int main() {
    int N_per_block = 1024;  // 4 KB — fits in L1
    int iters = 1000;
    int num_blocks = 256;
    int block = 256;

    float *d_src, *d_out;
    HIP_CHECK(hipMalloc(&d_src, N_per_block * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));

    float *h = (float*)malloc(N_per_block * sizeof(float));
    for (int i = 0; i < N_per_block; i++) h[i] = 1.0f;
    HIP_CHECK(hipMemcpy(d_src, h, N_per_block * sizeof(float), hipMemcpyHostToDevice));
    free(h);
    HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(float)));

    l1_resident_kernel<<<num_blocks, block>>>(d_src, d_out, N_per_block, 1);
    HIP_CHECK(hipDeviceSynchronize());
    l1_resident_kernel<<<num_blocks, block>>>(d_src, d_out, N_per_block, iters);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
"""

# --- Benchmark 7: LDS bank conflicts ---
def _lds_source(conflict_mode: int) -> str:
    if conflict_mode == 0:
        # Conflict-free: consecutive threads access consecutive addresses
        access_pattern = "lds[tid % 8192]"
    else:
        # High conflict: stride-32 so all threads in wavefront hit same bank
        access_pattern = "lds[(tid * 32) % 8192]"
    return COMMON_HEADER + f"""
#define LDS_SIZE 8192

__global__ void lds_kernel(float* __restrict__ out, int iters) {{
    __shared__ float lds[LDS_SIZE];
    int tid = threadIdx.x;
    if (tid < LDS_SIZE) lds[tid] = 1.0f;
    __syncthreads();

    float acc = 0.0f;
    for (int i = 0; i < iters; i++) {{
        acc += {access_pattern};
    }}
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

# --- Benchmark 8: Pure VALU FMA compute ---
COMPUTE_VALU_FMA = COMMON_HEADER + r"""
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

# --- Benchmark 9: Atomic contention ---
def _atomic_source(high_contention: bool) -> str:
    if high_contention:
        kernel = """
__global__ void atomic_kernel(int* __restrict__ counter, int iters) {
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        atomicAdd(counter, 1);
    }
}"""
    else:
        kernel = """
__global__ void atomic_kernel(int* __restrict__ counters, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        atomicAdd(&counters[tid], 1);
    }
}"""
    return COMMON_HEADER + kernel + """

int main() {
    int num_blocks = 64, block = 256, iters = 1000;
    int total = num_blocks * block;
    int *d_counters;
    HIP_CHECK(hipMalloc(&d_counters, total * sizeof(int)));
    HIP_CHECK(hipMemset(d_counters, 0, total * sizeof(int)));

    atomic_kernel<<<num_blocks, block>>>(d_counters, 10);
    HIP_CHECK(hipDeviceSynchronize());
    atomic_kernel<<<num_blocks, block>>>(d_counters, iters);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_counters));
    return 0;
}
"""

# --- Benchmark 10: Mixed compute + memory (tunable AI) ---
def _mixed_source(K: int) -> str:
    return COMMON_HEADER + f"""
__global__ void mixed_kernel(const float* __restrict__ src,
                             float* __restrict__ dst, size_t N) {{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {{
        float val = src[idx];
        float acc = 0.0f;
        #pragma unroll 1
        for (int i = 0; i < {K}; i++) {{
            acc = __builtin_fmaf(val, val, acc);
        }}
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

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
    return 0;
}}
"""


# ---------------------------------------------------------------------------
# Benchmark definitions: source, metrics, expectations
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSpec:
    name: str
    description: str
    source: str                     # inline HIP source
    metrics: List[str]              # derived metrics to collect
    expected: Dict[str, dict]       # metric -> expectation
    arch_filter: List[str] = field(default_factory=list)  # empty = all archs


BENCHMARKS: List[BenchmarkSpec] = [
    BenchmarkSpec(
        name="bw_sequential_read",
        description="Coalesced stride-1 read from HBM (512 MB, array >> L2)",
        source=BW_SEQUENTIAL_READ,
        metrics=[
            "memory.coalescing_efficiency",
            "memory.hbm_read_bandwidth",
            "memory.bytes_transferred_hbm",
        ],
        expected={
            "memory.coalescing_efficiency":  {"range": (80.0, 100.0)},
            "memory.hbm_read_bandwidth":     {"range": (100.0, None)},
        },
    ),
    BenchmarkSpec(
        name="bw_sequential_write",
        description="Coalesced stride-1 write to HBM (512 MB)",
        source=BW_SEQUENTIAL_WRITE,
        metrics=[
            "memory.hbm_write_bandwidth",
        ],
        expected={
            "memory.hbm_write_bandwidth": {"range": (50.0, None)},
        },
    ),
    BenchmarkSpec(
        name="bw_copy",
        description="Copy dst[i]=src[i], both 512 MB — validates combined BW",
        source=BW_COPY,
        metrics=[
            "memory.hbm_bandwidth_utilization",
            "memory.hbm_read_bandwidth",
            "memory.hbm_write_bandwidth",
            "memory.bytes_transferred_hbm",
        ],
        expected={
            "memory.hbm_bandwidth_utilization": {"range": (5.0, 100.0)},
            "memory.hbm_read_bandwidth":        {"range": (50.0, None)},
            "memory.hbm_write_bandwidth":       {"range": (50.0, None)},
        },
    ),
    BenchmarkSpec(
        name="strided_s1",
        description="Stride-1 access (expect ~100% coalescing)",
        source=_strided_source(1),
        metrics=["memory.coalescing_efficiency"],
        expected={"memory.coalescing_efficiency": {"range": (80.0, 100.0)}},
    ),
    BenchmarkSpec(
        name="strided_s2",
        description="Stride-2 access (expect ~50% coalescing)",
        source=_strided_source(2),
        metrics=["memory.coalescing_efficiency"],
        expected={"memory.coalescing_efficiency": {"range": (35.0, 65.0)}},
    ),
    BenchmarkSpec(
        name="strided_s4",
        description="Stride-4 access (expect ~25% coalescing)",
        source=_strided_source(4),
        metrics=["memory.coalescing_efficiency"],
        expected={"memory.coalescing_efficiency": {"range": (15.0, 40.0)}},
    ),
    BenchmarkSpec(
        name="strided_s8",
        description="Stride-8 access (expect ~12.5% coalescing)",
        source=_strided_source(8),
        metrics=["memory.coalescing_efficiency"],
        expected={"memory.coalescing_efficiency": {"range": (5.0, 50.0)}},
    ),
    BenchmarkSpec(
        name="cache_l2_resident",
        description="8 MB array iterated 100x — L2 should cache after first pass",
        source=CACHE_L2_RESIDENT,
        metrics=["memory.l2_hit_rate", "memory.l2_bandwidth"],
        expected={
            "memory.l2_hit_rate": {"range": (50.0, 100.0)},
        },
    ),
    BenchmarkSpec(
        name="cache_l1_resident",
        description="4 KB per block iterated 1000x — L1 should cache",
        source=CACHE_L1_RESIDENT,
        metrics=["memory.l1_hit_rate"],
        expected={
            "memory.l1_hit_rate": {"range": (80.0, 100.0)},
        },
    ),
    BenchmarkSpec(
        name="lds_conflict_free",
        description="Sequential LDS access — no bank conflicts expected",
        source=_lds_source(0),
        metrics=["memory.lds_bank_conflicts"],
        expected={"memory.lds_bank_conflicts": {"range": (0.0, 2.0)}},
    ),
    BenchmarkSpec(
        name="lds_high_conflict",
        description="Stride-32 LDS access — high bank conflicts expected",
        source=_lds_source(1),
        metrics=["memory.lds_bank_conflicts"],
        expected={"memory.lds_bank_conflicts": {"range": (5.0, None)}},
    ),
    BenchmarkSpec(
        name="compute_valu_fma",
        description="Pure FP32 FMA — validates FLOPS counters",
        source=COMPUTE_VALU_FMA,
        metrics=["compute.total_flops", "compute.hbm_gflops"],
        expected={
            # 512 blocks * 256 threads = 2048 waves * 100000 iters * 2 * 64
            "compute.total_flops": {"value": 2048 * 100000 * 2 * 64, "tolerance_pct": 5.0},
            "compute.hbm_gflops":  {"range": (1.0, None)},
        },
    ),
    BenchmarkSpec(
        name="atomic_high_contention",
        description="All threads atomicAdd to 1 address — high atomic latency",
        source=_atomic_source(True),
        metrics=["memory.atomic_latency"],
        expected={"memory.atomic_latency": {"range": (100.0, None)}},
        arch_filter=["gfx942"],
    ),
    BenchmarkSpec(
        name="atomic_low_contention",
        description="Each thread atomicAdd to own address — lower latency",
        source=_atomic_source(False),
        metrics=["memory.atomic_latency"],
        expected={"memory.atomic_latency": {"range": (0.0, None)}},
        arch_filter=["gfx942"],
    ),
    BenchmarkSpec(
        name="mixed_ai_k100",
        description="100 FMAs per load — high arithmetic intensity",
        source=_mixed_source(100),
        metrics=[
            "compute.hbm_arithmetic_intensity",
            "compute.l2_arithmetic_intensity",
            "compute.hbm_gflops",
            "compute.total_flops",
        ],
        expected={
            "compute.hbm_arithmetic_intensity": {"range": (5.0, None)},
            "compute.total_flops":              {"range": (1.0, None)},
        },
    ),
    BenchmarkSpec(
        name="mixed_ai_k1",
        description="1 FMA per load — low arithmetic intensity",
        source=_mixed_source(1),
        metrics=["compute.hbm_arithmetic_intensity", "compute.total_flops"],
        expected={
            "compute.hbm_arithmetic_intensity": {"range": (0.01, 50.0)},
        },
    ),
]


# ---------------------------------------------------------------------------
# Expectation checking
# ---------------------------------------------------------------------------

def check_expectation(value: float, expected: dict) -> dict:
    """Check a metric value against its expectation."""
    if "range" in expected:
        lo, hi = expected["range"]
        lo = lo if lo is not None else float("-inf")
        hi = hi if hi is not None else float("inf")
        passed = lo <= value <= hi
        return {"pass": passed, "value": value,
                "reason": f"value={value:.4f}, range=[{lo}, {hi}]",
                "expected": f"[{lo}, {hi}]"}

    if "value" in expected:
        exp = expected["value"]
        tol = expected.get("tolerance_pct", 10.0)
        margin = abs(exp) * tol / 100.0
        passed = abs(value - exp) <= margin
        return {"pass": passed, "value": value,
                "reason": f"value={value:.4f}, expected={exp:.4f} +/-{tol}%",
                "expected": f"{exp:.4f} +/-{tol}%"}

    return {"pass": True, "value": value, "reason": "no expectation", "expected": "N/A"}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def detect_arch() -> str:
    from metrix.backends.detect import detect_gpu_arch
    return detect_gpu_arch()


def get_rocm_info() -> dict:
    info = {}
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            if "Name:" in line and "gfx" in line.lower():
                info["gpu"] = line.strip()
                break
    except Exception:
        pass
    try:
        r = subprocess.run(["rocprofv3", "--version"], capture_output=True, text=True, timeout=10)
        info["rocprofv3"] = (r.stdout.strip() or r.stderr.strip())[:80]
    except Exception:
        pass
    return info


def run_one(spec: BenchmarkSpec, arch: str, build_dir: Path,
            num_replays: int = 3) -> dict:
    """Profile one benchmark and check derived metrics."""
    print(f"\n{'─' * 60}")
    print(f"BENCHMARK: {spec.name}")
    print(f"  {spec.description}")

    # Compile
    try:
        binary = _compile_hip(spec.source, spec.name, build_dir)
    except RuntimeError as e:
        print(f"  COMPILE ERROR: {e}")
        return {"benchmark": spec.name, "status": "ERROR", "reason": str(e)}

    # Profile
    from metrix.api import Metrix
    try:
        profiler = Metrix(arch=arch)
        results = profiler.profile(
            command=str(binary),
            metrics=spec.metrics,
            num_replays=num_replays,
            aggregate_by_kernel=True,
            timeout_seconds=120,
        )
    except Exception as e:
        print(f"  PROFILE ERROR: {e}")
        return {"benchmark": spec.name, "status": "ERROR", "reason": str(e)}

    if not results.kernels:
        print("  WARNING: No kernels profiled")
        return {"benchmark": spec.name, "status": "ERROR", "reason": "no kernels"}

    # Pick the kernel with longest duration (the measured run, not warmup)
    kernel = max(results.kernels, key=lambda k: k.duration_us.avg)
    print(f"  Kernel:   {kernel.name}")
    print(f"  Duration: {kernel.duration_us.avg:.1f} us")

    # Check metrics
    metric_results = {}
    all_pass = True

    for metric_name in spec.metrics:
        if metric_name not in kernel.metrics:
            print(f"  {metric_name}: NOT REPORTED")
            metric_results[metric_name] = {"pass": False, "value": None,
                                           "reason": "not in results"}
            all_pass = False
            continue

        value = kernel.metrics[metric_name].avg
        exp = spec.expected.get(metric_name)

        if exp is None:
            print(f"  {metric_name}: {value:.4f} (informational)")
            metric_results[metric_name] = {"pass": True, "value": value,
                                           "reason": "informational"}
            continue

        check = check_expectation(value, exp)
        tag = "PASS" if check["pass"] else "FAIL"
        print(f"  {metric_name}: {value:.4f} — {tag} ({check['reason']})")
        metric_results[metric_name] = check
        if not check["pass"]:
            all_pass = False

    return {
        "benchmark": spec.name,
        "status": "PASS" if all_pass else "FAIL",
        "kernel": kernel.name,
        "duration_us": kernel.duration_us.avg,
        "metrics": metric_results,
    }


def generate_report(results: List[dict], arch: str, rocm_info: dict) -> str:
    lines = [
        "# Metrix Derived Counter Validation Report\n",
        f"**Architecture:** {arch}",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    for k, v in rocm_info.items():
        lines.append(f"**{k}:** {v}")
    lines.append("")

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] in ("ERROR", "SKIP"))

    lines.append(f"## Summary: {passed}/{total} passed, {failed} failed, {errors} errors/skipped\n")

    lines.append("| Benchmark | Status | Details |")
    lines.append("|-----------|--------|---------|")
    for r in results:
        if r["status"] in ("ERROR", "SKIP"):
            lines.append(f"| {r['benchmark']} | **{r['status']}** | {r.get('reason','')} |")
        elif "metrics" in r:
            fails = [m for m, v in r["metrics"].items() if not v.get("pass", True)]
            detail = "Failed: " + ", ".join(fails) if fails else f"{len(r['metrics'])} metrics OK"
            lines.append(f"| {r['benchmark']} | **{r['status']}** | {detail} |")

    lines.append("\n## Detailed Results\n")
    for r in results:
        if r["status"] in ("ERROR", "SKIP"):
            lines.append(f"### {r['benchmark']} — {r['status']}\n{r.get('reason','')}\n")
            continue
        lines.append(f"### {r['benchmark']} — {r['status']}\n")
        if "kernel" in r:
            lines.append(f"- **Kernel:** `{r['kernel']}`")
            lines.append(f"- **Duration:** {r.get('duration_us',0):.1f} us\n")
        if "metrics" in r:
            lines.append("| Metric | Value | Expected | Result |")
            lines.append("|--------|-------|----------|--------|")
            for metric, chk in r["metrics"].items():
                val = f"{chk['value']:.4f}" if chk["value"] is not None else "N/A"
                exp = chk.get("expected", "N/A")
                tag = "PASS" if chk["pass"] else "**FAIL**"
                lines.append(f"| {metric} | {val} | {exp} | {tag} |")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Validate metrix derived counters")
    parser.add_argument("--arch", help="GPU architecture (auto-detect if omitted)")
    parser.add_argument("--benchmark", help="Run a single benchmark by name")
    parser.add_argument("--list", action="store_true", help="List benchmarks")
    parser.add_argument("--report", help="Save markdown report")
    parser.add_argument("--output", help="Save JSON results")
    parser.add_argument("--num-replays", type=int, default=3)
    parser.add_argument("--build-dir", default="/tmp/metrix_bench_build")
    args = parser.parse_args()

    if args.list:
        for b in BENCHMARKS:
            print(f"  {b.name:25s} — {b.description}")
        return 0

    arch = args.arch or detect_arch()
    print(f"Architecture: {arch}")

    # Filter benchmarks
    specs = [b for b in BENCHMARKS
             if not b.arch_filter or arch in b.arch_filter]
    if args.benchmark:
        specs = [b for b in specs if b.name == args.benchmark]
        if not specs:
            print(f"Unknown benchmark: {args.benchmark}")
            return 1

    rocm_info = get_rocm_info()
    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {len(specs)} benchmarks, {args.num_replays} replays each")
    all_results = []
    for spec in specs:
        result = run_one(spec, arch, build_dir, args.num_replays)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in all_results if r["status"] == "PASS")
    failed = sum(1 for r in all_results if r["status"] == "FAIL")
    total = len(all_results)
    print(f"  PASS: {passed}/{total}   FAIL: {failed}/{total}")
    for r in all_results:
        if r["status"] == "FAIL" and "metrics" in r:
            for m, chk in r["metrics"].items():
                if not chk.get("pass", True):
                    print(f"  FAIL: {r['benchmark']}.{m} — {chk['reason']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"arch": arch, "rocm_info": rocm_info, "results": all_results}, f, indent=2)
        print(f"\nJSON: {args.output}")

    report = generate_report(all_results, arch, rocm_info)
    if args.report:
        with open(args.report, "w") as f:
            f.write(report)
        print(f"Report: {args.report}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
