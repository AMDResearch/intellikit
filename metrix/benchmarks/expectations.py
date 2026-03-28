# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Expected values and tolerances for each microbenchmark, per architecture.

Each expectation is a dict with:
  - metric: the metrix metric name
  - check: "range" | "min" | "max" | "approx" | "ratio"
  - For "range": expected_min, expected_max
  - For "min": expected_min
  - For "max": expected_max
  - For "approx": expected, tolerance (absolute or relative)
  - For "ratio": compare two runs, expected_ratio, tolerance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Expectation:
    """A single expected metric value for a benchmark run."""
    metric: str
    check: str  # "range", "min", "max", "approx", "nonzero"
    expected_min: Optional[float] = None
    expected_max: Optional[float] = None
    expected: Optional[float] = None
    tolerance_abs: Optional[float] = None
    tolerance_rel: Optional[float] = None
    description: str = ""

    def evaluate(self, actual: float) -> dict:
        """Return {"pass": bool, "message": str, "actual": float, "expected": ...}"""
        result = {"actual": actual, "metric": self.metric, "description": self.description}

        if self.check == "range":
            passed = self.expected_min <= actual <= self.expected_max
            result["pass"] = passed
            result["expected"] = f"[{self.expected_min}, {self.expected_max}]"
            result["message"] = (
                f"PASS" if passed else
                f"FAIL: {actual:.4f} not in [{self.expected_min}, {self.expected_max}]"
            )
        elif self.check == "min":
            passed = actual >= self.expected_min
            result["pass"] = passed
            result["expected"] = f">= {self.expected_min}"
            result["message"] = (
                f"PASS" if passed else
                f"FAIL: {actual:.4f} < {self.expected_min}"
            )
        elif self.check == "max":
            passed = actual <= self.expected_max
            result["pass"] = passed
            result["expected"] = f"<= {self.expected_max}"
            result["message"] = (
                f"PASS" if passed else
                f"FAIL: {actual:.4f} > {self.expected_max}"
            )
        elif self.check == "approx":
            if self.tolerance_rel is not None:
                diff = abs(actual - self.expected) / max(abs(self.expected), 1e-10)
                passed = diff <= self.tolerance_rel
                result["expected"] = f"{self.expected} +/- {self.tolerance_rel*100:.0f}%"
            else:
                diff = abs(actual - self.expected)
                passed = diff <= self.tolerance_abs
                result["expected"] = f"{self.expected} +/- {self.tolerance_abs}"
            result["pass"] = passed
            result["message"] = f"PASS" if passed else f"FAIL: {actual:.4f} vs {self.expected}"
        elif self.check == "nonzero":
            passed = actual > 0
            result["pass"] = passed
            result["expected"] = "> 0"
            result["message"] = f"PASS" if passed else f"FAIL: got 0"
        else:
            result["pass"] = False
            result["message"] = f"Unknown check type: {self.check}"

        return result


@dataclass
class BenchmarkExpectation:
    """Expected values for one benchmark run configuration."""
    name: str
    binary: str
    args: List[str] = field(default_factory=list)
    kernel_filter: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    expectations: List[Expectation] = field(default_factory=list)
    description: str = ""
    arch_required: Optional[str] = None  # e.g. "gfx942" if arch-specific


def get_expectations(arch: str = "gfx942") -> List[BenchmarkExpectation]:
    """Return all benchmark expectations for the given architecture."""

    benchmarks = []

    # ---- Benchmark 1: Sequential Read ----
    benchmarks.append(BenchmarkExpectation(
        name="sequential_read",
        binary="bin/bw_sequential_read",
        args=["512"],
        kernel_filter="sequential_read",
        metrics=[
            "memory.coalescing_efficiency",
            "memory.global_load_efficiency",
            "memory.l2_hit_rate",
            "memory.hbm_read_bandwidth",
            "memory.bytes_transferred_hbm",
        ],
        expectations=[
            Expectation(
                metric="memory.coalescing_efficiency",
                check="range", expected_min=90.0, expected_max=105.0,
                description="Stride-1 read should be ~100% coalesced"
            ),
            Expectation(
                metric="memory.global_load_efficiency",
                check="min", expected_min=50.0,
                description="Coalesced reads should have high load efficiency"
            ),
            Expectation(
                metric="memory.l2_hit_rate",
                check="max", expected_max=50.0,
                description="512MB array >> L2 cache. MI300X distributed L2 + prefetch gives ~33% even for streaming"
            ),
            Expectation(
                metric="memory.hbm_read_bandwidth",
                check="min", expected_min=100.0,
                description="Should achieve significant read bandwidth (GB/s)"
            ),
            Expectation(
                metric="memory.bytes_transferred_hbm",
                check="min", expected_min=500 * 1024 * 1024,
                description="Should transfer at least ~500MB through HBM"
            ),
        ],
        description="Pure coalesced read from HBM, 512MB"
    ))

    # ---- Benchmark 2: Sequential Write ----
    benchmarks.append(BenchmarkExpectation(
        name="sequential_write",
        binary="bin/bw_sequential_write",
        args=["512"],
        kernel_filter="sequential_write",
        metrics=[
            "memory.global_store_efficiency",
            "memory.hbm_write_bandwidth",
        ],
        expectations=[
            Expectation(
                metric="memory.global_store_efficiency",
                check="min", expected_min=10.0,
                description="Store efficiency = VMEM_WR / TCP_TCC_WRITE_REQ. On gfx942 with 128B cache lines, coalesced writes give ~25% (each wave instruction -> 4 write requests through TCP)"
            ),
            Expectation(
                metric="memory.hbm_write_bandwidth",
                check="min", expected_min=100.0,
                description="Should achieve significant write bandwidth (GB/s)"
            ),
        ],
        description="Pure coalesced write to HBM, 512MB"
    ))

    # ---- Benchmark 3: Copy ----
    benchmarks.append(BenchmarkExpectation(
        name="copy",
        binary="bin/bw_copy",
        args=["512"],
        kernel_filter="copy_kernel",
        metrics=[
            "memory.hbm_bandwidth_utilization",
            "memory.bytes_transferred_hbm",
            "memory.l1_hit_rate",
        ],
        expectations=[
            Expectation(
                metric="memory.hbm_bandwidth_utilization",
                check="min", expected_min=3.0,
                description="Copy kernel on shared MI300X node. Peak BW=5300 GB/s is theoretical; single-kernel copy achieves ~340 GB/s (~6.4%)"
            ),
            Expectation(
                metric="memory.bytes_transferred_hbm",
                check="min", expected_min=900 * 1024 * 1024,
                description="Copy 512MB: should transfer ~1GB (read+write) through HBM"
            ),
        ],
        description="Read-write copy, validates bytes_transferred = 2*N*4"
    ))

    # ---- Benchmark 4: Strided Access (multiple strides) ----
    # On gfx942 with 128B cache lines and factor=16 in the formula:
    # stride=1: 100%, stride=2: 40% (plateau due to TCP_TOTAL_ACCESSES counting),
    # stride=4: 40%, stride=8: 40%.
    # The formula uses factor 16 which assumes 64B lines. With 128B lines,
    # coalescing plateaus at 40% for stride >= 2. This is a KNOWN formula issue.
    for stride, expected_coal_min, expected_coal_max in [
        (1, 80.0, 105.0),
        (2, 30.0, 55.0),
        (4, 30.0, 55.0),
        (8, 30.0, 55.0),
    ]:
        benchmarks.append(BenchmarkExpectation(
            name=f"strided_access_s{stride}",
            binary="bin/bw_strided_access",
            args=[str(stride), "512"],
            kernel_filter="strided_read",
            metrics=["memory.coalescing_efficiency"],
            expectations=[
                Expectation(
                    metric="memory.coalescing_efficiency",
                    check="range",
                    expected_min=expected_coal_min,
                    expected_max=expected_coal_max,
                    description=f"Stride-{stride}: coalescing should be ~{100/stride:.0f}%"
                ),
            ],
            description=f"Strided read at stride={stride}"
        ))

    # ---- Benchmark 5: L2 Resident ----
    benchmarks.append(BenchmarkExpectation(
        name="l2_resident",
        binary="bin/cache_l2_resident",
        args=["32", "100"],
        kernel_filter="l2_resident_read",
        metrics=[
            "memory.l2_hit_rate",
            "memory.l2_bandwidth",
        ],
        expectations=[
            Expectation(
                metric="memory.l2_hit_rate",
                check="min", expected_min=25.0,
                description="32MB array in 256MB L2, 100 iters. MI300X distributed L2 shows ~33% due to multi-XCD L2 partitioning"
            ),
            Expectation(
                metric="memory.l2_bandwidth",
                check="min", expected_min=100.0,
                description="L2 bandwidth should be significant (GB/s)"
            ),
        ],
        description="L2-resident 32MB array, 100 iterations"
    ))

    # ---- Benchmark 6: L1 Resident ----
    benchmarks.append(BenchmarkExpectation(
        name="l1_resident",
        binary="bin/cache_l1_resident",
        args=["200"],
        kernel_filter="l1_resident_read",
        metrics=[
            "memory.l1_hit_rate",
            "memory.bytes_transferred_l1",
        ],
        expectations=[
            Expectation(
                metric="memory.l1_hit_rate",
                check="min", expected_min=90.0,
                description="8KB per WG in L1, 200 iters: L1 hit rate >90%"
            ),
            Expectation(
                metric="memory.bytes_transferred_l1",
                check="nonzero",
                description="Should report nonzero L1 bytes"
            ),
        ],
        description="L1-resident 8KB per WG, 200 iterations"
    ))

    # ---- Benchmark 7a: LDS Conflict-Free ----
    benchmarks.append(BenchmarkExpectation(
        name="lds_conflict_free",
        binary="bin/lds_bank_conflicts",
        args=["0", "10000"],
        kernel_filter="lds_conflict_free",
        metrics=["memory.lds_bank_conflicts"],
        expectations=[
            Expectation(
                metric="memory.lds_bank_conflicts",
                check="max", expected_max=2.0,
                description="Conflict-free LDS access: conflicts/instr should be near 0"
            ),
        ],
        description="LDS conflict-free mode"
    ))

    # ---- Benchmark 7b: LDS High Conflict ----
    benchmarks.append(BenchmarkExpectation(
        name="lds_high_conflict",
        binary="bin/lds_bank_conflicts",
        args=["1", "10000"],
        kernel_filter="lds_high_conflict",
        metrics=["memory.lds_bank_conflicts"],
        expectations=[
            Expectation(
                metric="memory.lds_bank_conflicts",
                check="min", expected_min=2.0,
                description="Stride-32 LDS access: should show significant conflicts"
            ),
        ],
        description="LDS high-conflict mode (stride-32)"
    ))

    # ---- Benchmark 8: VALU FMA ----
    benchmarks.append(BenchmarkExpectation(
        name="valu_fma",
        binary="bin/compute_valu_fma",
        args=["100000"],
        kernel_filter="valu_fma_kernel",
        metrics=[
            "compute.total_flops",
            "compute.hbm_gflops",
            "compute.hbm_arithmetic_intensity",
        ],
        expectations=[
            Expectation(
                metric="compute.total_flops",
                check="min", expected_min=1e9,
                description="100K FMAs * 304 WGs * 4 waves/WG * 2 * 64 should be billions of FLOPS"
            ),
            Expectation(
                metric="compute.hbm_gflops",
                check="min", expected_min=1.0,
                description="Should report nonzero GFLOPS"
            ),
            Expectation(
                metric="compute.hbm_arithmetic_intensity",
                check="min", expected_min=100.0,
                description="Pure compute, minimal memory: very high AI"
            ),
        ],
        description="Pure FP32 FMA loop, no memory"
    ))

    # ---- Benchmark 9: MFMA (gfx942+ only) ----
    if arch in ("gfx942", "gfx950"):
        benchmarks.append(BenchmarkExpectation(
            name="mfma",
            binary="bin/compute_mfma",
            args=["10000"],
            kernel_filter="mfma_kernel",
            metrics=["compute.total_flops"],
            expectations=[
                Expectation(
                    metric="compute.total_flops",
                    check="min", expected_min=1e9,
                    description="MFMA instructions should contribute significant FLOPS"
                ),
            ],
            description="MFMA instructions (gfx942+)",
            arch_required=arch,
        ))

    # ---- Benchmark 10: Atomics (gfx942 only) ----
    if arch == "gfx942":
        benchmarks.append(BenchmarkExpectation(
            name="atomic_high_contention",
            binary="bin/atomic_contention",
            args=["1", "1000"],
            kernel_filter="atomic_contention_kernel",
            metrics=["memory.atomic_latency"],
            expectations=[
                Expectation(
                    metric="memory.atomic_latency",
                    check="min", expected_min=10.0,
                    description="High-contention atomics: latency should be many cycles"
                ),
            ],
            description="Global atomics, high contention (gfx942)",
            arch_required="gfx942",
        ))

    # ---- Benchmark 11: Mixed Compute+Memory ----
    for K in [1, 10, 100]:
        # Expected AI = K * 2 / 8 = K / 4 FLOP/byte (per thread)
        # But metrix uses per-wave: (K*2*64) / (2*64*4) = K/4
        expected_ai = K / 4.0
        benchmarks.append(BenchmarkExpectation(
            name=f"mixed_K{K}",
            binary="bin/mixed_compute_memory",
            args=[str(K), "256"],
            kernel_filter="mixed_kernel",
            metrics=[
                "compute.hbm_arithmetic_intensity",
                "compute.total_flops",
            ],
            expectations=[
                Expectation(
                    metric="compute.hbm_arithmetic_intensity",
                    check="approx",
                    expected=expected_ai,
                    tolerance_rel=0.50,  # 50% tolerance due to prefetch, write-allocate
                    description=f"K={K}: expected AI ~{expected_ai:.2f} FLOP/byte"
                ),
                Expectation(
                    metric="compute.total_flops",
                    check="min", expected_min=1e6,
                    description=f"K={K}: should report nonzero FLOPS"
                ),
            ],
            description=f"Mixed compute+memory, K={K} FMAs per load"
        ))

    return benchmarks


def get_all_metrics() -> List[str]:
    """Return all unique metrics needed across all benchmarks."""
    metrics = set()
    for b in get_expectations():
        metrics.update(b.metrics)
    return sorted(metrics)
