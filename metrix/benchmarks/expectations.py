"""
Expected derived metric values for each microbenchmark.

Each entry defines what the metrix-derived metrics SHOULD report for a
kernel with known, analytically predictable behavior.  The validation
runner compares metrix output against these expectations.

Tolerance types:
  - tolerance_pct:  allowed % deviation from expected value
  - tolerance_abs:  allowed absolute deviation (for values near 0)
  - range:          (min, max) acceptable range
"""

# ---------------------------------------------------------------------------
# Per-benchmark expectations
# ---------------------------------------------------------------------------

EXPECTATIONS = {
    # ------------------------------------------------------------------
    "bw_sequential_read": {
        "description": "Coalesced stride-1 read from HBM (array >> L2)",
        "binary": "bw_sequential_read",
        "args": [],
        "metrics": [
            "memory.l2_hit_rate",
            "memory.l1_hit_rate",
            "memory.coalescing_efficiency",
            "memory.global_load_efficiency",
            "memory.hbm_read_bandwidth",
            "memory.bytes_transferred_hbm",
        ],
        "expected": {
            "memory.coalescing_efficiency":  {"range": (85.0, 100.0)},
            "memory.global_load_efficiency": {"range": (85.0, 100.0)},
            "memory.l2_hit_rate":            {"range": (0.0, 10.0)},
            "memory.l1_hit_rate":            {"range": (0.0, 15.0)},
            "memory.hbm_read_bandwidth":     {"range": (100.0, None)},   # >100 GB/s sanity check
        },
    },

    # ------------------------------------------------------------------
    "bw_sequential_write": {
        "description": "Coalesced stride-1 write to HBM (array >> L2)",
        "binary": "bw_sequential_write",
        "args": [],
        "metrics": [
            "memory.global_store_efficiency",
            "memory.hbm_write_bandwidth",
        ],
        "expected": {
            "memory.global_store_efficiency": {"range": (85.0, 100.0)},
            "memory.hbm_write_bandwidth":     {"range": (50.0, None)},
        },
    },

    # ------------------------------------------------------------------
    "bw_copy": {
        "description": "Copy kernel: dst[i] = src[i], both arrays >> L2",
        "binary": "bw_copy",
        "args": [],
        "metrics": [
            "memory.hbm_bandwidth_utilization",
            "memory.hbm_read_bandwidth",
            "memory.hbm_write_bandwidth",
        ],
        "expected": {
            "memory.hbm_bandwidth_utilization": {"range": (10.0, 100.0)},
            "memory.hbm_read_bandwidth":        {"range": (50.0, None)},
            "memory.hbm_write_bandwidth":       {"range": (50.0, None)},
        },
    },

    # ------------------------------------------------------------------
    "bw_strided_access_s1": {
        "description": "Stride-1 access (baseline, expect ~100% coalescing)",
        "binary": "bw_strided_access",
        "args": ["33554432", "1"],  # 32M elements, stride 1
        "metrics": [
            "memory.coalescing_efficiency",
            "memory.global_load_efficiency",
        ],
        "expected": {
            "memory.coalescing_efficiency":  {"range": (85.0, 100.0)},
            "memory.global_load_efficiency": {"range": (85.0, 100.0)},
        },
    },

    "bw_strided_access_s2": {
        "description": "Stride-2 access (expect ~50% coalescing)",
        "binary": "bw_strided_access",
        "args": ["33554432", "2"],
        "metrics": [
            "memory.coalescing_efficiency",
        ],
        "expected": {
            "memory.coalescing_efficiency": {"range": (35.0, 65.0)},
        },
    },

    "bw_strided_access_s4": {
        "description": "Stride-4 access (expect ~25% coalescing)",
        "binary": "bw_strided_access",
        "args": ["33554432", "4"],
        "metrics": [
            "memory.coalescing_efficiency",
        ],
        "expected": {
            "memory.coalescing_efficiency": {"range": (15.0, 40.0)},
        },
    },

    "bw_strided_access_s8": {
        "description": "Stride-8 access (expect ~12.5% coalescing)",
        "binary": "bw_strided_access",
        "args": ["33554432", "8"],
        "metrics": [
            "memory.coalescing_efficiency",
        ],
        "expected": {
            "memory.coalescing_efficiency": {"range": (5.0, 25.0)},
        },
    },

    # ------------------------------------------------------------------
    "cache_l2_resident": {
        "description": "8 MB array iterated 100x — should hit L2 after warmup",
        "binary": "cache_l2_resident",
        "args": [],
        "metrics": [
            "memory.l2_hit_rate",
            "memory.l2_bandwidth",
            "memory.bytes_transferred_l2",
        ],
        "expected": {
            "memory.l2_hit_rate": {"range": (80.0, 100.0)},
        },
    },

    # ------------------------------------------------------------------
    "cache_l1_resident": {
        "description": "4 KB/block array iterated 1000x — should hit L1",
        "binary": "cache_l1_resident",
        "args": [],
        "metrics": [
            "memory.l1_hit_rate",
            "memory.bytes_transferred_l1",
        ],
        "expected": {
            "memory.l1_hit_rate": {"range": (90.0, 100.0)},
        },
    },

    # ------------------------------------------------------------------
    "lds_conflict_free": {
        "description": "LDS access with no bank conflicts",
        "binary": "lds_bank_conflicts",
        "args": ["0", "10000"],  # mode 0 = conflict-free
        "metrics": [
            "memory.lds_bank_conflicts",
        ],
        "expected": {
            "memory.lds_bank_conflicts": {"range": (0.0, 1.0)},
        },
    },

    "lds_high_conflict": {
        "description": "LDS access with stride-32 bank conflicts",
        "binary": "lds_bank_conflicts",
        "args": ["1", "10000"],  # mode 1 = high-conflict
        "metrics": [
            "memory.lds_bank_conflicts",
        ],
        "expected": {
            "memory.lds_bank_conflicts": {"range": (2.0, None)},  # should be >> 0
        },
    },

    # ------------------------------------------------------------------
    "compute_valu_fma": {
        "description": "Pure FP32 FMA compute — validate FLOPS counters",
        "binary": "compute_valu_fma",
        "args": ["512", "100000"],
        "metrics": [
            "compute.total_flops",
            "compute.hbm_gflops",
        ],
        "expected": {
            # 512 blocks * 256 threads = 131072 threads = 2048 waves
            # 2048 waves * 100000 iters * 2 FLOP * 64 lanes = 26,214,400,000,000 FLOPS
            "compute.total_flops": {"value": 2048 * 100000 * 2 * 64, "tolerance_pct": 5.0},
            "compute.hbm_gflops":  {"range": (1.0, None)},  # sanity: nonzero
        },
    },

    # ------------------------------------------------------------------
    "compute_mfma_asm": {
        "description": "MFMA inline asm — validate MFMA FLOPS counters",
        "binary": "compute_mfma",
        "args": ["0", "50000"],  # mode 0 = inline asm
        "arch_filter": ["gfx942", "gfx90a"],  # MFMA asm requires CDNA
        "metrics": [
            "compute.total_flops",
        ],
        "expected": {
            "compute.total_flops": {"range": (1.0, None)},  # just confirm nonzero
        },
    },

    # ------------------------------------------------------------------
    "atomic_high_contention": {
        "description": "All threads atomicAdd to one address — high latency",
        "binary": "atomic_contention",
        "args": ["0", "1000"],
        "arch_filter": ["gfx942"],
        "metrics": [
            "memory.atomic_latency",
        ],
        "expected": {
            "memory.atomic_latency": {"range": (10.0, None)},  # expect high cycle count
        },
    },

    "atomic_low_contention": {
        "description": "Each thread atomicAdd to own address — low latency",
        "binary": "atomic_contention",
        "args": ["1", "1000"],
        "arch_filter": ["gfx942"],
        "metrics": [
            "memory.atomic_latency",
        ],
        "expected": {
            "memory.atomic_latency": {"range": (0.0, None)},  # expect lower than high-contention
        },
    },

    # ------------------------------------------------------------------
    "mixed_ai_k100": {
        "description": "100 FMAs per load — high arithmetic intensity",
        "binary": "mixed_compute_memory",
        "args": ["67108864", "100"],  # 64M floats, K=100
        "metrics": [
            "compute.hbm_arithmetic_intensity",
            "compute.l2_arithmetic_intensity",
            "compute.hbm_gflops",
            "compute.total_flops",
        ],
        "expected": {
            "compute.hbm_arithmetic_intensity": {"range": (10.0, None)},
            "compute.total_flops":              {"range": (1.0, None)},
        },
    },

    "mixed_ai_k1": {
        "description": "1 FMA per load — low arithmetic intensity",
        "binary": "mixed_compute_memory",
        "args": ["67108864", "1"],  # 64M floats, K=1
        "metrics": [
            "compute.hbm_arithmetic_intensity",
            "compute.total_flops",
        ],
        "expected": {
            # With K=1 AI should be much lower than K=100
            "compute.hbm_arithmetic_intensity": {"range": (0.01, 100.0)},
        },
    },
}


def get_benchmarks_for_arch(arch: str) -> dict:
    """Filter benchmarks to those that apply to the given architecture."""
    result = {}
    for name, spec in EXPECTATIONS.items():
        arch_filter = spec.get("arch_filter")
        if arch_filter and arch not in arch_filter:
            continue
        result[name] = spec
    return result


def check_expectation(metric_name: str, value: float, expected: dict) -> dict:
    """
    Check a metric value against its expectation.

    Returns: {"pass": bool, "reason": str, "value": float, "expected": str}
    """
    if "range" in expected:
        lo, hi = expected["range"]
        lo = lo if lo is not None else float("-inf")
        hi = hi if hi is not None else float("inf")
        passed = lo <= value <= hi
        return {
            "pass": passed,
            "reason": f"value={value:.4f}, expected range [{lo}, {hi}]",
            "value": value,
            "expected": f"[{lo}, {hi}]",
        }

    if "value" in expected:
        exp_val = expected["value"]
        tol_pct = expected.get("tolerance_pct", 10.0)
        margin = abs(exp_val) * tol_pct / 100.0
        passed = abs(value - exp_val) <= margin
        return {
            "pass": passed,
            "reason": f"value={value:.4f}, expected={exp_val:.4f} +/-{tol_pct}%",
            "value": value,
            "expected": f"{exp_val:.4f} +/-{tol_pct}%",
        }

    if "tolerance_abs" in expected:
        exp_val = expected.get("value", 0.0)
        tol = expected["tolerance_abs"]
        passed = abs(value - exp_val) <= tol
        return {
            "pass": passed,
            "reason": f"value={value:.4f}, expected={exp_val:.4f} +/-{tol}",
            "value": value,
            "expected": f"{exp_val:.4f} +/-{tol}",
        }

    return {"pass": True, "reason": "no expectation defined", "value": value, "expected": "N/A"}
