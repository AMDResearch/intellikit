"""
Compute-focused metric definitions

These are display metadata only. Actual computation is in counter_defs.yaml
and the backend implementations.
"""

from .categories import MetricCategory

# ═══════════════════════════════════════════════════════════════════
# COMPUTE THROUGHPUT METRICS
# ═══════════════════════════════════════════════════════════════════

COMPUTE_THROUGHPUT_METRICS = {
    "compute.gpu_utilization": {
        "name": "GPU Utilization",
        "description": "Fraction of elapsed cycles the GPU was actively executing work",
        "unit": "Percent",
        "category": MetricCategory.COMPUTE,
    },
    "compute.total_flops": {
        "name": "Total FLOPS",
        "description": "Total floating-point operations performed by the kernel",
        "unit": "FLOPS",
        "category": MetricCategory.COMPUTE,
    },
    "compute.hbm_gflops": {
        "name": "HBM Compute Throughput",
        "description": "Compute throughput (GFLOPS) normalized by kernel execution time",
        "unit": "GFLOP/s",
        "category": MetricCategory.COMPUTE,
    },
}

# ═══════════════════════════════════════════════════════════════════
# ARITHMETIC INTENSITY METRICS
# ═══════════════════════════════════════════════════════════════════

ARITHMETIC_INTENSITY_METRICS = {
    "compute.hbm_arithmetic_intensity": {
        "name": "HBM Arithmetic Intensity",
        "description": "Ratio of floating-point operations to HBM bytes transferred (FLOP/byte)",
        "unit": "FLOPs/Byte",
        "category": MetricCategory.COMPUTE,
    },
    "compute.l2_arithmetic_intensity": {
        "name": "L2 Arithmetic Intensity",
        "description": "Ratio of floating-point operations to L2 cache bytes accessed (FLOP/byte)",
        "unit": "FLOPs/Byte",
        "category": MetricCategory.COMPUTE,
    },
    "compute.l1_arithmetic_intensity": {
        "name": "L1 Arithmetic Intensity",
        "description": "Ratio of floating-point operations to L1 cache bytes accessed (FLOP/byte)",
        "unit": "FLOPs/Byte",
        "category": MetricCategory.COMPUTE,
    },
}

# ═══════════════════════════════════════════════════════════════════
# COMBINED COMPUTE METRIC CATALOG
# ═══════════════════════════════════════════════════════════════════

COMPUTE_METRICS = {**COMPUTE_THROUGHPUT_METRICS, **ARITHMETIC_INTENSITY_METRICS}
