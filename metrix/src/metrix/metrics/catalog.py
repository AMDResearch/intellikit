"""
Main metric catalog and profiles
"""

from .memory_metrics import MEMORY_METRICS
from .compute_metrics import COMPUTE_METRICS

# ═══════════════════════════════════════════════════════════════════
# COMPLETE METRIC CATALOG
# ═══════════════════════════════════════════════════════════════════

METRIC_CATALOG = {
    **MEMORY_METRICS,
    **COMPUTE_METRICS,
    # Will add occupancy, bottleneck metrics later
}

# ═══════════════════════════════════════════════════════════════════
# PRE-DEFINED METRIC PROFILES
# ═══════════════════════════════════════════════════════════════════

METRIC_PROFILES = {
    "quick": {
        "description": "Fast overview - minimal counters",
        "metrics": [
            "memory.hbm_bandwidth_utilization",
            "memory.l2_hit_rate",
        ],
        "estimated_passes": 1,
    },
    "memory": {
        "description": "Deep dive into GPU memory system performance",
        "metrics": [
            # Bandwidth
            "memory.hbm_bandwidth_utilization",
            "memory.hbm_read_bandwidth",
            "memory.hbm_write_bandwidth",
            "memory.bytes_transferred_hbm",
            # Cache efficiency
            "memory.l1_hit_rate",
            "memory.l2_hit_rate",
            "memory.l2_bandwidth",
            # Access patterns
            "memory.coalescing_efficiency",
            "memory.global_load_efficiency",
            "memory.global_store_efficiency",
            # LDS
            "memory.lds_bank_conflicts",
            # Note: memory.lds_utilization requires kernel metadata, not hardware counters
            # Atomic operations
            "memory.atomic_latency",
        ],
        "estimated_passes": 2,
        "focus": "memory_system",
        "typical_bottlenecks": [
            "uncoalesced_memory_access",
            "low_cache_hit_rate",
            "lds_bank_conflicts",
            "atomic_contention",
        ],
    },
    "memory_bandwidth": {
        "description": "Focus on bandwidth utilization only",
        "metrics": [
            "memory.hbm_bandwidth_utilization",
            "memory.hbm_read_bandwidth",
            "memory.hbm_write_bandwidth",
            "memory.bytes_transferred_hbm",
            "memory.l2_bandwidth",
        ],
        "estimated_passes": 1,
    },
    "memory_cache": {
        "description": "Focus on cache hierarchy efficiency",
        "metrics": [
            "memory.l1_hit_rate",
            "memory.l2_hit_rate",
            "memory.l2_bandwidth",
            "memory.coalescing_efficiency",
        ],
        "estimated_passes": 1,
    },
    "compute": {
        "description": "Compute and arithmetic intensity analysis",
        "metrics": [
            "compute.total_flops",
            "compute.hbm_gflops",
            "compute.hbm_arithmetic_intensity",
            "compute.l2_arithmetic_intensity",
            "compute.l1_arithmetic_intensity",
        ],
        "estimated_passes": 3,
        "focus": "compute_performance",
        "typical_bottlenecks": ["low_arithmetic_intensity", "memory_bound_kernel"],
    },
}


def get_metrics_by_category(category: str) -> list:
    """Get all metrics in a category"""
    return [
        metric_name for metric_name, metric_def in METRIC_CATALOG.items() if metric_def["category"].value == category
    ]


def get_metric_info(metric_name: str) -> dict:
    """Get detailed information about a metric"""
    if metric_name not in METRIC_CATALOG:
        raise ValueError(f"Unknown metric: {metric_name}")
    return METRIC_CATALOG[metric_name]


def list_all_metrics() -> list:
    """List all available metrics"""
    return list(METRIC_CATALOG.keys())


def list_all_profiles() -> list:
    """List all available profiles"""
    return list(METRIC_PROFILES.keys())
