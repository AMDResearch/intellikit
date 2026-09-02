"""
Main metric catalog and profiles
"""

from .memory_metrics import MEMORY_METRICS
from .compute_metrics import COMPUTE_METRICS
from typing import Iterable, List, Tuple

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
        metric_name
        for metric_name, metric_def in METRIC_CATALOG.items()
        if metric_def["category"].value == category
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


def resolve_profile_metrics(
    profile_name: str,
    available: Iterable[str],
    arch: str,
    unsupported: Iterable[str] = (),
) -> Tuple[List[str], List[str]]:
    """
    Narrow a profile's metric list to what the current architecture knows about.

    Profiles are defined once for every GPU, but metric coverage varies by
    architecture: the CDNA-only entries in ``memory`` have no counterpart on
    RDNA, for instance. Collecting the intersection lets a profile stay useful
    on hardware that implements only part of it.

    Metrics the backend marks unsupported *with a reason* are kept in the
    selection so the caller can report that reason. Dropping them here would
    replace "TCC_EA_ATOMIC_LEVEL_sum is broken on MI200" with a bare
    "not available", losing the part the user can act on.

    Args:
        profile_name: Key into METRIC_PROFILES.
        available: Metric names the backend can compute on this architecture.
        arch: Architecture string, used only for messages.
        unsupported: Metric names the backend rejects with an explicit reason.

    Returns:
        (metrics to collect, metrics dropped as unknown here), both in the
        order the profile declares them.

    Raises:
        ValueError: If the profile is unknown, or if no metric in it exists on
            this architecture.
    """
    if profile_name not in METRIC_PROFILES:
        raise ValueError(
            f"Unknown profile: {profile_name}. Available: {list(METRIC_PROFILES.keys())}"
        )

    known = set(available) | set(unsupported)
    requested = METRIC_PROFILES[profile_name]["metrics"]
    selected = [m for m in requested if m in known]
    dropped = [m for m in requested if m not in known]

    if not selected:
        raise ValueError(
            f"Profile '{profile_name}' has no metrics available on {arch}. "
            f"Unsupported here: {', '.join(dropped)}"
        )

    return selected, dropped
