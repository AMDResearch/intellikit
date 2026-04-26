"""
Memory-focused metric definitions

These are display metadata only. Actual computation is in counter_defs.yaml
and the backend implementations.
"""

from .categories import MetricCategory

# ═══════════════════════════════════════════════════════════════════
# MEMORY BANDWIDTH METRICS
# ═══════════════════════════════════════════════════════════════════

MEMORY_BANDWIDTH_METRICS = {
    "memory.hbm_bandwidth_utilization": {
        "name": "HBM Bandwidth Utilization",
        "description": "Percentage of peak HBM bandwidth utilized",
        "unit": "Percent",
        "category": MetricCategory.MEMORY_BANDWIDTH,
    },
    "memory.hbm_read_bandwidth": {
        "name": "HBM Read Bandwidth",
        "description": "Achieved read bandwidth from HBM in GB/s",
        "unit": "GB/s",
        "category": MetricCategory.MEMORY_BANDWIDTH,
    },
    "memory.hbm_write_bandwidth": {
        "name": "HBM Write Bandwidth",
        "description": "Achieved write bandwidth to HBM in GB/s",
        "unit": "GB/s",
        "category": MetricCategory.MEMORY_BANDWIDTH,
    },
    "memory.bytes_transferred_hbm": {
        "name": "Total HBM Bytes Transferred",
        "description": "Total bytes transferred through HBM (read + write)",
        "unit": "Bytes",
        "category": MetricCategory.MEMORY_BANDWIDTH,
    },
    "memory.bytes_transferred_l2": {
        "name": "Total L2 Bytes Transferred",
        "description": "Total bytes accessed through L2 cache",
        "unit": "Bytes",
        "category": MetricCategory.MEMORY_BANDWIDTH,
    },
    "memory.bytes_transferred_l1": {
        "name": "Total L1 Bytes Transferred",
        "description": "Total bytes accessed through L1 cache",
        "unit": "Bytes",
        "category": MetricCategory.MEMORY_BANDWIDTH,
    },
}

# ═══════════════════════════════════════════════════════════════════
# CACHE EFFICIENCY METRICS
# ═══════════════════════════════════════════════════════════════════

CACHE_METRICS = {
    "memory.l2_hit_rate": {
        "name": "L2 Cache Hit Rate",
        "description": "Percentage of L2 cache accesses that hit",
        "unit": "Percent",
        "category": MetricCategory.MEMORY_CACHE,
    },
    "memory.l1_hit_rate": {
        "name": "L1 Cache Hit Rate",
        "description": "Percentage of L1 (TCP) cache accesses that hit",
        "unit": "Percent",
        "category": MetricCategory.MEMORY_CACHE,
    },
    "memory.l2_bandwidth": {
        "name": "L2 Cache Bandwidth Utilization",
        "description": "Percentage of peak L2 bandwidth utilized",
        "unit": "Percent",
        "category": MetricCategory.MEMORY_CACHE,
    },
}

# ═══════════════════════════════════════════════════════════════════
# MEMORY ACCESS PATTERN METRICS
# ═══════════════════════════════════════════════════════════════════

MEMORY_PATTERN_METRICS = {
    "memory.coalescing_efficiency": {
        "name": "Memory Coalescing Efficiency",
        "description": "How well memory accesses from threads in a wavefront coalesce into fewer transactions",
        "unit": "Percent",
        "category": MetricCategory.MEMORY_PATTERN,
    },
    "memory.global_load_efficiency": {
        "name": "Global Load Efficiency",
        "description": "Ratio of requested global load bytes to actual bytes transferred",
        "unit": "Percent",
        "category": MetricCategory.MEMORY_PATTERN,
    },
    "memory.global_store_efficiency": {
        "name": "Global Store Efficiency",
        "description": "Ratio of requested global store bytes to actual bytes transferred",
        "unit": "Percent",
        "category": MetricCategory.MEMORY_PATTERN,
    },
}

# ═══════════════════════════════════════════════════════════════════
# LDS (LOCAL DATA SHARE / SHARED MEMORY) METRICS
# ═══════════════════════════════════════════════════════════════════

LDS_METRICS = {
    "memory.lds_utilization": {
        "name": "LDS Utilization",
        "description": "Percentage of available LDS (shared memory) used",
        "unit": "Percent",
        "category": MetricCategory.MEMORY_LDS,
    },
    "memory.lds_bank_conflicts": {
        "name": "LDS Bank Conflicts",
        "description": "Number of LDS bank conflicts per instruction",
        "unit": "Conflicts per Access",
        "category": MetricCategory.MEMORY_LDS,
    },
}

# ═══════════════════════════════════════════════════════════════════
# ATOMIC OPERATION METRICS
# ═══════════════════════════════════════════════════════════════════

ATOMIC_METRICS = {
    "memory.atomic_latency": {
        "name": "Atomic Operation Latency",
        "description": "Average latency of atomic operations at L2 cache (cycles per atomic operation)",
        "unit": "Cycles",
        "category": MetricCategory.MEMORY_PATTERN,
    }
}

# ═══════════════════════════════════════════════════════════════════
# COMBINED MEMORY METRIC CATALOG
# ═══════════════════════════════════════════════════════════════════

MEMORY_METRICS = {
    **MEMORY_BANDWIDTH_METRICS,
    **CACHE_METRICS,
    **MEMORY_PATTERN_METRICS,
    **LDS_METRICS,
    **ATOMIC_METRICS,
}
