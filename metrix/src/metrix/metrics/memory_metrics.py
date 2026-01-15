"""
Memory-focused metric definitions
Top-down approach: Define what we want to know, not how to measure it

NOTE: The `derived_from` field contains CONCEPTUAL counter names for documentation.
Actual hardware counter names vary by architecture (e.g., TCC_EA_* vs TCC_EA0_*).
For architecture-specific counter names, see the backend implementations in
metrix/backends/gfx942.py, gfx1201.py, etc.
"""

from .categories import MetricCategory

# ═══════════════════════════════════════════════════════════════════
# MEMORY BANDWIDTH METRICS
# ═══════════════════════════════════════════════════════════════════

MEMORY_BANDWIDTH_METRICS = {
    "memory.hbm_bandwidth_utilization": {
        "name": "HBM Bandwidth Utilization",
        "description": "Percentage of peak HBM (High Bandwidth Memory) bandwidth utilized",
        "unit": "percent",
        "category": MetricCategory.MEMORY_BANDWIDTH,
        # NOTE: These are conceptual counter names. Actual names vary by architecture:
        # - MI300 (gfx942): TCC_EA0_RDREQ_sum, TCC_EA0_WRREQ_sum
        # - MI200 (gfx90a): TCC_EA_RDREQ_sum, TCC_EA_WRREQ_sum
        "derived_from": [
            "TCC_EA_RDREQ_sum",  # Read requests to memory controller
            "TCC_EA_WRREQ_sum",  # Write requests to memory controller
            "GRBM_GUI_ACTIVE",  # GPU active cycles
        ],
        "formula": """
            # Each request is 64 bytes
            bytes_read = TCC_EA_RDREQ_sum * 64
            bytes_write = TCC_EA_WRREQ_sum * 64
            total_bytes = bytes_read + bytes_write

            # Calculate time in nanoseconds
            time_ns = (GRBM_GUI_ACTIVE * 1000) / gpu_freq_mhz

            # Achieved bandwidth in GB/s
            achieved_bw = total_bytes / time_ns

            # Compare to peak
            peak_bw = device_specs['hbm_bandwidth_gbs']
            utilization = (achieved_bw / peak_bw) * 100

            return utilization
        """,
        "device_specific": True,
        "interpretation": {
            "excellent": (80, 100, "Memory bandwidth is well utilized"),
            "good": (60, 80, "Good bandwidth utilization"),
            "fair": (40, 60, "Moderate bandwidth usage"),
            "poor": (0, 40, "Low bandwidth utilization - may be compute bound or low occupancy"),
        },
    },
    "memory.hbm_read_bandwidth": {
        "name": "HBM Read Bandwidth",
        "description": "Achieved read bandwidth from HBM in GB/s",
        "unit": "GB/s",
        "category": MetricCategory.MEMORY_BANDWIDTH,
        "derived_from": ["TCC_EA_RDREQ_sum", "GRBM_GUI_ACTIVE"],
        "formula": """
            bytes_read = TCC_EA_RDREQ_sum * 64
            time_ns = (GRBM_GUI_ACTIVE * 1000) / gpu_freq_mhz
            return bytes_read / time_ns
        """,
        "device_specific": True,
    },
    "memory.hbm_write_bandwidth": {
        "name": "HBM Write Bandwidth",
        "description": "Achieved write bandwidth to HBM in GB/s",
        "unit": "GB/s",
        "category": MetricCategory.MEMORY_BANDWIDTH,
        "derived_from": ["TCC_EA_WRREQ_sum", "GRBM_GUI_ACTIVE"],
        "formula": """
            bytes_write = TCC_EA_WRREQ_sum * 64
            time_ns = (GRBM_GUI_ACTIVE * 1000) / gpu_freq_mhz
            return bytes_write / time_ns
        """,
        "device_specific": True,
    },
    "memory.bytes_transferred_hbm": {
        "name": "Total HBM Bytes Transferred",
        "description": "Total bytes transferred through HBM (read + write)",
        "unit": "bytes",
        "category": MetricCategory.MEMORY_BANDWIDTH,
        "derived_from": ["TCC_EA_RDREQ_sum", "TCC_EA_WRREQ_sum"],
        "formula": """
            return (TCC_EA_RDREQ_sum + TCC_EA_WRREQ_sum) * 64
        """,
    },
    "memory.bytes_transferred_l2": {
        "name": "Total L2 Bytes Transferred",
        "description": "Total bytes accessed through L2 cache",
        "unit": "bytes",
        "category": MetricCategory.MEMORY_BANDWIDTH,
        "derived_from": ["TCC_REQ_sum"],
        "formula": """
            # L2 cache line is 128 bytes
            return TCC_REQ_sum * 128
        """,
    },
    "memory.bytes_transferred_l1": {
        "name": "Total L1 Bytes Transferred",
        "description": "Total bytes accessed through L1 cache",
        "unit": "bytes",
        "category": MetricCategory.MEMORY_BANDWIDTH,
        "derived_from": ["TCP_TOTAL_CACHE_ACCESSES_sum"],
        "formula": """
            # L1 cache line is 128 bytes on gfx942
            return TCP_TOTAL_CACHE_ACCESSES_sum * 128
        """,
    },
}

# ═══════════════════════════════════════════════════════════════════
# CACHE EFFICIENCY METRICS
# ═══════════════════════════════════════════════════════════════════

CACHE_METRICS = {
    "memory.l2_hit_rate": {
        "name": "L2 Cache Hit Rate",
        "description": "Percentage of L2 cache accesses that hit",
        "unit": "percent",
        "category": MetricCategory.MEMORY_CACHE,
        "derived_from": ["TCC_HIT_sum", "TCC_MISS_sum"],
        "formula": """
            total_accesses = TCC_HIT_sum + TCC_MISS_sum
            if total_accesses == 0:
                return 0.0
            return (TCC_HIT_sum / total_accesses) * 100
        """,
        "interpretation": {
            "excellent": (80, 100, "Excellent cache reuse"),
            "good": (60, 80, "Good cache efficiency"),
            "fair": (40, 60, "Moderate cache efficiency"),
            "poor": (0, 40, "Poor cache reuse - data access patterns may need optimization"),
        },
    },
    "memory.l1_hit_rate": {
        "name": "L1 Cache Hit Rate",
        "description": "Percentage of L1 (TCP) cache accesses that hit",
        "unit": "percent",
        "category": MetricCategory.MEMORY_CACHE,
        "derived_from": ["TCP_TCC_READ_REQ_sum", "TCC_EA_RDREQ_sum"],
        "formula": """
            # L1 hits = requests that didn't go to L2
            l1_requests = TCP_TCC_READ_REQ_sum
            l2_requests = TCC_EA_RDREQ_sum

            if l1_requests == 0:
                return 0.0

            l1_hits = l1_requests - l2_requests
            return (l1_hits / l1_requests) * 100
        """,
        "interpretation": {
            "excellent": (70, 100, "Excellent L1 cache locality"),
            "good": (50, 70, "Good L1 cache usage"),
            "fair": (30, 50, "Moderate L1 cache efficiency"),
            "poor": (0, 30, "Poor L1 cache utilization - check data locality"),
        },
    },
    "memory.l2_bandwidth": {
        "name": "L2 Cache Bandwidth Utilization",
        "description": "Percentage of peak L2 bandwidth utilized",
        "unit": "percent",
        "category": MetricCategory.MEMORY_CACHE,
        "derived_from": ["TCC_EA_RDREQ_sum", "TCC_EA_WRREQ_sum", "GRBM_GUI_ACTIVE"],
        "formula": """
            bytes_transferred = (TCC_EA_RDREQ_sum + TCC_EA_WRREQ_sum) * 64
            time_ns = (GRBM_GUI_ACTIVE * 1000) / gpu_freq_mhz
            achieved_bw = bytes_transferred / time_ns
            peak_bw = device_specs['l2_bandwidth_gbs']
            return (achieved_bw / peak_bw) * 100
        """,
        "device_specific": True,
    },
}

# ═══════════════════════════════════════════════════════════════════
# MEMORY ACCESS PATTERN METRICS
# ═══════════════════════════════════════════════════════════════════

MEMORY_PATTERN_METRICS = {
    "memory.coalescing_efficiency": {
        "name": "Memory Coalescing Efficiency",
        "description": "How well memory accesses from threads in a wavefront coalesce into fewer transactions",
        "unit": "percent",
        "category": MetricCategory.MEMORY_PATTERN,
        "derived_from": ["TCP_TOTAL_CACHE_ACCESSES_sum", "TCP_TOTAL_ACCESSES_sum"],
        "formula": """
            # Higher is better - means more threads' accesses coalesce
            total_wavefront_accesses = TCP_TOTAL_ACCESSES_sum
            actual_cache_accesses = TCP_TOTAL_CACHE_ACCESSES_sum

            if actual_cache_accesses == 0:
                return 100.0

            # Ideal: 64 threads in a wavefront access contiguous 64 bytes → 1 cache line
            # Poor: 64 threads access scattered memory → 64 cache lines
            efficiency = (total_wavefront_accesses / actual_cache_accesses) * 100
            return min(100.0, efficiency)
        """,
        "interpretation": {
            "excellent": (80, 100, "Excellent memory coalescing - threads access contiguous memory"),
            "good": (60, 80, "Good coalescing"),
            "fair": (40, 60, "Moderate coalescing - some optimization possible"),
            "poor": (0, 40, "Poor coalescing - threads access scattered memory, causing many transactions"),
        },
    },
    "memory.global_load_efficiency": {
        "name": "Global Load Efficiency",
        "description": "Ratio of requested global load bytes to actual bytes transferred",
        "unit": "percent",
        "category": MetricCategory.MEMORY_PATTERN,
        "derived_from": ["TCP_TOTAL_CACHE_ACCESSES_sum", "TCC_EA_RDREQ_sum"],
        "formula": """
            requested_cache_lines = TCP_TOTAL_CACHE_ACCESSES_sum
            actual_memory_transactions = TCC_EA_RDREQ_sum

            if actual_memory_transactions == 0:
                return 100.0

            # Each cache line is 64 bytes
            efficiency = (requested_cache_lines / actual_memory_transactions) * 100
            return min(100.0, efficiency)
        """,
    },
    "memory.global_store_efficiency": {
        "name": "Global Store Efficiency",
        "description": "Ratio of requested global store bytes to actual bytes transferred",
        "unit": "percent",
        "category": MetricCategory.MEMORY_PATTERN,
        "derived_from": ["TCP_TOTAL_WRITE_sum", "TCC_EA_WRREQ_sum"],
        "formula": """
            requested_writes = TCP_TOTAL_WRITE_sum
            actual_memory_writes = TCC_EA_WRREQ_sum

            if actual_memory_writes == 0:
                return 100.0

            efficiency = (requested_writes / actual_memory_writes) * 100
            return min(100.0, efficiency)
        """,
    },
}

# ═══════════════════════════════════════════════════════════════════
# LDS (LOCAL DATA SHARE / SHARED MEMORY) METRICS
# ═══════════════════════════════════════════════════════════════════

LDS_METRICS = {
    "memory.lds_utilization": {
        "name": "LDS Utilization",
        "description": "Percentage of available LDS (shared memory) used",
        "unit": "percent",
        "category": MetricCategory.MEMORY_LDS,
        "derived_from": ["KERNEL_LDS_SIZE"],  # From kernel metadata
        "formula": """
            lds_used = KERNEL_LDS_SIZE
            lds_available = device_specs['lds_size_per_cu']
            return (lds_used / lds_available) * 100
        """,
        "requires_kernel_info": True,
        "device_specific": True,
    },
    "memory.lds_bank_conflicts": {
        "name": "LDS Bank Conflicts",
        "description": "Number of LDS bank conflicts per instruction",
        "unit": "conflicts/instruction",
        "category": MetricCategory.MEMORY_LDS,
        "derived_from": ["SQ_LDS_BANK_CONFLICT", "SQ_INSTS_LDS"],
        "formula": """
            if SQ_INSTS_LDS == 0:
                return 0.0
            return SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS
        """,
        "interpretation": {
            "excellent": (0, 0.1, "No significant bank conflicts"),
            "good": (0.1, 0.3, "Low bank conflicts"),
            "fair": (0.3, 0.5, "Moderate bank conflicts"),
            "poor": (0.5, float("inf"), "High bank conflicts - check LDS access patterns"),
        },
    },
}

# ═══════════════════════════════════════════════════════════════════
# ATOMIC OPERATION METRICS
# ═══════════════════════════════════════════════════════════════════

ATOMIC_METRICS = {
    "memory.atomic_latency": {
        "name": "Atomic Operation Latency",
        "description": "Average latency of atomic operations at L2 cache (cycles per atomic operation). NOTE: Not available on MI200 (gfx90a) due to broken counters.",
        "unit": "cycles",
        "category": MetricCategory.MEMORY_PATTERN,
        "derived_from": ["TCC_EA_ATOMIC_LEVEL_sum", "TCC_EA_ATOMIC_sum"],
        "formula": """
            if TCC_EA_ATOMIC_sum == 0:
                return 0.0
            return TCC_EA_ATOMIC_LEVEL_sum / TCC_EA_ATOMIC_sum
        """,
        "architecture_support": {
            "gfx942": True,  # MI300X - supported
            "gfx90a": False,  # MI200 - counters broken
            "gfx1201": False,  # Navi3x - not supported
        },
        "interpretation": {
            "excellent": (0, 50, "Very low latency - minimal atomic contention"),
            "good": (50, 100, "Acceptable atomic operation latency"),
            "fair": (100, 200, "Moderate atomic contention"),
            "poor": (
                200,
                float("inf"),
                "High atomic contention - consider reducing atomic operations or using different synchronization",
            ),
        },
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
