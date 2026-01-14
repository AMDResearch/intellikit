"""
Metric category definitions
"""

from enum import Enum


class MetricCategory(str, Enum):
    """High-level metric categories"""

    OCCUPANCY = "occupancy"
    MEMORY_BANDWIDTH = "memory_bandwidth"
    MEMORY_CACHE = "memory_cache"
    MEMORY_PATTERN = "memory_pattern"
    MEMORY_LDS = "memory_lds"
    COMPUTE = "compute"
    INSTRUCTION = "instruction"
    BOTTLENECK = "bottleneck"
