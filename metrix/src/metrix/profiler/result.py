"""
Profiling result data structures (placeholder)
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class KernelDispatch:
    """Single kernel dispatch result"""

    dispatch_id: int
    kernel_name: str
    device_id: int
    grid_size: Tuple[int, int, int]
    block_size: Tuple[int, int, int]
    duration_ns: int
    raw_counters: Dict[str, float]
    metrics: Dict[str, any]
    source_file: Optional[str] = None
    source_line: Optional[int] = None


class CollectionResult:
    """Result of a profiling session"""

    def __init__(self):
        self.dispatches = []

    def query(self, kernel_pattern=None):
        """Query dispatches - to be implemented"""
        return self.dispatches

    def to_json(self, filepath):
        """Export to JSON - to be implemented"""
        pass

    def to_dataframe(self):
        """Export to DataFrame - to be implemented"""
        pass
