"""
Metrix - GPU Profiling. Decoded.
Clean, human-readable metrics for AMD GPUs
"""

__version__ = "0.1.0"

from .api import Metrix
from .profiler.engine import Profiler
from .profiler.result import CollectionResult, KernelDispatch

__all__ = ["Metrix", "Profiler", "CollectionResult", "KernelDispatch"]
