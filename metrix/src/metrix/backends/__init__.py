"""
Hardware counter backends for different AMD GPU architectures

Clean design:
- No exposed hw counter mappings
- Counter names appear EXACTLY ONCE (as function parameters)
- Backends define metrics with @metric decorator
"""

from .base import CounterBackend, DeviceSpecs, ProfileResult, Statistics
from .gfx942 import GFX942Backend
from .gfx1201 import GFX1201Backend
from .gfx90a import GFX90aBackend
from .decorator import metric
from .detect import detect_gpu_arch, detect_or_default

__all__ = [
    "CounterBackend",
    "DeviceSpecs",
    "ProfileResult",
    "Statistics",
    "GFX942Backend",
    "GFX1201Backend",
    "GFX90aBackend",
    "metric",
    "detect_gpu_arch",
    "detect_or_default",
]


def get_backend(arch: str) -> CounterBackend:
    """Get counter backend for architecture"""
    backends = {
        "gfx90a": GFX90aBackend,
        "mi200": GFX90aBackend,
        "gfx942": GFX942Backend,
        "mi300x": GFX942Backend,
        "mi300": GFX942Backend,
        "gfx1201": GFX1201Backend,
    }

    backend_class = backends.get(arch.lower())
    if backend_class is None:
        raise ValueError(f"Unsupported architecture: {arch}. Supported: {', '.join(backends.keys())}")

    return backend_class()
