"""
Hardware counter backends for different AMD GPU architectures

Clean design:
- No exposed hw counter mappings
- Counter names appear EXACTLY ONCE (as function parameters)
- Backends define metrics with @metric decorator
"""

from .base import CounterBackend, DeviceSpecs, ProfileResult, Statistics
from .gfx942 import GFX942Backend
from .gfx950 import GFX950Backend
from .gfx1201 import GFX1201Backend
from .gfx90a import GFX90aBackend
from .gfx1151 import GFX1151Backend
from .decorator import metric
from .detect import detect_gpu_arch, detect_or_default


__all__ = [
    "CounterBackend",
    "DeviceSpecs",
    "ProfileResult",
    "Statistics",
    "GFX942Backend",
    "GFX950Backend",
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
        "gfx950": GFX950Backend,
        "mi350x": GFX950Backend,
        "mi350": GFX950Backend,
        "mi355x": GFX950Backend,
        "gfx1201": GFX1201Backend,
        "gfx1151": GFX1151Backend,
    }

    backend_class = backends.get(arch.lower())
    if backend_class is None:
        raise ValueError(
            f"Unsupported architecture: {arch}. Supported: {', '.join(backends.keys())}"
        )

    return backend_class()
