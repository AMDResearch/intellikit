"""
Metric decorator for backend implementations

Counter names appear EXACTLY ONCE - as function parameter names
"""

import inspect
from typing import Callable, Optional


def metric(name: str, unsupported_reason: Optional[str] = None):
    """
    Decorator that auto-discovers counter requirements from function signature

    Usage:
        @metric("memory.l2_hit_rate")
        def _l2_hit_rate(self, TCC_HIT_sum, TCC_MISS_sum):
            total = TCC_HIT_sum + TCC_MISS_sum
            return (TCC_HIT_sum / total) * 100 if total else 0.0

        # Mark as unsupported on specific architecture:
        @metric("memory.atomic_latency", unsupported_reason="Counter broken on MI200")
        def _atomic_latency(self, TCC_EA_ATOMIC_LEVEL_sum, TCC_EA_ATOMIC_sum):
            ...

    The decorator:
    1. Inspects function signature to discover counter names
    2. Wraps function to extract counter values from backend._raw_data
    3. No counter names appear anywhere else in the codebase!

    Args:
        name: Metric name (e.g., "memory.l2_hit_rate")
        unsupported_reason: If set, marks this metric as unsupported on this backend

    Returns:
        Decorated function that extracts counters and computes metric
    """
    def decorator(func: Callable) -> Callable:
        # Introspect function signature to find parameter names
        sig = inspect.signature(func)
        param_names = [
            p.name for p in sig.parameters.values()
            if p.name != 'self'  # Skip 'self' parameter
        ]

        # Parameter names ARE the hardware counter names!
        def wrapper(self) -> float:
            """
            Wrapper extracts counter values from backend._raw_data
            and passes them as kwargs to the original function
            """
            # Extract values for each counter parameter
            kwargs = {
                counter: self._raw_data.get(counter, 0)
                for counter in param_names
            }
            return func(self, **kwargs)

        # Attach metadata for discovery
        wrapper._metric_name = name
        wrapper._metric_counters = param_names
        wrapper._unsupported_reason = unsupported_reason
        wrapper._original_func = func  # Keep for introspection/debugging

        return wrapper

    return decorator

