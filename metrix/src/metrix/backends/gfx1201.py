"""
GFX1201 (RDNA4) Backend

Metrics are loaded from counter_defs.yaml.
This file provides architecture-specific infrastructure only.
Device specs are queried from hipGetDeviceProperties at runtime.
"""

from .base import CounterBackend, ProfileResult
from .device_info import query_device_specs
from ..profiler.rocprof_wrapper import ROCProfV3Wrapper
from pathlib import Path
from typing import List, Optional


class GFX1201Backend(CounterBackend):
    """AMD RDNA4 (gfx1201) backend."""

    def _get_device_specs(self):
        return query_device_specs("gfx1201")

    def _run_rocprof(
        self,
        command: str,
        counters: List[str],
        kernel_filter: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout_seconds: Optional[int] = 0,
        kernel_iteration_range: Optional[str] = None,
    ) -> List[ProfileResult]:
        wrapper = ROCProfV3Wrapper(timeout_seconds=timeout_seconds)
        extra_counters_path = Path(__file__).parent / "counter_defs.yaml"

        return wrapper.profile(
            command=command,
            counters=counters,
            kernel_filter=kernel_filter,
            cwd=cwd,
            kernel_iteration_range=kernel_iteration_range,
            extra_counters_path=extra_counters_path if extra_counters_path.exists() else None,
            arch=self.device_specs.arch,
        )
