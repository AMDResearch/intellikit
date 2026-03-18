"""
GFX1201 (RDNA4) Backend

Metrics are loaded from counter_defs.yaml.
"""

from .base import CounterBackend, DeviceSpecs, ProfileResult, Statistics
from ..profiler.rocprof_wrapper import ROCProfV3Wrapper
from pathlib import Path
from typing import List, Optional


class GFX1201Backend(CounterBackend):
    """AMD RDNA4 (gfx1201) backend."""

    def get_metric_counters(self, metric: str) -> List[str]:
        if metric not in self._metrics:
            return [metric]
        return list(self._metrics[metric]["counters"])

    def get_required_counters(self, metrics: List[str]) -> List[str]:
        counters = set()
        skip = {"duration_us"}
        for metric in metrics:
            if metric not in self._metrics:
                counters.add(metric)
            else:
                counters.update(c for c in self._metrics[metric]["counters"] if c not in skip)
        return list(counters)

    def compute_metric_stats(self, dispatch_key: str, metric: str) -> Statistics:
        if dispatch_key not in self._aggregated:
            raise KeyError(f"Unknown dispatch key: {dispatch_key}")
        counter_stats = self._aggregated[dispatch_key]
        if metric not in self._metrics:
            if metric in counter_stats:
                return counter_stats[metric]
            return Statistics(min=0.0, max=0.0, avg=0.0, count=0)
        metric_min = self._compute_with_stat_type(metric, counter_stats, "min")
        metric_max = self._compute_with_stat_type(metric, counter_stats, "max")
        metric_avg = self._compute_with_stat_type(metric, counter_stats, "avg")
        first_counter = list(counter_stats.keys())[0]
        count = counter_stats[first_counter].count
        return Statistics(min=metric_min, max=metric_max, avg=metric_avg, count=count)

    def _get_device_specs(self) -> DeviceSpecs:
        return DeviceSpecs(
            arch="gfx1201",
            name="AMD Radeon Graphics (RDNA4)",
            wavefront_size=32,
        )

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
