"""
High-level Metrix API - Simplified interface for profiling

This provides a clean, stateful API for users who want a simple interface:
    profiler = Metrix(arch="gfx942")
    results = profiler.profile("./my_app", metrics=["memory.l2_hit_rate"])
    print(results.kernels[0].metrics["memory.l2_hit_rate"].avg)
"""

import re
from typing import List, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

from .backends import get_backend, Statistics, detect_or_default
from .backends.base import CounterBackend
from .metrics import METRIC_PROFILES, METRIC_CATALOG
from .logger import logger


@dataclass
class KernelResults:
    """
    Clean result object for a single kernel
    """

    name: str
    duration_us: Statistics
    metrics: Dict[str, Statistics]
    dispatch_count: int = 1

    @property
    def avg_time_us(self) -> float:
        """Average per-dispatch kernel duration in microseconds."""
        if self.duration_us is None:
            return 0.0
        return self.duration_us.avg / max(self.dispatch_count, 1)


@dataclass
class ProfilingResults:
    """
    Results from a profiling run
    """

    command: str
    kernels: List[KernelResults]
    total_kernels: int


class Metrix:
    """
    High-level Metrix API

    Usage:
        profiler = Metrix()
        results = profiler.profile("./my_app", metrics=["memory.l2_hit_rate"])

        for kernel in results.kernels:
            print(f"{kernel.name}: {kernel.metrics['memory.l2_hit_rate'].avg:.2f}%")
    """

    def __init__(self, arch: Optional[str] = None):
        """
        Initialize Metrix

        Args:
            arch: GPU architecture (gfx942, gfx90a) or None to auto-detect

        Note: Use Python's logging module to control verbosity (logging.INFO, logging.DEBUG, etc.)
        """
        self.arch = detect_or_default(arch)

        # Initialize backend
        self.backend = get_backend(self.arch)

        logger.info(f"Initialized for {self.backend.device_specs.arch}")

    def profile(
        self,
        command: str,
        metrics: Optional[List[str]] = None,
        profile: Optional[str] = None,
        kernel_filter: Optional[str] = None,
        time_only: bool = False,
        num_replays: int = 1,
        aggregate_by_kernel: bool = True,
        cwd: Optional[str] = None,
        timeout_seconds: Optional[int] = 0,
    ) -> ProfilingResults:
        """
        Profile a command

        Args:
            command: Command to profile (e.g., "./my_app" or "./my_app arg1 arg2")
            metrics: List of metrics to collect (e.g., ["memory.l2_hit_rate"])
            profile: Use a preset profile ("quick", "memory", etc.)
            kernel_filter: Regular expression to filter kernels by name.
                Only kernels whose names match the provided regular expression will be
                included in profiling results. All other kernel dispatches will be ignored
                by the profiler.

                Examples:
                  ``"^gemm.*"``        - kernels whose names start with "gemm"
                  ``".*attention.*"``   - kernels whose names contain "attention"
                  ``"gemm|attention"``  - kernels matching either pattern
            time_only: Only collect timing, no hardware counters
            num_replays: Number of times to replay/run the command (default: 1)
            aggregate_by_kernel: Aggregate dispatches by kernel name (default: True)
            cwd: Working directory for command execution (default: None)
            timeout_seconds: Timeout in seconds for profiling (default: 0, zero or None for no timeout)

        Returns:
            ProfilingResults object with all collected data
        """

        # Determine what to collect
        explicitly_requested = False  # Track if metrics were explicitly requested
        if time_only:
            metrics_to_compute = []
        elif metrics:
            metrics_to_compute = metrics
            explicitly_requested = True  # User explicitly specified metrics
        elif profile:
            if profile not in METRIC_PROFILES:
                raise ValueError(
                    f"Unknown profile: {profile}. Available: {list(METRIC_PROFILES.keys())}"
                )
            metrics_to_compute = METRIC_PROFILES[profile]["metrics"]
        else:
            # Default: all available metrics
            metrics_to_compute = self.backend.get_available_metrics()

        # Check for unsupported metrics (explicitly marked with a reason)
        unsupported = {
            m: self.backend._unsupported_metrics[m]
            for m in metrics_to_compute
            if m in self.backend._unsupported_metrics
        }

        # Check for unavailable metrics (no definition for this architecture)
        available = set(self.backend.get_available_metrics())
        unavailable = {
            m for m in metrics_to_compute
            if m not in available and m not in unsupported
        }

        if unsupported or unavailable:
            if explicitly_requested:
                # User explicitly requested unsupported/unavailable metric - fail with error
                if unsupported:
                    metric_name = list(unsupported.keys())[0]
                    reason = unsupported[metric_name]
                    raise ValueError(f"Metric '{metric_name}' is not supported: {reason}")
                else:
                    metric_name = next(iter(unavailable))
                    raise ValueError(
                        f"Metric '{metric_name}' is not available on {self.backend.device_specs.arch}. "
                        f"Available metrics: {', '.join(sorted(available))}"
                    )
            else:
                # Metrics from profile/category - filter and warn
                for metric_name, reason in unsupported.items():
                    logger.warning(
                        f"Skipping '{metric_name}' (not supported on {self.backend.device_specs.arch}): {reason}"
                    )
                for metric_name in unavailable:
                    logger.warning(
                        f"Skipping '{metric_name}' (not available on {self.backend.device_specs.arch})"
                    )
                metrics_to_compute = [
                    m for m in metrics_to_compute
                    if m not in unsupported and m not in unavailable
                ]

        if not metrics_to_compute and not time_only:
            logger.warning(
                f"No metrics available on {self.backend.device_specs.arch} "
                f"for the requested profile/metrics. Running in time-only mode."
            )
            time_only = True

        # Use simple kernel filter (no regex)
        rocprof_filter = kernel_filter

        logger.info(f"Profiling: {command}")
        logger.info(f"Collecting {len(metrics_to_compute)} metrics across {num_replays} replay(s)")
        if rocprof_filter:
            logger.info(f"Kernel filter: {rocprof_filter}")

        # Profile using backend (filtering at rocprofv3 level)
        logger.debug(f"Calling backend.profile with {len(metrics_to_compute)} metrics")
        self.backend.profile(
            command=command,
            metrics=metrics_to_compute,
            num_replays=num_replays,
            aggregate_by_kernel=aggregate_by_kernel,
            kernel_filter=rocprof_filter,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
        logger.debug("Backend.profile completed")

        # Get results (already filtered by rocprofv3)
        dispatch_keys = self.backend.get_dispatch_keys()

        if not dispatch_keys:
            logger.warning("No kernels profiled")
            return ProfilingResults(command=command, kernels=[], total_kernels=0)

        # Build result objects
        kernel_results = []
        for dispatch_key in dispatch_keys:
            # Get duration
            duration = self.backend._aggregated[dispatch_key].get("duration_us")

            # Compute metrics
            computed_metrics = {}
            for metric in metrics_to_compute:
                try:
                    computed_metrics[metric] = self.backend.compute_metric_stats(
                        dispatch_key, metric
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute {metric} for {dispatch_key}: {e}")

            # Extract per-run dispatch count (set by _merge_dispatches)
            dispatch_count = self.backend._aggregated[dispatch_key].get("_num_dispatches", 1)
            if not isinstance(dispatch_count, (int, float)):
                dispatch_count = 1

            # Create clean result object
            kernel_result = KernelResults(
                name=dispatch_key,
                duration_us=duration,
                metrics=computed_metrics,
                dispatch_count=int(dispatch_count),
            )
            kernel_results.append(kernel_result)

        return ProfilingResults(
            command=command, kernels=kernel_results, total_kernels=len(kernel_results)
        )

    def list_metrics(self, category: Optional[str] = None) -> List[str]:
        """
        List available metrics

        Args:
            category: Filter by category (optional)

        Returns:
            List of metric names
        """
        if category:
            return [
                name for name, defn in METRIC_CATALOG.items() if defn["category"].value == category
            ]
        return self.backend.get_available_metrics()

    def list_profiles(self) -> List[str]:
        """List available profiling profiles"""
        return list(METRIC_PROFILES.keys())

    def get_metric_info(self, metric_name: str) -> dict:
        """Get detailed information about a metric"""
        if metric_name not in METRIC_CATALOG:
            raise ValueError(f"Unknown metric: {metric_name}")
        return METRIC_CATALOG[metric_name]
