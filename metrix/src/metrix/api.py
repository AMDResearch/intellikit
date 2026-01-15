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
            kernel_filter: Kernel name substring to filter
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
                raise ValueError(f"Unknown profile: {profile}. Available: {list(METRIC_PROFILES.keys())}")
            metrics_to_compute = METRIC_PROFILES[profile]["metrics"]
        else:
            # Default: all available metrics
            metrics_to_compute = self.backend.get_available_metrics()

        # Check for unsupported metrics
        unsupported = {
            m: self.backend._unsupported_metrics[m]
            for m in metrics_to_compute
            if m in self.backend._unsupported_metrics
        }
        if unsupported:
            if explicitly_requested:
                # User explicitly requested unsupported metric - fail with error
                metric_name = list(unsupported.keys())[0]
                reason = unsupported[metric_name]
                raise ValueError(f"Metric '{metric_name}' is not supported: {reason}")
            else:
                # Metrics from profile/category - filter and warn
                for metric_name, reason in unsupported.items():
                    logger.warning(
                        f"Skipping '{metric_name}' (not supported on {self.backend.device_specs.arch}): {reason}"
                    )
                metrics_to_compute = [m for m in metrics_to_compute if m not in unsupported]

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
                    computed_metrics[metric] = self.backend.compute_metric_stats(dispatch_key, metric)
                except Exception as e:
                    logger.warning(f"Failed to compute {metric} for {dispatch_key}: {e}")

            # Create clean result object
            kernel_result = KernelResults(name=dispatch_key, duration_us=duration, metrics=computed_metrics)
            kernel_results.append(kernel_result)

        return ProfilingResults(command=command, kernels=kernel_results, total_kernels=len(kernel_results))

    def list_metrics(self, category: Optional[str] = None) -> List[str]:
        """
        List available metrics

        Args:
            category: Filter by category (optional)

        Returns:
            List of metric names
        """
        if category:
            return [name for name, defn in METRIC_CATALOG.items() if defn["category"].value == category]
        return self.backend.get_available_metrics()

    def list_profiles(self) -> List[str]:
        """List available profiling profiles"""
        return list(METRIC_PROFILES.keys())

    def get_metric_info(self, metric_name: str) -> dict:
        """Get detailed information about a metric"""
        if metric_name not in METRIC_CATALOG:
            raise ValueError(f"Unknown metric: {metric_name}")
        return METRIC_CATALOG[metric_name]
