"""
Clean backend architecture - no exposed mappings!

Backends provide metric computation methods decorated with @metric.
Counter names appear EXACTLY ONCE - as function parameter names.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass(frozen=True)
class DeviceSpecs:
    """Device-specific hardware specifications"""

    arch: str
    name: str

    # Compute specs
    num_cu: int
    max_waves_per_cu: int
    wavefront_size: int
    base_clock_mhz: float  # MHz (base GPU clock frequency)

    # Memory specs
    hbm_bandwidth_gbs: float  # GB/s
    l2_bandwidth_gbs: float  # GB/s
    l2_size_mb: float  # MB
    lds_size_per_cu_kb: float  # KB

    # Compute capabilities
    fp32_tflops: float
    fp64_tflops: float
    int8_tops: float

    # Clock speeds
    boost_clock_mhz: int


@dataclass
class Statistics:
    """Min/max/avg statistics for a value"""

    min: float
    max: float
    avg: float
    count: int


@dataclass
class ProfileResult:
    """Single kernel dispatch profiling result"""

    dispatch_id: int
    kernel_name: str
    gpu_id: int
    duration_ns: int
    grid_size: tuple
    workgroup_size: tuple
    counters: Dict[str, float]

    # Kernel resources
    lds_per_workgroup: int = 0
    arch_vgpr: int = 0
    accum_vgpr: int = 0
    sgpr: int = 0


class CounterBackend(ABC):
    """
    Base class for architecture-specific profiling backends

    Design principles:
    1. Backends define metrics using @metric decorator
    2. Counter names appear EXACTLY ONCE (as function parameters)
    3. No exposed mappings or translation layers
    4. Base class orchestrates, derived class implements
    """

    def __init__(self):
        """Initialize backend and discover metrics"""
        self.device_specs = self._get_device_specs()
        self._metrics = {}
        self._unsupported_metrics = {}
        self._discover_metrics()
        self._raw_data = {}  # Current raw counter values (for metric computation)
        self._aggregated = {}  # Aggregated results: {dispatch_key: {counter: Statistics}}

    @abstractmethod
    def _get_device_specs(self) -> DeviceSpecs:
        """Return architecture specifications"""
        pass

    def _discover_metrics(self) -> None:
        """
        Auto-discover all @metric decorated methods and identify unsupported ones

        Populates both self._metrics (supported) and self._unsupported_metrics (unsupported)
        """
        for attr_name in dir(self):
            method = getattr(self, attr_name)
            if hasattr(method, "_metric_name"):
                name = method._metric_name
                if hasattr(method, "_unsupported_reason") and method._unsupported_reason:
                    # Mark as unsupported
                    self._unsupported_metrics[name] = method._unsupported_reason
                else:
                    # Register as available
                    self._metrics[name] = {"counters": method._metric_counters, "compute": method}

    def get_available_metrics(self) -> List[str]:
        """Get list of all metrics supported by this backend"""
        return list(self._metrics.keys())

    def get_metric_counters(self, metric: str) -> List[str]:
        """
        Get the actual hardware counter names required for a specific metric.

        This returns the architecture-specific counter names as defined in this
        backend's @metric decorated methods.

        Args:
            metric: Metric name (e.g., "memory.l2_hit_rate")

        Returns:
            List of hardware counter names required for this metric

        Raises:
            ValueError: If metric is unknown
        """
        if metric not in self._metrics:
            available = ", ".join(self.get_available_metrics())
            raise ValueError(f"Unknown metric '{metric}'. Available metrics: {available}")
        return list(self._metrics[metric]["counters"])

    def get_required_counters(self, metrics: List[str]) -> List[str]:
        """
        Get all hardware counters needed for requested metrics

        Args:
            metrics: List of metric names

        Returns:
            List of unique hardware counter names

        Raises:
            ValueError: If any metric is unknown
        """
        counters = set()
        for metric in metrics:
            if metric not in self._metrics:
                available = ", ".join(self.get_available_metrics())
                raise ValueError(f"Unknown metric '{metric}'. Available metrics: {available}")
            counters.update(self._metrics[metric]["counters"])
        return list(counters)

    def _get_counter_block(self, counter_name: str) -> str:
        """
        Extract hardware block name from counter name based on prefix.

        AMD counter names follow the pattern: BLOCK_COUNTER_NAME
        Examples: SQ_INSTS_LDS -> SQ, TCC_HIT_sum -> TCC

        Args:
            counter_name: Counter name (e.g., "SQ_INSTS_LDS", "TCC_HIT_sum")

        Returns:
            Hardware block name (e.g., "SQ", "TCC")
        """
        # Extract prefix before first underscore
        if "_" in counter_name:
            return counter_name.split("_")[0]
        return "UNKNOWN"

    def _get_counter_block_limits(self) -> Dict[str, int]:
        """
        Return per-hardware-block counter limits for this architecture.

        Override this method in derived classes to specify how many counters
        from each hardware block can be collected simultaneously.

        Returns:
            Dict mapping block_name -> max_counters_per_pass
        """
        # Default: no block limits defined. Backends that care about block-aware
        # packing should override this in their gfxXXXX.py implementation.
        return {}

    def _get_counter_groups(self, counters: List[str]) -> List[List[str]]:
        """
        Architecture-specific hook to group counters into passes.

        Default implementation uses a simple max-per-pass chunking strategy
        without any knowledge of hardware blocks. Architectures that need
        more control should override this in their gfxXXXX backend.
        """
        from ..logger import logger

        if not counters:
            return [[]]

        max_per_pass = 14  # Conservative default for generic backends
        if len(counters) <= max_per_pass:
            return [counters]

        passes: List[List[str]] = []
        for i in range(0, len(counters), max_per_pass):
            passes.append(counters[i : i + max_per_pass])

        logger.info(f"Splitting {len(counters)} counters into {len(passes)} simple passes")
        return passes

    def _split_counters_into_passes(self, counters: List[str]) -> List[List[str]]:
        """
        Split counters into multiple profiling passes.

        This method is called by the base `profile` implementation before
        invoking rocprofv3. The actual grouping strategy is delegated to
        the per-backend `_get_counter_groups` hook so that any hardware-
        specific logic lives in the gfxXXXX backends (optionally using
        helpers from `common.py`).
        """
        return self._get_counter_groups(counters)

    def profile(
        self,
        command: str,
        metrics: List[str],
        num_replays: int = 10,
        aggregate_by_kernel: bool = False,
        kernel_filter: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout_seconds: Optional[int] = 0,
    ):
        """
        Profile command with two-level aggregation and multi-pass support

        Level 1 (within-replay): Optionally merge same kernel dispatches
        Level 2 (across-replays): Aggregate same dispatch across replays

        Args:
            command: Command to profile
            metrics: List of metric names to compute
            num_replays: Number of times to replay/run the command
            aggregate_by_kernel: If True, merge dispatches with same kernel name
            kernel_filter: Regex pattern to filter kernels at rocprofv3 level
            cwd: Working directory for command execution
            timeout_seconds: Timeout in seconds for profiling (default: 0, None for no timeout)

        Returns:
            self (for chaining)
        """
        from ..logger import logger

        # Get counters needed
        counters = self.get_required_counters(metrics)

        # Split counters into passes based on hardware compatibility
        counter_passes = self._split_counters_into_passes(counters)

        if len(counter_passes) > 1:
            logger.info(f"Splitting {len(counters)} counters into {len(counter_passes)} compatibility-based passes")

        # Collect all replays across all passes
        all_results_by_kernel = {}

        for pass_num, pass_counters in enumerate(counter_passes, 1):
            if len(counter_passes) > 1:
                logger.info(f"Pass {pass_num}/{len(counter_passes)}: collecting {len(pass_counters)} counters")

            pass_results = []
            for replay_id in range(num_replays):
                results = self._run_rocprof(command, pass_counters, kernel_filter, cwd=cwd, timeout_seconds=timeout_seconds)
                # Tag with replay_id for debugging
                for r in results:
                    r.run_id = replay_id
                pass_results.extend(results)

            # Merge counter data from this pass with previous passes
            for result in pass_results:
                # Use (kernel_name, dispatch_id, replay_id) as key
                key = (result.kernel_name, result.dispatch_id, getattr(result, "run_id", 0))

                if key not in all_results_by_kernel:
                    all_results_by_kernel[key] = result
                else:
                    # Merge counter data from this pass
                    all_results_by_kernel[key].counters.update(result.counters)

        # Convert back to list
        all_results = list(all_results_by_kernel.values())

        # Aggregate based on strategy
        if aggregate_by_kernel:
            self._aggregated = self._aggregate_by_kernel_then_runs(all_results, num_replays)
        else:
            self._aggregated = self._aggregate_by_dispatch_across_runs(all_results)

        return self

    def get_dispatch_keys(self) -> List[str]:
        """Get list of all dispatch/kernel keys in aggregated results"""
        return list(self._aggregated.keys())

    def compute_metric_stats(self, dispatch_key: str, metric: str) -> Statistics:
        """
        Compute metric statistics from aggregated counter stats

        Args:
            dispatch_key: Dispatch/kernel identifier
            metric: Metric name

        Returns:
            Statistics(min, max, avg, count)
        """
        if dispatch_key not in self._aggregated:
            raise KeyError(f"Unknown dispatch key: {dispatch_key}")

        if metric not in self._metrics:
            raise ValueError(f"Unknown metric: {metric}")

        counter_stats = self._aggregated[dispatch_key]

        # Compute metric using min/max/avg of each counter
        metric_min = self._compute_with_stat_type(metric, counter_stats, "min")
        metric_max = self._compute_with_stat_type(metric, counter_stats, "max")
        metric_avg = self._compute_with_stat_type(metric, counter_stats, "avg")

        # Get count from any counter (all should have same count)
        first_counter = list(counter_stats.keys())[0]
        count = counter_stats[first_counter].count

        return Statistics(min=metric_min, max=metric_max, avg=metric_avg, count=count)

    def _compute_with_stat_type(self, metric: str, counter_stats: Dict[str, Statistics], stat_type: str) -> float:
        """
        Extract one stat type (min/max/avg) from counter stats and compute metric

        Args:
            metric: Metric name
            counter_stats: Dict of counter_name -> Statistics
            stat_type: 'min', 'max', or 'avg'

        Returns:
            Computed metric value
        """
        # Extract the requested statistic for each counter
        self._raw_data = {
            counter: getattr(counter_stats[counter], stat_type)
            for counter in self._metrics[metric]["counters"]
            if counter in counter_stats
        }

        # Call the metric's compute function (decorated method)
        return self._metrics[metric]["compute"]()

    @abstractmethod
    def _run_rocprof(
        self, command: str, counters: List[str], kernel_filter: Optional[str] = None, cwd: Optional[str] = None, timeout_seconds: Optional[int] = 0
    ) -> List[ProfileResult]:
        """
        Run rocprofv3 and return results

        Args:
            command: Command to profile
            counters: List of hardware counter names to collect
            kernel_filter: Optional regex pattern to filter kernels

        Returns:
            List of ProfileResult objects
        """
        pass

    def _aggregate_by_dispatch_across_runs(self, results: List[ProfileResult]) -> Dict[str, Dict[str, Statistics]]:
        """
        Aggregate by (dispatch_id, kernel_name) across runs

        Returns: {dispatch_key: {counter_name: Statistics}}
        """
        # Group by dispatch_id:kernel_name
        groups = defaultdict(list)
        for result in results:
            key = f"dispatch_{result.dispatch_id}:{result.kernel_name}"
            groups[key].append(result)

        # Compute stats for each group
        aggregated = {}
        for key, dispatches in groups.items():
            aggregated[key] = self._compute_counter_stats(dispatches)

        return aggregated

    def _aggregate_by_kernel_then_runs(
        self, results: List[ProfileResult], num_replays: int
    ) -> Dict[str, Dict[str, Statistics]]:
        """
        Merge same kernels within each replay, then aggregate across replays

        Returns: {kernel_name: {counter_name: Statistics}}
        """
        # Group by replay, then by kernel
        replays = defaultdict(lambda: defaultdict(list))
        for result in results:
            replay_id = getattr(result, "run_id", 0)  # Keep field name for compatibility
            replays[replay_id][result.kernel_name].append(result)

        # Merge within each replay (sum counters)
        merged_replays = []
        for replay_id, kernels in replays.items():
            for kernel_name, dispatches in kernels.items():
                merged = self._merge_dispatches(dispatches)
                merged_replays.append(merged)

        # Now aggregate merged results across replays
        groups = defaultdict(list)
        for merged in merged_replays:
            groups[merged.kernel_name].append(merged)

        aggregated = {}
        for kernel_name, dispatches in groups.items():
            aggregated[kernel_name] = self._compute_counter_stats(dispatches)

        return aggregated

    def _compute_counter_stats(self, dispatches: List[ProfileResult]) -> Dict[str, Statistics]:
        """
        Compute min/max/avg statistics for each counter across dispatches

        Args:
            dispatches: List of ProfileResult objects

        Returns:
            Dict mapping counter_name -> Statistics
        """
        counter_values = defaultdict(list)
        duration_values = []

        for dispatch in dispatches:
            for counter, value in dispatch.counters.items():
                counter_values[counter].append(value)
            duration_values.append(dispatch.duration_ns / 1000.0)  # Convert to microseconds

        stats = {}

        # Counter statistics
        for counter, values in counter_values.items():
            stats[counter] = Statistics(
                min=min(values), max=max(values), avg=sum(values) / len(values), count=len(values)
            )

        # Duration statistics
        if duration_values:
            stats["duration_us"] = Statistics(
                min=min(duration_values),
                max=max(duration_values),
                avg=sum(duration_values) / len(duration_values),
                count=len(duration_values),
            )

        return stats

    def _merge_dispatches(self, dispatches: List[ProfileResult]) -> ProfileResult:
        """
        Merge multiple dispatches by summing their counters

        Used for within-run aggregation by kernel name

        Args:
            dispatches: List of ProfileResult objects for same kernel

        Returns:
            Single ProfileResult with summed counters
        """
        if not dispatches:
            raise ValueError("Cannot merge empty dispatch list")

        first = dispatches[0]
        merged_counters = defaultdict(float)
        total_duration = 0

        for dispatch in dispatches:
            for counter, value in dispatch.counters.items():
                merged_counters[counter] += value
            total_duration += dispatch.duration_ns

        return ProfileResult(
            dispatch_id=first.dispatch_id,
            kernel_name=first.kernel_name,
            gpu_id=first.gpu_id,
            duration_ns=total_duration,
            grid_size=first.grid_size,
            workgroup_size=first.workgroup_size,
            counters=dict(merged_counters),
            lds_per_workgroup=first.lds_per_workgroup,
            arch_vgpr=first.arch_vgpr,
            accum_vgpr=first.accum_vgpr,
            sgpr=first.sgpr,
        )
