"""
Clean backend architecture - no exposed mappings!

Backends provide metric computation methods decorated with @metric.
Counter names appear EXACTLY ONCE - as function parameter names.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Sequence
from dataclasses import dataclass
from collections import defaultdict


@dataclass(frozen=True)
class DeviceSpecs:
    """Device-specific hardware specifications"""

    arch: str
    name: str

    # Compute specs
    num_cu: int = 0
    max_waves_per_cu: int = 0
    wavefront_size: int = 32
    base_clock_mhz: float = 0.0

    # Memory specs
    hbm_bandwidth_gbs: float = 0.0
    l2_bandwidth_gbs: float = 0.0
    l2_size_mb: float = 0.0
    lds_size_per_cu_kb: float = 0.0

    # Compute capabilities
    fp32_tflops: float = 0.0
    fp64_tflops: float = 0.0
    int8_tops: float = 0.0

    # Clock speeds
    boost_clock_mhz: int = 0


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
    global_rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    node_rank: int = 0
    hostname: str = ""
    launcher: str = "single"


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
        self._load_yaml_metrics_if_available()  # Load YAML metrics if available (takes precedence)
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

    def _load_yaml_metrics_if_available(self):
        """
        Load metrics from counter_defs.yaml if it exists.
        If YAML exists, it takes precedence over @metric decorators.
        """
        import yaml
        import re
        from pathlib import Path

        arch = self.device_specs.arch
        yaml_path = Path(__file__).parent / "counter_defs.yaml"

        if not yaml_path.exists():
            return

        print(f"📄 Loading YAML-based metrics from {yaml_path.name}")

        try:
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  Failed to load YAML metrics: {e}")
            return

        if not yaml_data or "rocprofiler-sdk" not in yaml_data:
            return

        # Parse YAML and collect metrics matching this architecture first
        yaml_metrics = {}
        counters_section = yaml_data["rocprofiler-sdk"].get("counters", [])

        for counter_def in counters_section:
            counter_name = counter_def.get("name")
            definitions = counter_def.get("definitions", [])

            if not definitions:
                continue

            # Find the first definition that matches this architecture
            definition = None
            for defn in definitions:
                archs = defn.get("architectures", [])
                if not archs or arch in archs:
                    definition = defn
                    break

            if definition is None:
                continue

            # Register counters: derived, reduce(), and built-in
            if "expression" in definition:
                expression = definition["expression"]

                # Check if this is a simple reduce() expression
                import re

                reduce_match = re.match(
                    r"^reduce\([A-Z_0-9]+,\s*(?:sum|max|min)\)$", expression.strip()
                )

                if reduce_match:
                    yaml_metrics[counter_name] = {
                        "counters": [counter_name],
                        "compute": lambda cn=counter_name: self._raw_data.get(cn, 0.0),
                    }
                else:
                    required_counters = self._extract_counters_from_expression(expression)
                    compute_fn = self._create_yaml_compute_function(expression, counter_name)
                    yaml_metrics[counter_name] = {
                        "counters": required_counters,
                        "compute": compute_fn,
                    }
            else:
                yaml_metrics[counter_name] = {
                    "counters": [counter_name],
                    "compute": lambda cn=counter_name: self._raw_data.get(cn, 0.0),
                }

        if not yaml_metrics:
            return

        # YAML metrics found for this arch -- replace @metric-based metrics
        self._metrics.clear()
        self._unsupported_metrics.clear()
        self._metrics.update(yaml_metrics)
        print(f"✓ Loaded {len(self._metrics)} YAML-based metrics for {arch}")

    def _extract_counters_from_expression(self, expression: str) -> List[str]:
        """Extract counter names from YAML expression"""
        import re

        counters = set()

        # Extract from reduce() calls
        for match in re.finditer(r"reduce\(([A-Z_0-9]+),\s*(?:sum|max|min)\)", expression):
            counters.add(match.group(1))

        # Extract standalone counter names
        for match in re.finditer(r"\b([A-Z][A-Z_0-9]*(?:_sum)?)\b", expression):
            counter_name = match.group(1)
            if counter_name not in ["CU_NUM"]:
                counters.add(counter_name)

        return sorted(list(counters))

    def _create_yaml_compute_function(self, expression: str, metric_name: str):
        """Create a callable that evaluates YAML expression"""
        import re

        def compute():
            namespace = dict(self._raw_data)
            namespace["CU_NUM"] = self.device_specs.num_cu

            # Replace reduce(X, op) with X_op
            processed_expr = re.sub(
                r"reduce\(([A-Z_0-9]+),\s*(sum|max|min)\)", r"\1_\2", expression
            )

            # Add safe math functions
            import math

            namespace.update(
                {
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "sqrt": math.sqrt,
                    "log": math.log,
                    "exp": math.exp,
                }
            )

            try:
                result = eval(processed_expr, {"__builtins__": {}}, namespace)
                return float(result) if result is not None else 0.0
            except (ZeroDivisionError, KeyError, NameError, TypeError):
                return 0.0

        compute._yaml_expression = expression
        compute._metric_name = metric_name
        return compute

    def get_metric_counters(self, metric: str) -> List[str]:
        """
        Get the actual hardware counter names required for a specific metric.

        This returns the architecture-specific counter names as defined in this
        backend's @metric decorated methods.

        Args:
            metric: Metric name (e.g., "memory.l2_hit_rate")

        Returns:
            List of hardware counter names required for this metric
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
        """
        counters = set()
        # duration_us comes from profiler timestamps, not rocprof - do not request it
        skip = {"duration_us"}
        for metric in metrics:
            if metric not in self._metrics:
                available = ", ".join(self.get_available_metrics())
                raise ValueError(f"Unknown metric '{metric}'. Available metrics: {available}")
            counters.update(c for c in self._metrics[metric]["counters"] if c not in skip)
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

        max_per_pass = 6
        if len(counters) <= max_per_pass:
            return [counters]

        passes: List[List[str]] = []
        for i in range(0, len(counters), max_per_pass):
            passes.append(counters[i : i + max_per_pass])

        logger.info(f"Splitting {len(counters)} counters into {len(passes)} simple passes")
        return passes

    def _compute_derived_metrics(self, kernel_results: dict) -> dict:
        """
        Compute derived metrics for all kernels after aggregation.

        This is needed when using category-based batching, where derived metrics
        (like occupancy, bandwidth utilization, hit rates) depend on counters
        from multiple categories collected in separate passes.
        """
        from ..logger import logger

        @dataclass
        class MetricStats:
            min: float
            max: float
            avg: float
            count: int

        for kernel_name, kernel_data in kernel_results.items():
            # Get all available metrics for this backend
            available_metrics = self.get_available_metrics()

            # Try to compute each derived metric
            for metric_name in available_metrics:
                # Skip if already computed or if it's a raw counter
                if metric_name in kernel_data:
                    continue

                # Skip unsupported metrics
                if metric_name in self._unsupported_metrics:
                    continue

                # Get the metric function
                metric_info = self._metrics.get(metric_name)
                if not metric_info:
                    continue

                # Extract counters and function from metric_info dict
                if isinstance(metric_info, dict):
                    required_params = metric_info.get("counters", [])
                    metric_func = metric_info.get("compute")
                    if not metric_func or not required_params:
                        continue
                else:
                    # Old-style: metric_info is the function directly
                    metric_func = metric_info
                    if hasattr(metric_func, "_metric_counters"):
                        required_params = metric_func._metric_counters
                    elif hasattr(metric_func, "_original_func"):
                        import inspect

                        sig = inspect.signature(metric_func._original_func)
                        required_params = [p for p in sig.parameters.keys() if p != "self"]
                    else:
                        continue

                # Check if all required counters are available
                missing_counters = [p for p in required_params if p not in kernel_data]
                if missing_counters:
                    continue  # Can't compute this metric

                try:
                    # Extract counter values (use avg across replays)
                    counter_values = {param: kernel_data[param].avg for param in required_params}

                    # Call the metric function
                    # If it has _original_func, call that directly with counter_values
                    # Otherwise, call the function itself (it will read from self._raw_data)
                    if hasattr(metric_func, "_original_func"):
                        derived_value = metric_func._original_func(self, **counter_values)
                    else:
                        # For dict-style metrics, the function needs self._raw_data set
                        # Save original _raw_data
                        old_raw_data = getattr(self, "_raw_data", None)
                        try:
                            # Temporarily set _raw_data to our counter values
                            self._raw_data = counter_values
                            derived_value = metric_func()
                        finally:
                            # Restore original _raw_data
                            if old_raw_data is not None:
                                self._raw_data = old_raw_data
                            elif hasattr(self, "_raw_data"):
                                delattr(self, "_raw_data")

                    # Add to kernel_data as a MetricStats object
                    # (use the same value for min/max/avg since it's derived from averages)
                    kernel_data[metric_name] = MetricStats(
                        min=derived_value,
                        max=derived_value,
                        avg=derived_value,
                        count=kernel_data[required_params[0]].count,  # Use count from first counter
                    )

                except Exception as e:
                    # If computation fails, just skip this metric
                    logger.debug(f"Could not compute {metric_name} for {kernel_name}: {e}")
                    continue

        return kernel_results

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
        command: str | Sequence[str],
        metrics: List[str],
        num_replays: int = 5,
        aggregate_by_kernel: bool = False,
        launcher: Optional[str | Sequence[str]] = None,
        kernel_filter: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout_seconds: Optional[int] = 0,
        use_kernel_iteration_range: bool = False,  # Disabled: rocprofv3 hangs with multiple counter blocks
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
            kernel_filter: Regular expression to filter kernels by name.
                Only kernels whose names match the pattern will be included
                in profiling results.
            cwd: Working directory for command execution
            timeout_seconds: Timeout in seconds for profiling (default: 0, None for no timeout)

        Returns:
            self (for chaining)
        """
        from ..logger import logger

        # Get counters needed
        counters = self.get_required_counters(metrics)

        # Split metrics into category-based batches to avoid rocprofv3 hangs
        # Group by category (memory.*, proprietary.*, etc.) for better organization
        MAX_METRICS_PER_BATCH = 6
        if len(metrics) > MAX_METRICS_PER_BATCH:
            logger.info(f"Grouping {len(metrics)} metrics by category for efficient profiling")

            # Group metrics by category (prefix before the dot)
            from collections import defaultdict

            category_groups = defaultdict(list)
            for metric in metrics:
                category = metric.split(".")[0] if "." in metric else "other"
                category_groups[category].append(metric)

            # Split large categories into sub-batches
            batches = []
            for category, category_metrics in sorted(category_groups.items()):
                if len(category_metrics) <= MAX_METRICS_PER_BATCH:
                    batches.append((category, category_metrics))
                else:
                    # Split large category into chunks
                    for i in range(0, len(category_metrics), MAX_METRICS_PER_BATCH):
                        chunk = category_metrics[i : i + MAX_METRICS_PER_BATCH]
                        batch_label = f"{category} [{i // MAX_METRICS_PER_BATCH + 1}]"
                        batches.append((batch_label, chunk))

            all_kernel_results = {}
            total_batches = len(batches)

            # Process each batch
            for batch_num, (batch_label, batch_metrics) in enumerate(batches, 1):
                sep = "=" * 60
                print(f"\n{sep}")
                print(
                    f"📊 Profiling {batch_label} ({batch_num}/{total_batches}): {len(batch_metrics)} metrics"
                )
                sep = "=" * 60
                print(f"{sep}")
                logger.info(
                    f"Profiling {batch_label} metrics ({batch_num}/{total_batches}): {len(batch_metrics)} metrics"
                )

                # Recursively call profile with batch (won't recurse since len <= MAX_METRICS_PER_BATCH)
                batch_result = self.profile(
                    command=command,
                    metrics=batch_metrics,
                    num_replays=num_replays,
                    aggregate_by_kernel=aggregate_by_kernel,
                    launcher=launcher,
                    kernel_filter=kernel_filter,
                    cwd=cwd,
                    timeout_seconds=timeout_seconds,
                    use_kernel_iteration_range=use_kernel_iteration_range,
                )

                # Merge batch results
                for kernel_name, kernel_data in batch_result._aggregated.items():
                    if kernel_name not in all_kernel_results:
                        all_kernel_results[kernel_name] = {}
                    all_kernel_results[kernel_name].update(kernel_data)

                # Show results for this batch immediately (grouped by kernel)
                print(f"\n✓ {batch_label} RESULTS:")
                for kernel_name, kernel_data in batch_result._aggregated.items():
                    print(f"  [{kernel_name}]")
                    for metric_name, metric_stats in sorted(kernel_data.items()):
                        if hasattr(metric_stats, "avg"):
                            print(f"    {metric_name}: {metric_stats.avg:.2f}")

            logger.info(f"✓ Collected all {len(metrics)} metrics across {total_batches} batches")

            # Compute derived metrics now that all counters are aggregated
            print(f"\n{'=' * 60}")
            print("📊 Computing derived metrics...")
            print("=" * 60)
            all_kernel_results = self._compute_derived_metrics(all_kernel_results)
            print("✓ Derived metrics computed\n")

            # Display derived metrics
            print("\n✓ DERIVED METRICS:")
            for kernel_name, kernel_data in all_kernel_results.items():
                derived_found = False
                for metric_name in sorted(kernel_data.keys()):
                    # Show metrics with common derived metric patterns
                    if any(
                        x in metric_name
                        for x in ["percent", "rate", "efficiency", "utilization", "bandwidth"]
                    ):
                        if not derived_found:
                            print(f"  {kernel_name}:")
                            derived_found = True
                        metric_stats = kernel_data[metric_name]
                        if hasattr(metric_stats, "avg"):
                            # Format as percentage if it's a percent metric
                            if (
                                "percent" in metric_name
                                or "rate" in metric_name
                                or "efficiency" in metric_name
                                or "utilization" in metric_name
                            ):
                                print(f"    {metric_name}: {metric_stats.avg:.2f}%")
                            else:
                                print(f"    {metric_name}: {metric_stats.avg:.2f}")
            print("")

            # Return merged results - set our _aggregated and return self
            self._aggregated = all_kernel_results
            return self

        # If metrics <= MAX_METRICS_PER_BATCH, continue with normal flow

        # Split counters into passes based on hardware compatibility
        counter_passes = self._split_counters_into_passes(counters)

        if len(counter_passes) > 1:
            logger.info(
                f"Splitting {len(counters)} counters into {len(counter_passes)} compatibility-based passes"
            )

        # Collect all replays across all passes
        all_results_by_kernel = {}

        for pass_num, pass_counters in enumerate(counter_passes, 1):
            if True:  # Always show per-pass results
                logger.info(
                    f"Pass {pass_num}/{len(counter_passes)}: collecting {len(pass_counters)} counters"
                )

            pass_results = []

            # Use kernel_iteration_range for faster profiling
            if use_kernel_iteration_range:
                iteration_range = f"[1,{num_replays}]"
                logger.info(
                    f"  Using kernel_iteration_range={iteration_range} (rocprofv3 internal iterations)"
                )
                results = self._run_rocprof(
                    command,
                    pass_counters,
                    kernel_filter,
                    cwd=cwd,
                    launcher=launcher,
                    timeout_seconds=timeout_seconds,
                    kernel_iteration_range=iteration_range,
                )
                # Tag all results with replay_id 0 since rocprofv3 handles iterations
                for r in results:
                    r.run_id = 0
                pass_results.extend(results)
            else:
                # Legacy mode: run application multiple times
                for replay_id in range(num_replays):
                    # Show progress every 10 replays or at key milestones
                    if num_replays >= 20 and (
                        replay_id == 0 or (replay_id + 1) % 10 == 0 or replay_id == num_replays - 1
                    ):
                        logger.info(f"  Replay {replay_id + 1}/{num_replays}...")

                    results = self._run_rocprof(
                        command,
                        pass_counters,
                        kernel_filter,
                        cwd=cwd,
                        launcher=launcher,
                        timeout_seconds=timeout_seconds,
                    )
                    # Tag with replay_id for debugging
                    for r in results:
                        r.run_id = replay_id
                    pass_results.extend(results)

            # Report per-pass statistics for agent visibility
            if True:  # Always show per-pass results
                # Compute statistics for this pass only
                pass_stats = self._aggregate_by_kernel_then_runs(pass_results, num_replays)
                logger.info(f"\n  Pass {pass_num}/{len(counter_passes)} Results:")
                for kernel_name, counter_stats in pass_stats.items():
                    logger.info(f"    {kernel_name}:")
                    for counter_name, stats in counter_stats.items():
                        if counter_name.startswith("_"):
                            continue
                        if counter_name == "duration_us":
                            logger.info(
                                f"      {counter_name}: {stats.avg:.2f} us (min={stats.min:.2f}, max={stats.max:.2f})"
                            )
                        else:
                            logger.info(
                                f"      {counter_name}: {stats.avg:,.0f} (min={stats.min:,.0f}, max={stats.max:,.0f})"
                            )

            # Merge counter data from this pass with previous passes
            for result in pass_results:
                # Use (kernel_name, dispatch_id, replay_id, rank) as key.
                # Including global_rank prevents dispatches from different
                # ranks being silently merged when they share the same
                # kernel_name and dispatch_id.
                key = (
                    result.kernel_name,
                    result.dispatch_id,
                    getattr(result, "run_id", 0),
                    result.global_rank,
                )

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

        counter_stats = self._aggregated[dispatch_key]

        if metric not in self._metrics:
            raise ValueError(f"Unknown metric: {metric}")

        # Compute metric using min/max/avg of each counter
        metric_min = self._compute_with_stat_type(metric, counter_stats, "min")
        metric_max = self._compute_with_stat_type(metric, counter_stats, "max")
        metric_avg = self._compute_with_stat_type(metric, counter_stats, "avg")

        # Get count from any counter (all should have same count)
        first_counter = list(counter_stats.keys())[0]
        count = counter_stats[first_counter].count

        return Statistics(min=metric_min, max=metric_max, avg=metric_avg, count=count)

    def _compute_with_stat_type(
        self, metric: str, counter_stats: Dict[str, Statistics], stat_type: str
    ) -> float:
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
        # Pass profiler duration for rate metrics (not a hardware counter)
        if "duration_us" in counter_stats:
            self._current_duration_us = getattr(counter_stats["duration_us"], stat_type)
        else:
            self._current_duration_us = 0.0

        # Call the metric's compute function (decorated method)
        return self._metrics[metric]["compute"]()

    @abstractmethod
    def _run_rocprof(
        self,
        command: str | Sequence[str],
        counters: List[str],
        kernel_filter: Optional[str] = None,
        cwd: Optional[str] = None,
        launcher: Optional[str | Sequence[str]] = None,
        timeout_seconds: Optional[int] = 0,
        kernel_iteration_range: Optional[str] = None,
    ) -> List[ProfileResult]:
        """
        Run rocprofv3 and return results

        Args:
            command: Command to profile
            counters: List of hardware counter names to collect
            kernel_filter: Optional regular expression to filter kernels by name
            cwd: Optional working directory for command execution
            timeout_seconds: Timeout in seconds for profiling (default: 0, zero or None for no timeout)

        Returns:
            List of ProfileResult objects
        """
        pass

    def _aggregate_by_dispatch_across_runs(
        self, results: List[ProfileResult]
    ) -> Dict[str, Dict[str, Statistics]]:
        """
        Aggregate by (dispatch_id, kernel_name) across runs

        Returns: {dispatch_key: {counter_name: Statistics}}
        """
        # Group by dispatch_id:kernel_name
        groups = defaultdict(list)
        for result in results:
            if result.world_size > 1:
                key = (
                    f"rank_{result.global_rank}:dispatch_{result.dispatch_id}:{result.kernel_name}"
                )
            else:
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
            if merged.world_size > 1:
                key = f"rank_{merged.global_rank}:{merged.kernel_name}"
            else:
                key = merged.kernel_name
            groups[key].append(merged)

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

        # Propagate per-run dispatch count (set by _merge_dispatches)
        num_dispatches_vals = [getattr(d, "_num_dispatches", 1) for d in dispatches]
        if num_dispatches_vals:
            stats["_num_dispatches"] = round(sum(num_dispatches_vals) / len(num_dispatches_vals))

        return stats

    def _merge_dispatches(self, dispatches: List[ProfileResult]) -> ProfileResult:
        """
        Merge multiple dispatches by summing their counters

        Used for within-run aggregation by kernel name (e.g. multi-pass profiling).
        Counters are summed (each pass contributes one subset; others are 0).
        Duration is stored as the average per dispatch so rate metrics (e.g. GFLOPS
        = flops / time) use per-run time, not total time across passes.

        Args:
            dispatches: List of ProfileResult objects for same kernel

        Returns:
            Single ProfileResult with merged counters and average duration_ns
        """
        if not dispatches:
            raise ValueError("Cannot merge empty dispatch list")

        first = dispatches[0]
        merged_counters = defaultdict(float)
        total_duration = 0

        _AVG_PATTERNS = ("Percent", "Hit", "Util", "Busy", "Occupancy", "Mean", "Rate", "Ratio")

        def _should_average(name: str) -> bool:
            return any(p in name for p in _AVG_PATTERNS)

        non_summable_counts = defaultdict(int)

        for dispatch in dispatches:
            for counter, value in dispatch.counters.items():
                merged_counters[counter] += value
                if _should_average(counter):
                    non_summable_counts[counter] += 1
            total_duration += dispatch.duration_ns

        for counter, count in non_summable_counts.items():
            if count > 0:
                merged_counters[counter] /= count

        avg_duration_ns = total_duration // len(dispatches)

        merged = ProfileResult(
            dispatch_id=first.dispatch_id,
            kernel_name=first.kernel_name,
            gpu_id=first.gpu_id,
            duration_ns=avg_duration_ns,
            grid_size=first.grid_size,
            workgroup_size=first.workgroup_size,
            counters=dict(merged_counters),
            lds_per_workgroup=first.lds_per_workgroup,
            arch_vgpr=first.arch_vgpr,
            accum_vgpr=first.accum_vgpr,
            sgpr=first.sgpr,
            global_rank=first.global_rank,
            local_rank=first.local_rank,
            world_size=first.world_size,
            node_rank=first.node_rank,
            hostname=first.hostname,
            launcher=first.launcher,
        )
        merged._num_dispatches = len(dispatches)
        return merged
