"""
Profile command implementation - Clean backend-driven design
"""

import sys
import json
import csv
import re
from pathlib import Path
from typing import List, Dict
from io import StringIO

from ..backends import get_backend, Statistics, detect_or_default
from ..metrics import METRIC_PROFILES, METRIC_CATALOG
from ..logger import logger
from ..utils.distributed import (
    apply_rank_suffix,
    detect_distributed_context,
    normalize_command_argv,
)


def profile_command(args):
    command_argv = normalize_command_argv(_normalize_cli_target(args.target))
    command_display = " ".join(command_argv)
    dist_context = detect_distributed_context()

    """Execute profile command using clean backend API"""

    # Auto-detect architecture
    arch = detect_or_default(None)
    logger.info(f"Detected architecture: {arch}")

    # Get backend
    backend = get_backend(arch)

    # Determine what metrics to collect
    explicitly_requested = False  # Track if metrics were explicitly requested
    if args.time_only:
        # Time-only mode: no metrics needed
        metrics_to_compute = []
        mode = "timing-only"
    elif args.metrics:
        # Specific metrics requested
        metrics_to_compute = [m.strip() for m in args.metrics.split(",")]
        mode = f"custom ({len(metrics_to_compute)} metrics)"
        explicitly_requested = True  # User explicitly specified metrics via --metrics
    else:
        # Use profile or all metrics
        if args.profile is None:
            # Default: profile ALL available metrics
            metrics_to_compute = backend.get_available_metrics()
            mode = f"all metrics ({len(metrics_to_compute)} total)"
        else:
            profile_name = args.profile
            if profile_name not in METRIC_PROFILES:
                logger.error(f"Unknown profile '{profile_name}'")
                logger.error(f"Available profiles: {', '.join(METRIC_PROFILES.keys())}")
                return 1

            metrics_to_compute = METRIC_PROFILES[profile_name]["metrics"]
            mode = f"profile '{profile_name}'"

    # Check for unsupported metrics
    unsupported = {
        m: backend._unsupported_metrics[m]
        for m in metrics_to_compute
        if m in backend._unsupported_metrics
    }
    if unsupported:
        if explicitly_requested:
            # User explicitly requested unsupported metric via --metrics flag - fail with error
            metric_name = list(unsupported.keys())[0]
            reason = unsupported[metric_name]
            logger.error(
                f"ERROR: Metric '{metric_name}' is not supported on {backend.device_specs.arch}"
            )
            logger.error(f"Reason: {reason}")
            return 1
        else:
            # Metrics from profile/category - filter and warn
            for metric_name, reason in unsupported.items():
                logger.warning(
                    f"Skipping '{metric_name}' (not supported on {backend.device_specs.arch}): {reason}"
                )
            metrics_to_compute = [m for m in metrics_to_compute if m not in unsupported]

    # Log configuration
    logger.info(f"{'=' * 80}")
    logger.info(f"Metrix: {mode}")
    logger.info(f"Target: {command_display}")
    if dist_context.is_distributed:
        logger.info(
            "Distributed context: launcher=%s rank=%s/%s local_rank=%s host=%s",
            dist_context.launcher,
            dist_context.global_rank,
            dist_context.world_size,
            dist_context.local_rank,
            dist_context.hostname,
        )
    if args.num_replays > 1:
        logger.info(f"Replays: {args.num_replays}")
    if args.kernel:
        logger.info(f"Filter: {args.kernel}")
    logger.info(f"{'=' * 80}")

    # Build kernel filter (regular expression, passed through to profiler)
    kernel_filter = args.kernel if args.kernel else None

    # Profile using backend (handles multi-replay & aggregation internally!)
    try:
        if args.num_replays > 1:
            logger.info(f"Running {args.num_replays} replays...")

        backend.profile(
            command=command_argv,
            metrics=metrics_to_compute,
            num_replays=args.num_replays,
            aggregate_by_kernel=args.aggregate,
            launcher=args.launcher,
            kernel_filter=kernel_filter,
        )
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Get dispatch keys (filtering already done at rocprofv3 level)
    dispatch_keys = backend.get_dispatch_keys()
    if not dispatch_keys:
        if kernel_filter:
            logger.warning(f"No kernels matched filter '{args.kernel}'")
        else:
            logger.warning("No kernels profiled")
        return 1

    # Apply top-K filter if specified
    if args.top:
        # Sort by average duration and take top K
        dispatch_durations = []
        for key in dispatch_keys:
            duration_stats = backend._aggregated[key].get("duration_us")
            if duration_stats:
                dispatch_durations.append((key, duration_stats.avg))

        dispatch_durations.sort(key=lambda x: x[1], reverse=True)
        dispatch_keys = [key for key, _ in dispatch_durations[: args.top]]

    # Compute metrics for each dispatch
    results = {}
    for dispatch_key in dispatch_keys:
        results[dispatch_key] = {
            "duration_us": backend._aggregated[dispatch_key].get("duration_us"),
            "metrics": {},
        }

        for metric in metrics_to_compute:
            try:
                results[dispatch_key]["metrics"][metric] = backend.compute_metric_stats(
                    dispatch_key, metric
                )
            except Exception as e:
                logger.warning(f"Failed to compute {metric} for {dispatch_key}: {e}")

    # Output results
    output_path = args.output
    if output_path and dist_context.is_distributed:
        output_path = apply_rank_suffix(output_path, dist_context)
        logger.info(
            "Distributed output path for rank %s: %s", dist_context.global_rank, output_path
        )

    if output_path:
        # Detect format from file extension
        output_file = Path(output_path)
        ext = output_file.suffix.lower()

        if ext == ".json":
            _write_json_output(output_file, results, metrics_to_compute, dist_context)
        elif ext == ".csv":
            _write_csv_output(
                output_file, results, metrics_to_compute, args.aggregate, dist_context
            )
        else:
            # Default to text
            _write_text_output(
                output_file, results, metrics_to_compute, args.aggregate, dist_context
            )
    else:
        # Print to stdout
        _print_text_results(
            results, metrics_to_compute, args.aggregate, args.no_counters, dist_context
        )

    logger.info(f"Profiled {len(results)} dispatch(es)/kernel(s)")

    return 0


def _normalize_cli_target(target) -> str | list[str]:
    """Normalize argparse target (string or remainder list) into command input."""
    if isinstance(target, list):
        if target and target[0] == "--":
            return target[1:]
        return target
    return target


def _print_text_results(
    results: Dict, metrics: List[str], aggregated: bool, no_counters: bool, dist_context
):
    """Print results to stdout in human-readable format"""

    # Group metrics by category

    categories = {}
    for metric in metrics:
        if metric in METRIC_CATALOG:
            cat = METRIC_CATALOG[metric].get("category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(metric)

    # Category display names
    cat_names = {
        "memory_bandwidth": "MEMORY BANDWIDTH",
        "memory_cache": "CACHE PERFORMANCE",
        "memory_coalescing": "MEMORY COALESCING",
        "memory_lds": "LOCAL DATA SHARE (LDS)",
        "memory_atomic": "ATOMIC OPERATIONS",
    }

    for dispatch_key, data in results.items():
        print(f"\n{'─' * 80}")
        if aggregated:
            print(f"Kernel: {dispatch_key}")
        else:
            # dispatch_key may be:
            # - "dispatch_1:kernel_name"
            # - "rank_1:dispatch_1:kernel_name" (distributed)
            parts = dispatch_key.split(":")
            if (
                len(parts) >= 2
                and parts[0].startswith("rank_")
                and parts[1].startswith("dispatch_")
            ):
                dispatch_id = parts[1].replace("dispatch_", "")
                kernel_name = ":".join(parts[2:]) if len(parts) > 2 else ""
                print(f"Dispatch #{dispatch_id}: {kernel_name}")
            elif len(parts) >= 2 and parts[0].startswith("dispatch_"):
                dispatch_id = parts[0].replace("dispatch_", "")
                kernel_name = ":".join(parts[1:])
                print(f"Dispatch #{dispatch_id}: {kernel_name}")
            else:
                print(f"Kernel: {dispatch_key}")
        if dist_context.is_distributed:
            print(
                f"Rank: {dist_context.global_rank}/{dist_context.world_size} "
                f"(local={dist_context.local_rank}, host={dist_context.hostname})"
            )
        print(f"{'─' * 80}")

        # Duration
        duration = data.get("duration_us")
        if duration:
            print(f"Duration: {duration.min:.2f} - {duration.max:.2f} μs (avg={duration.avg:.2f})")

        # Metrics by category
        for cat, cat_metrics in categories.items():
            print(f"\n{cat_names.get(cat, cat.upper())}:")
            for metric in cat_metrics:
                if metric in data["metrics"]:
                    stats = data["metrics"][metric]
                    metric_def = METRIC_CATALOG[metric]
                    name = metric_def["name"]
                    unit = metric_def.get("unit", "")

                    # Log detailed stats at DEBUG level
                    logger.debug(
                        f"  {name}: min={stats.min:.2f}, max={stats.max:.2f}, avg={stats.avg:.2f}"
                    )

                    # Print average to stdout
                    print(f"  {name:45s} {stats.avg:10.2f} {unit}")


def _write_json_output(output_path: Path, results: Dict, metrics: List[str], dist_context):
    """Write results to JSON file"""
    json_data = {
        "_rank": {
            "global_rank": dist_context.global_rank,
            "local_rank": dist_context.local_rank,
            "world_size": dist_context.world_size,
            "hostname": dist_context.hostname,
            "launcher": dist_context.launcher,
        }
    }

    for dispatch_key, data in results.items():
        json_data[dispatch_key] = {
            "duration_us": {
                "min": data["duration_us"].min,
                "max": data["duration_us"].max,
                "avg": data["duration_us"].avg,
            }
            if data.get("duration_us")
            else None,
            "metrics": {},
        }

        for metric, stats in data["metrics"].items():
            json_data[dispatch_key]["metrics"][metric] = {
                "min": stats.min,
                "max": stats.max,
                "avg": stats.avg,
                "count": stats.count,
            }

    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"Results written to {output_path}")


def _write_csv_output(
    output_path: Path, results: Dict, metrics: List[str], aggregated: bool, dist_context
):
    """Write results to CSV file"""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = [
            "global_rank",
            "local_rank",
            "world_size",
            "hostname",
            "dispatch_key",
            "duration_min_us",
            "duration_max_us",
            "duration_avg_us",
        ]
        for metric in metrics:
            header.extend(
                [
                    f"{metric}_min",
                    f"{metric}_max",
                    f"{metric}_avg",
                ]
            )
        writer.writerow(header)

        # Data rows
        for dispatch_key, data in results.items():
            row = [
                dist_context.global_rank,
                dist_context.local_rank,
                dist_context.world_size,
                dist_context.hostname,
                dispatch_key,
            ]

            duration = data.get("duration_us")
            if duration:
                row.extend([duration.min, duration.max, duration.avg])
            else:
                row.extend([0, 0, 0])

            for metric in metrics:
                if metric in data["metrics"]:
                    stats = data["metrics"][metric]
                    row.extend([stats.min, stats.max, stats.avg])
                else:
                    row.extend([0, 0, 0])

            writer.writerow(row)

    logger.info(f"Results written to {output_path}")


def _write_text_output(
    output_path: Path, results: Dict, metrics: List[str], aggregated: bool, dist_context
):
    """Write results to text file"""
    buffer = StringIO()

    # Redirect print to buffer
    old_stdout = sys.stdout
    sys.stdout = buffer

    _print_text_results(results, metrics, aggregated, no_counters=False, dist_context=dist_context)

    sys.stdout = old_stdout

    with open(output_path, "w") as f:
        f.write(buffer.getvalue())

    logger.info(f"Results written to {output_path}")
