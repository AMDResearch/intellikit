#!/usr/bin/env python3
"""
Metrix CLI - Main entry point
"""

import sys
import argparse
from pathlib import Path

from ..metrics import METRIC_PROFILES, METRIC_CATALOG
from ..metrics.catalog import list_all_metrics, list_all_profiles, get_metric_info
from .profile_cmd import profile_command
from .list_cmd import list_command
from .info_cmd import info_command


def create_parser():
    """Create argument parser"""

    parser = argparse.ArgumentParser(
        prog="metrix",
        description="Metrix - GPU Profiling. Decoded.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick memory analysis
  metrix profile --profile memory ./my_app

  # Custom metrics
  metrix profile --metrics memory.hbm_bandwidth,memory.l2_hit_rate ./my_app

  # Filter specific kernels
  metrix profile --profile memory --kernel "matmul*" ./my_app

  # List available metrics
  metrix list metrics --category memory

  # Get metric information
  metrix info metric memory.hbm_bandwidth_utilization
""",
    )

    parser.add_argument("--version", action="version", version="Metrix 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Profile command
    profile_parser = subparsers.add_parser(
        "profile",
        help="Profile a GPU application",
        description="Collect performance metrics from GPU kernels",
    )

    profile_parser.add_argument(
        "target",
        help="Target application command (e.g., ./my_app or './my_app arg1 arg2')",
    )

    profile_group = profile_parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profile",
        "-p",
        choices=list(METRIC_PROFILES.keys()),
        default=None,
        help="Use pre-defined metric profile (default: all metrics)",
    )

    profile_group.add_argument("--metrics", "-m", help="Comma-separated list of metrics to collect")

    profile_group.add_argument(
        "--time-only",
        action="store_true",
        help="Only collect timing, no hardware counters",
    )

    profile_parser.add_argument(
        "--kernel", "-k", help="Filter kernels by name pattern (supports wildcards)"
    )

    profile_parser.add_argument(
        "--top", type=int, metavar="K", help="Show only top K slowest kernels"
    )

    profile_parser.add_argument(
        "--output",
        "-o",
        help="Output file (format auto-detected: .json, .csv, .txt). If not specified, prints to stdout as text.",
    )

    profile_parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Profiling timeout in seconds (default: 60)",
    )

    profile_parser.add_argument(
        "--log",
        "-l",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default="warning",
        help="Set logging level (default: warning)",
    )

    profile_parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode - minimal output"
    )

    profile_parser.add_argument(
        "--no-counters",
        action="store_true",
        help="Don't show raw counter values in output",
    )

    profile_parser.add_argument(
        "--num-replays",
        "-n",
        type=int,
        default=10,
        metavar="N",
        help="Replay/run the application N times and aggregate results (default: 10)",
    )

    profile_parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Force aggregation by kernel name (auto-enabled when --num-replays > 1)",
    )

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List available metrics, profiles, or counters",
        description="Show available profiling options",
    )

    list_parser.add_argument(
        "item_type",
        choices=["metrics", "profiles", "counters", "devices"],
        help="Type of items to list",
    )

    list_parser.add_argument("--category", "-c", help="Filter by category (for metrics)")

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Get detailed information about a metric, profile, or counter",
        description="Show detailed information",
    )

    info_subparsers = info_parser.add_subparsers(dest="info_type", help="Information type")

    metric_info = info_subparsers.add_parser("metric", help="Metric information")
    metric_info.add_argument("name", help="Metric name")

    profile_info = info_subparsers.add_parser("profile", help="Profile information")
    profile_info.add_argument("name", help="Profile name")

    counter_info = info_subparsers.add_parser("counter", help="Counter information")
    counter_info.add_argument("name", help="Counter name")

    return parser


def main():
    """Main CLI entry point"""
    from ..logger import logger

    parser = create_parser()

    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    # If first arg is not a subcommand, assume 'profile'
    # This allows: metrix ./app instead of metrix profile ./app
    # Also allows: metrix -n 5 ./app instead of metrix profile -n 5 ./app
    if len(sys.argv) > 1 and sys.argv[1] not in [
        "profile",
        "list",
        "info",
        "--version",
        "-h",
        "--help",
    ]:
        # Insert 'profile' before any flags or target
        sys.argv.insert(1, "profile")

    args = parser.parse_args()

    # Configure logging level
    if hasattr(args, "log"):
        logger.set_level(args.log)

    # Route to appropriate command
    try:
        if args.command == "profile":
            return profile_command(args)
        elif args.command == "list":
            return list_command(args)
        elif args.command == "info":
            return info_command(args)
        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        # Show traceback if log level is debug
        if hasattr(args, "log") and args.log == "debug":
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
