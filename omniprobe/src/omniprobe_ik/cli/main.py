#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""CLI entry point for Omniprobe IntelliKit integration."""

import argparse
import sys

from .. import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="omniprobe-ik",
        description="Omniprobe - GPU Memory and Basic-Block Analysis (IntelliKit)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Memory coalescing analysis
  omniprobe-ik memory ./my_gpu_app

  # Basic-block timing analysis
  omniprobe-ik basic-blocks ./my_gpu_app

  # Filter to specific kernels
  omniprobe-ik memory --kernel "matmul.*" ./my_gpu_app

  # Output as JSON
  omniprobe-ik memory --output results.json ./my_gpu_app
""",
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- memory subcommand --------------------------------------------------
    mem_parser = subparsers.add_parser(
        "memory",
        help="Analyze GPU memory access patterns",
        description="Detect uncoalesced global accesses, LDS bank conflicts, and excess cache lines",
    )
    _add_common_args(mem_parser)

    # -- basic-blocks subcommand --------------------------------------------
    bb_parser = subparsers.add_parser(
        "basic-blocks",
        help="Analyze basic-block ISA timing",
        description="Measure cycle counts and percentile distributions for each basic block",
    )
    _add_common_args(bb_parser)

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared between memory and basic-blocks subcommands."""
    parser.add_argument(
        "target",
        help="Target application command to analyze",
    )
    parser.add_argument(
        "--kernel",
        "-k",
        help="Filter kernels by name (ECMAScript regex)",
    )
    parser.add_argument(
        "--dispatches",
        "-d",
        choices=["all", "random", "1"],
        default="all",
        help="Which dispatches to capture (default: all)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (.json). If not specified, prints to stdout.",
    )


def memory_command(args) -> int:
    """Execute memory analysis."""
    from omniprobe_ik import Omniprobe

    probe = Omniprobe()
    result = probe.analyze_memory(
        command=args.target,
        kernel_filter=args.kernel,
        dispatches=args.dispatches,
    )

    output = _format_memory_result(result)
    _write_output(output, args.output)
    return 0


def bb_command(args) -> int:
    """Execute basic-block analysis."""
    from omniprobe_ik import Omniprobe

    probe = Omniprobe()
    result = probe.analyze_basic_blocks(
        command=args.target,
        kernel_filter=args.kernel,
        dispatches=args.dispatches,
    )

    output = _format_bb_result(result)
    _write_output(output, args.output)
    return 0


def _format_memory_result(result) -> str:
    """Format memory analysis result for display."""
    if not result.kernels:
        return "No kernels found."

    lines = []
    for kernel in result.kernels:
        lines.append(f"Kernel: {kernel.kernel_name} (dispatch {kernel.dispatch_id})")
        lines.append(f"  Memory accesses: {len(kernel.accesses)}")

        for access in kernel.accesses:
            loc = f"{access.source_file}:{access.line}:{access.column}"
            lines.append(f"  {loc}")
            lines.append(f"    Excess cache lines: {access.excess_cache_lines}")
            lines.append(f"    Bank conflicts:     {access.bank_conflicts}")
            if access.memory_space:
                lines.append(f"    Memory space:       {access.memory_space}")
            if access.access_type:
                lines.append(f"    Access type:        {access.access_type}")
            if access.execution_count:
                lines.append(f"    Execution count:    {access.execution_count}")
        lines.append("")

    return "\n".join(lines)


def _format_bb_result(result) -> str:
    """Format basic-block analysis result for display."""
    if not result.kernels:
        return "No kernels found."

    lines = []
    for kernel in result.kernels:
        lines.append(f"Kernel: {kernel.kernel_name} (dispatch {kernel.dispatch_id})")
        lines.append(f"  Basic blocks: {len(kernel.basic_blocks)}")

        for bb in kernel.basic_blocks:
            loc = f"{bb.source_file}:{bb.line}"
            lines.append(f"  BB {bb.bb_id} ({loc})")
            lines.append(f"    Cycles: min={bb.min_cycles:.1f}  max={bb.max_cycles:.1f}")
            lines.append(
                f"    Percentiles: p25={bb.p25:.1f}  p50={bb.p50:.1f}  "
                f"p75={bb.p75:.1f}  p95={bb.p95:.1f}  p99={bb.p99:.1f}"
            )
            if bb.wave_count:
                lines.append(f"    Wave count: {bb.wave_count}")
        lines.append("")

    return "\n".join(lines)


def _write_output(content: str, output_path: str | None) -> None:
    """Write output to file or stdout."""
    if output_path:
        from pathlib import Path

        Path(output_path).write_text(content)
        print(f"Results written to {output_path}", file=sys.stderr)
    else:
        print(content)


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    try:
        if args.command == "memory":
            return memory_command(args)
        elif args.command == "basic-blocks":
            return bb_command(args)
        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130

    except ImportError as e:
        print(f"\nError: {e}", file=sys.stderr)
        print(
            "Ensure Omniprobe is installed and its Python API is on PYTHONPATH.",
            file=sys.stderr,
        )
        return 1

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
