#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Samplex CLI - PC sampling for GPU kernels.
"""

import sys
import json
import argparse

from ..api import Samplex


def create_parser():
    parser = argparse.ArgumentParser(
        prog="samplex",
        description="Samplex - GPU PC Sampling. Where is my kernel stuck?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stochastic PC sampling (default, MI300+)
  samplex profile ./my_app

  # Host-trap PC sampling (MI200+)
  samplex profile --method host_trap ./my_app

  # Coarser sampling (less overhead, fewer samples)
  samplex profile --interval 4096 ./my_app

  # Filter specific kernels
  samplex profile --kernel "gemm.*" ./my_app

  # Show top 20 instructions
  samplex profile --top 20 ./my_app

  # JSON output
  samplex profile -o results.json ./my_app

  # List available PC sampling configs
  samplex list-configs
""",
    )

    parser.add_argument("--version", action="version", version="Samplex 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Profile command
    profile_parser = subparsers.add_parser(
        "profile",
        help="Run PC sampling on a GPU application",
    )

    profile_parser.add_argument(
        "target",
        help="Target application command",
    )

    profile_parser.add_argument(
        "--method", "-m",
        choices=["stochastic", "host_trap"],
        default="stochastic",
        help="Sampling method (default: stochastic). stochastic = hardware-based, cycle-accurate, MI300+. host_trap = software-based, time-based, MI200+.",
    )

    profile_parser.add_argument(
        "--interval", "-i",
        type=int,
        default=65536,
        help="Sampling interval (default: 65536). For stochastic: cycles (power of 2). For host_trap: nanoseconds.",
    )

    profile_parser.add_argument(
        "--kernel", "-k",
        help="Filter kernels by name (regex)",
    )

    profile_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top instructions to show per kernel (default: 10)",
    )

    profile_parser.add_argument(
        "--output", "-o",
        help="Output file (.json or .csv). Prints to stdout if not specified.",
    )

    profile_parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Profiling timeout in seconds (default: no timeout)",
    )

    profile_parser.add_argument(
        "--log", "-l",
        choices=["debug", "info", "warning", "error"],
        default="warning",
        help="Log level (default: warning)",
    )

    # List configs command
    subparsers.add_parser(
        "list-configs",
        help="List available PC sampling configurations",
    )

    return parser


def format_text_output(results):
    """Format results as human-readable text."""
    lines = []
    unit_label = "cycles" if results.method == "stochastic" else "ns"
    lines.append(f"Samplex PC Sampling Results ({results.method}, interval={results.interval} {unit_label})")
    lines.append(f"{'=' * 70}")
    lines.append(f"Command:    {results.command}")
    lines.append(f"Method:     {results.method}")
    lines.append(f"Samples:    {results.total_samples}")
    lines.append(f"Dispatches: {results.total_dispatches}")
    lines.append("")

    # Global opcode breakdown
    lines.append(f"Global Instruction Breakdown")
    lines.append(f"{'-' * 70}")
    for h in results.global_top_opcodes:
        lines.append(f"  {h.percentage:5.1f}%  {h.sample_count:6d}  {h.opcode}")
    lines.append("")

    # Per-kernel results
    is_stochastic = results.method == "stochastic"
    for kernel in results.kernels:
        lines.append(f"Kernel: {kernel.name}")
        lines.append(f"{'-' * 70}")
        lines.append(f"  Samples:     {kernel.total_samples}")
        lines.append(f"  Duration:    {kernel.duration_us:.1f} us")
        lines.append(f"  Full mask:   {kernel.full_mask_pct:.1f}%")
        if is_stochastic:
            lines.append(f"  Issued:      {kernel.issued_pct:.1f}%")
        if kernel.empty_instruction_count > 0:
            lines.append(f"  Holes:       {kernel.empty_instruction_count} (idle/between-wave gaps)")

        if is_stochastic and kernel.top_stall_reasons:
            lines.append(f"  Stall reasons:")
            for reason, pct in kernel.top_stall_reasons.items():
                lines.append(f"    {pct:5.1f}%  {reason}")

        lines.append(f"  Top instructions:")
        for h in kernel.top_instructions:
            instr_display = h.instruction[:60] if len(h.instruction) > 60 else h.instruction
            if is_stochastic:
                issued_tag = f" [issued={h.issued_count}, stalled={h.stalled_count}]"
            else:
                issued_tag = ""
            lines.append(f"    {h.percentage:5.1f}%  {h.sample_count:5d}  {instr_display}{issued_tag}")
        lines.append("")

    return "\n".join(lines)


def format_json_output(results):
    """Format results as JSON."""
    data = {
        "command": results.command,
        "method": results.method,
        "interval": results.interval,
        "total_samples": results.total_samples,
        "total_dispatches": results.total_dispatches,
        "global_top_opcodes": [
            {
                "opcode": h.opcode,
                "sample_count": h.sample_count,
                "percentage": round(h.percentage, 2),
            }
            for h in results.global_top_opcodes
        ],
        "kernels": [
            {
                "name": k.name,
                "total_samples": k.total_samples,
                "duration_us": round(k.duration_us, 2),
                "full_mask_pct": round(k.full_mask_pct, 2),
                "empty_instruction_count": k.empty_instruction_count,
                "issued_pct": round(k.issued_pct, 2),
                "top_stall_reasons": k.top_stall_reasons,
                "top_instructions": [
                    {
                        "opcode": h.opcode,
                        "instruction": h.instruction,
                        "sample_count": h.sample_count,
                        "percentage": round(h.percentage, 2),
                        "issued_count": h.issued_count,
                        "stalled_count": h.stalled_count,
                        "stall_reasons": h.stall_reasons if h.stall_reasons else None,
                        "instruction_types": h.instruction_types if h.instruction_types else None,
                    }
                    for h in k.top_instructions
                ],
            }
            for k in results.kernels
        ],
    }
    return json.dumps(data, indent=2)


def profile_command(args):
    """Execute the profile command."""
    sampler = Samplex(timeout_seconds=args.timeout)

    results = sampler.sample(
        command=args.target,
        interval=args.interval,
        method=args.method,
        kernel_filter=args.kernel,
        top_n=args.top,
    )

    # Format output
    if args.output:
        if args.output.endswith(".json"):
            output = format_json_output(results)
        else:
            output = format_text_output(results)

        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(format_text_output(results))

    return 0


def main():
    from ..logger import logger

    parser = create_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    # Allow: samplex ./app  (implicit profile)
    if len(sys.argv) > 1 and sys.argv[1] not in [
        "profile", "list-configs", "--version", "-h", "--help",
    ]:
        sys.argv.insert(1, "profile")

    args = parser.parse_args()

    if hasattr(args, "log"):
        logger.set_level(args.log)

    try:
        if args.command == "profile":
            return profile_command(args)
        elif args.command == "list-configs":
            sampler = Samplex()
            print(sampler.list_configs())
            return 0
        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if hasattr(args, "log") and args.log == "debug":
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
