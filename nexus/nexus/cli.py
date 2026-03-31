# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Nexus CLI -- trace GPU kernels and inspect saved traces.

Usage:
    nexus run [OPTIONS] -- <command...>
    nexus show [OPTIONS] <trace.json>
    nexus list <trace.json>
"""

from __future__ import annotations

import argparse
import json
import sys


# ---------------------------------------------------------------------------
# Text formatting helpers
# ---------------------------------------------------------------------------

_BANNER_WIDTH = 80
_BANNER_CHAR = "="
_SEP_CHAR = "\u2500"  # ─


def _banner(text: str) -> str:
    border = _BANNER_CHAR * _BANNER_WIDTH
    return f"{border}\n{text}\n{border}"


def _separator() -> str:
    return _SEP_CHAR * _BANNER_WIDTH


def _format_kernel_summary(kernel) -> str:
    """One-line summary: instruction/line counts and unique source files."""
    parts = []
    parts.append(f"Assembly: {len(kernel.assembly)} instructions")
    parts.append(f"HIP: {len(kernel.hip)} lines")
    files = sorted(set(kernel.files)) if kernel.files else []
    if files:
        parts.append(f"Files: {', '.join(files)}")
    return " | ".join(parts)


def _format_assembly(kernel) -> str:
    """Format assembly listing with sequential line numbers."""
    if not kernel.assembly:
        return "  (no assembly)"
    lines = []
    width = len(str(len(kernel.assembly)))
    for i, inst in enumerate(kernel.assembly, 1):
        lines.append(f"  {i:>{width}}. {inst}")
    return "\n".join(lines)


def _format_hip(kernel) -> str:
    """Format HIP source listing with original line numbers when available."""
    if not kernel.hip:
        return "  (no HIP source)"
    lines = []
    source_lines = kernel.lines
    has_line_info = bool(source_lines) and len(source_lines) == len(kernel.hip)
    if has_line_info:
        # Use original source line numbers; 4294967295 (UINT32_MAX) means unknown
        display_nums = []
        for ln in source_lines:
            if ln == 4294967295:
                display_nums.append("?")
            else:
                display_nums.append(str(ln))
        width = max(len(n) for n in display_nums)
        for num, src in zip(display_nums, kernel.hip):
            lines.append(f"  {num:>{width}}. {src}")
    else:
        width = len(str(len(kernel.hip)))
        for i, src in enumerate(kernel.hip, 1):
            lines.append(f"  {i:>{width}}. {src}")
    return "\n".join(lines)


def _format_trace(trace, command_str: str, show_assembly: bool, show_hip: bool) -> str:
    """Format an entire trace as human-readable text."""
    parts = []
    parts.append(_banner(f"Nexus Trace: {command_str}"))
    parts.append("")
    parts.append(f"{len(trace)} kernel(s) traced.")

    for kernel in trace:
        parts.append("")
        parts.append(_separator())
        parts.append(f"Kernel: {kernel.name}")
        parts.append(f"  {_format_kernel_summary(kernel)}")
        parts.append(_separator())

        if show_assembly:
            parts.append("")
            parts.append("Assembly:")
            parts.append(_format_assembly(kernel))

        if show_hip:
            parts.append("")
            parts.append("HIP Source:")
            parts.append(_format_hip(kernel))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def _build_run_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Trace GPU kernels by running a command",
        description="Run a command with Nexus HSA interception and display or save the kernel trace.",
    )
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Save trace to JSON file",
    )
    p.add_argument(
        "-l", "--log-level",
        type=int,
        default=0,
        help="Nexus log level 0-4 (default: 0)",
    )
    p.add_argument(
        "--search-prefix",
        default=None,
        help="Extra HIP search directories (colon-separated)",
    )
    p.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Suppress text output to stdout (useful with -o)",
    )
    p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to trace (use -- before the command)",
    )


def _run_run(args: argparse.Namespace) -> int:
    command = args.command
    # Strip leading "--" that separates nexus args from the traced command
    if command and command[0] == "--":
        command = command[1:]

    if not command:
        print("error: no command specified. Usage: nexus run -- <command...>", file=sys.stderr)
        return 1

    # Deferred import so show/list work without libnexus.so
    from nexus import Nexus

    try:
        nexus = Nexus(
            log_level=args.log_level,
            extra_search_prefix=args.search_prefix,
        )
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        trace = nexus.run(command, output=args.output)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        command_str = " ".join(command)
        print(_format_trace(trace, command_str, show_assembly=True, show_hip=True))

    return 0


# ---------------------------------------------------------------------------
# Subcommand: show
# ---------------------------------------------------------------------------

def _build_show_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "show",
        help="Display a saved trace",
        description="Display a previously saved Nexus trace with formatted text output.",
    )
    p.add_argument(
        "trace_file",
        help="Path to trace JSON file",
    )
    p.add_argument(
        "-k", "--kernel",
        default=None,
        help="Filter by kernel name (substring match)",
    )
    p.add_argument(
        "--no-assembly",
        action="store_false",
        dest="assembly",
        help="Hide assembly listing",
    )
    p.add_argument(
        "--no-hip",
        action="store_false",
        dest="hip",
        help="Hide HIP source listing",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        default=False,
        help="Output raw JSON instead of formatted text",
    )


def _run_show(args: argparse.Namespace) -> int:
    from nexus import Nexus

    try:
        trace = Nexus.load(args.trace_file)
    except FileNotFoundError:
        print(f"error: trace file not found: {args.trace_file}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, Exception) as e:
        print(f"error: failed to load trace: {e}", file=sys.stderr)
        return 1

    # Filter kernels if requested
    if args.kernel:
        filtered = [k for k in trace if args.kernel in k.name]
    else:
        filtered = list(trace)

    if args.json_output:
        data = {k.name: k._data for k in filtered}
        json.dump(data, sys.stdout, indent=2)
        print()
        return 0

    # Build formatted output for filtered kernels
    # Create a lightweight trace-like wrapper for formatting
    parts = []
    parts.append(_banner(f"Nexus Trace: {args.trace_file}"))
    parts.append("")
    parts.append(f"{len(filtered)} kernel(s)" + (f" matching '{args.kernel}'" if args.kernel else "") + ".")

    for kernel in filtered:
        parts.append("")
        parts.append(_separator())
        parts.append(f"Kernel: {kernel.name}")
        parts.append(f"  {_format_kernel_summary(kernel)}")
        parts.append(_separator())

        if args.assembly:
            parts.append("")
            parts.append("Assembly:")
            parts.append(_format_assembly(kernel))

        if args.hip:
            parts.append("")
            parts.append("HIP Source:")
            parts.append(_format_hip(kernel))

    print("\n".join(parts))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------

def _build_list_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "list",
        help="List kernel names from a trace (pipe-friendly)",
        description="Print kernel names from a saved trace, one per line.",
    )
    p.add_argument(
        "trace_file",
        help="Path to trace JSON file",
    )


def _run_list(args: argparse.Namespace) -> int:
    from nexus import Nexus

    try:
        trace = Nexus.load(args.trace_file)
    except FileNotFoundError:
        print(f"error: trace file not found: {args.trace_file}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, Exception) as e:
        print(f"error: failed to load trace: {e}", file=sys.stderr)
        return 1

    for name in trace.kernel_names:
        print(name)

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="Nexus: GPU kernel tracer for AMD ROCm",
    )
    subparsers = parser.add_subparsers(dest="subcommand")
    _build_run_parser(subparsers)
    _build_show_parser(subparsers)
    _build_list_parser(subparsers)

    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "run": _run_run,
        "show": _run_show,
        "list": _run_list,
    }
    sys.exit(handlers[args.subcommand](args))


if __name__ == "__main__":
    main()
