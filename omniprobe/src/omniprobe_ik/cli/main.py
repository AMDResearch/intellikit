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
Omniprobe only sees kernels compiled with its instrumentation plugin.
Build the application with the flags 'omniprobe-ik flags' prints first.

Examples:
  # Compile for memory analysis, then analyze
  hipcc -g $(omniprobe-ik flags memory) my_gpu_app.hip -o my_gpu_app
  omniprobe-ik memory ./my_gpu_app

  # Basic-block execution counts need a different plugin, so a separate build
  hipcc -g $(omniprobe-ik flags basic-blocks) my_gpu_app.hip -o my_gpu_app_bb
  omniprobe-ik basic-blocks ./my_gpu_app_bb

  # Filter to specific kernels; application arguments follow the target
  omniprobe-ik memory --kernel "matmul.*" --output out.json ./my_gpu_app input.dat
  omniprobe-ik memory --kernel "matmul.*" -- ./my_gpu_app --size 1024

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
        help="Count exactly how often each basic block executes",
        description=(
            "Count exactly how often each basic block executes (wave entries, active lanes), "
            "with source lines. Not timing: use metrix for time."
        ),
    )
    _add_common_args(bb_parser)

    # -- flags subcommand ---------------------------------------------------
    flags_parser = subparsers.add_parser(
        "flags",
        help="Print the hipcc flags that instrument kernels for an analysis",
        description=(
            "Print the hipcc flags (Omniprobe pass plugin, --offload-arch) "
            "that make kernels visible to the given analysis"
        ),
    )
    flags_parser.add_argument("analysis", choices=["memory", "basic-blocks"])
    flags_parser.add_argument(
        "--offload-arch",
        help="GPU architecture, e.g. gfx90a (default: detected with rocminfo)",
    )
    flags_parser.add_argument(
        "--rdc",
        action="store_true",
        help="Add -fgpu-rdc; only for applications that already build with relocatable device code",
    )

    # -- run subcommand (any analyzer or a bespoke handler .so) -------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run any Omniprobe analyzer, or a message handler you wrote (.so or .py)",
        description=(
            "Run omniprobe with any analyzer (Heatmap, AddressLogger, BasicBlockLogger, "
            "MemoryAnalysis, BasicBlockAnalysis), the path to a handler plugin (.so) or a "
            "Python handler module (.py), and print what it produced: the analyzer's JSON, "
            "or else what the application and handler printed (stdout)."
        ),
    )
    run_parser.add_argument(
        "analyzer", help="Analyzer name, or path to a handler .so or Python handler .py"
    )
    _add_common_args(run_parser)
    run_parser.add_argument(
        "--format", choices=["json", "csv"], default="json", help="Omniprobe log format"
    )

    # -- check subcommand ---------------------------------------------------
    check_parser = subparsers.add_parser(
        "check",
        help="Check that Omniprobe is installed and usable from here",
        description=(
            "Report the omniprobe executable, its instrumentation plugins, its Python API, "
            "the GPU architecture, and whether Omniprobe, hipcc and ROCM_PATH use the same "
            "ROCm. Exits 1 if anything is missing or mismatched."
        ),
    )
    check_parser.add_argument("--json", action="store_true", help="Print the report as JSON")

    # -- handler-flags subcommand -------------------------------------------
    subparsers.add_parser(
        "handler-flags",
        help="Print the hipcc command for building a bespoke message-handler plugin",
    )

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared between memory and basic-blocks subcommands."""
    parser.add_argument(
        "target",
        help="Application to analyze (its arguments follow it, or after --)",
    )
    parser.add_argument(
        "app_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the application",
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
        help=(
            "Which dispatches to capture (default: all). With --kernel, 'random' often "
            "captures nothing: Omniprobe draws from all dispatches, including excluded ones"
        ),
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
        command=_command(args),
        kernel_filter=args.kernel,
        dispatches=args.dispatches,
    )

    output = _format_memory_summary(result) + _format_memory_result(result)
    _write_output(output, args.output)
    return 0


def bb_command(args) -> int:
    """Execute basic-block analysis."""
    from omniprobe_ik import Omniprobe

    probe = Omniprobe()
    result = probe.analyze_basic_blocks(
        command=_command(args),
        kernel_filter=args.kernel,
        dispatches=args.dispatches,
    )

    from omniprobe_ik.api import uninstrumented_warning

    warning = uninstrumented_warning(result.uninstrumented_kernels)
    output = (f"WARNING: {warning}\n\n" if warning else "") + _format_bb_result(result)
    _write_output(output, args.output)
    return 0


def _command(args) -> str:
    """The application command line: target plus its arguments.

    Omniprobe splits the command on whitespace, so a single argument that
    itself contains spaces cannot be passed through.
    """
    app_args = [a for a in args.app_args if a != "--"] if args.app_args else []
    for arg in app_args:
        if any(ch.isspace() for ch in arg):
            raise ValueError(
                f"application argument {arg!r} contains whitespace, which Omniprobe cannot pass through"
            )
    return " ".join([args.target, *app_args])


def run_command(args) -> int:
    """Run any analyzer or handler and print its output."""
    from omniprobe_ik import run_analyzer, uninstrumented_warning

    run = run_analyzer(
        args.analyzer,
        _command(args),
        kernel_filter=args.kernel,
        dispatches=args.dispatches,
        log_format=args.format,
    )
    warning = uninstrumented_warning(run.uninstrumented_kernels)
    if warning:
        print(f"WARNING: {warning}", file=sys.stderr)
    _write_output(run.output or run.stdout, args.output)
    return 0


def handler_flags_command(args) -> int:
    """Print the build command for a bespoke handler."""
    from omniprobe_ik import handler_build_flags

    print(handler_build_flags()["compile"])
    return 0


def flags_command(args) -> int:
    """Print the compile flags for an analysis."""
    from omniprobe_ik import instrumentation_flags

    print(
        " ".join(instrumentation_flags(args.analysis, offload_arch=args.offload_arch, rdc=args.rdc))
    )
    return 0


def _format_memory_summary(result) -> str:
    """Ranked worst sites, printed before the per-kernel detail."""
    from omniprobe_ik import summarize_memory

    summary = summarize_memory(result)
    lines = [f"WARNING: {w}" for w in summary["warnings"]] + ["Top sites (worst first)"]
    for site in summary["global_sites"]:
        lines.append(
            f"  {site['source_file']}:{site['line']}:{site['column']}  {site['access_type']}  "
            f"{site['excess_per_execution']} excess cache lines per wave execution "
            f"({site['excess_cache_lines']} total, {site['kernel']})"
        )
    for site in summary["lds_sites"]:
        lines.append(
            f"  {site['source_file']}:{site['line']}:{site['column']}  LDS {site['access_type']}  "
            f"{site['conflicts_per_execution']} bank-conflict cycles per wave execution "
            f"({site['bank_conflicts']} total, {site['kernel']})"
        )
    if len(lines) == 1:
        lines.append("  none: no excess cache lines and no LDS bank conflicts")
    lines.append("")
    return "\n".join(lines) + "\n"


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

        for bb in sorted(kernel.basic_blocks, key=lambda b: b.wave_visits, reverse=True):
            loc = f"{bb.source_file}:{bb.line}"
            occupancy = bb.active_lanes / bb.wave_visits if bb.wave_visits else 0.0
            lines.append(
                f"  BB {bb.bb_id} ({loc}): {bb.wave_visits} wave entries, "
                f"{bb.active_lanes} active lanes ({occupancy:.1f} per entry)"
            )
        lines.append("")

    return "\n".join(lines)


def check_command(args) -> int:
    """Report whether Omniprobe is usable; exit 1 if not."""
    from ..api import check_installation

    report = check_installation()
    if args.json:
        import json

        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    plugins = report["plugins"]
    rows = [
        ("omniprobe executable", report["executable"]),
        ("executable runs", {True: "yes", False: "NO"}.get(report["executable_runs"])),
        ("plugin directory", report["plugin_dir"]),
        ("plugins", ", ".join(f"{a}: {'yes' if p else 'MISSING'}" for a, p in plugins.items())),
        ("Python API", report["python_api"]),
        ("GPU architecture", report["gpu_arch"]),
        ("Omniprobe's ROCm", report["omniprobe_rocm"]),
        ("hipcc's ROCm", report["compiler_rocm"]),
        ("ROCM_PATH", report["rocm_path"] or "(not set)"),
    ]
    for label, value in rows:
        print(f"{label + ':':22} {value if value else 'NOT FOUND'}")
    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    if report["ok"]:
        print("\nOK: Omniprobe is usable from this environment.")
        return 0
    print("\nProblems:")
    for problem in report["problems"]:
        print(f"  - {problem}")
    print(f"\nInstalling Omniprobe: {report['install_doc']}")
    return 1


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
        elif args.command == "flags":
            return flags_command(args)
        elif args.command == "run":
            return run_command(args)
        elif args.command == "check":
            return check_command(args)
        elif args.command == "handler-flags":
            return handler_flags_command(args)
        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
