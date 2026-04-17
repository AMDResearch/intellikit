"""kerncap CLI — kernel extraction and isolation tool."""

import json
import logging
import sys
from typing import Optional

import click

logger = logging.getLogger(__name__)


class _CliFormatter(logging.Formatter):
    """Logging formatter that colours warnings/errors and keeps INFO clean."""

    _YELLOW = "\033[33m"
    _RED = "\033[31m"
    _RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()

        if record.levelno >= logging.ERROR:
            prefix = f"{self._RED}ERROR{self._RESET}" if self._use_color else "ERROR"
            return f"\n  {prefix}: {msg}"

        if record.levelno >= logging.WARNING:
            prefix = f"{self._YELLOW}WARNING{self._RESET}" if self._use_color else "WARNING"
            return f"\n  {prefix}: {msg}"

        if record.levelno >= logging.INFO:
            return f"  {msg}"

        # DEBUG — include module for traceability
        return f"  DEBUG ({record.name}): {msg}"


def _setup_logging(level: int) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_CliFormatter(use_color=sys.stderr.isatty()))
    root = logging.getLogger("kerncap")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False


@click.group()
@click.version_option(package_name="kerncap")
@click.option(
    "-v", "--verbose", is_flag=True, default=False, help="Enable verbose (DEBUG) logging."
)
def main(verbose):
    """kerncap — Kernel extraction and isolation tool for HIP and Triton on AMD GPUs."""
    _setup_logging(logging.DEBUG if verbose else logging.INFO)


@main.command()
@click.argument("cmd", nargs=-1, required=True)
@click.option("--output", "-o", default=None, help="Write profile results to JSON file.")
@click.option(
    "--timeout",
    default=None,
    type=int,
    help="Maximum seconds to wait for the application (default: no limit).",
)
def profile(cmd, output, timeout):
    """Profile an application and rank kernels by execution time.

    CMD is the application command to profile (e.g., ./my_app --flag).
    """
    from kerncap.profiler import run_profile

    cmd_list = list(cmd)
    click.echo(f"Profiling: {' '.join(cmd_list)}")

    try:
        kernels = run_profile(cmd_list, output_path=output, timeout=timeout)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if not kernels:
        click.echo("No kernels found in profile.")
        return

    # Print ranked kernel table
    click.echo(f"\n{'Rank':<6}{'Kernel':<60}{'Calls':<8}{'Total (ms)':<14}{'Avg (us)':<12}{'%':<8}")
    click.echo("-" * 108)
    for i, k in enumerate(kernels[:20], 1):
        total_ms = k.total_duration_ns / 1e6
        avg_us = k.avg_duration_ns / 1e3
        click.echo(
            f"{i:<6}{k.name[:58]:<60}{k.calls:<8}"
            f"{total_ms:<14.3f}{avg_us:<12.1f}{k.percentage:<8.1f}"
        )

    if output:
        click.echo(f"\nProfile saved to {output}")


@main.command()
@click.argument("kernel_name")
@click.option("--cmd", required=True, help="Application command to run for capture.")
@click.option("--source-dir", default=None, help="Source directory to search.")
@click.option("--output", "-o", default=None, help="Output directory for reproducer.")
@click.option(
    "--language",
    type=click.Choice(["hip", "triton"]),
    default=None,
    help="Kernel language (auto-detected if omitted).",
)
@click.option(
    "--dispatch", default=-1, type=int, help="Dispatch index to capture (-1 = first match)."
)
@click.option(
    "--defines",
    "-D",
    multiple=True,
    default=(),
    help="Extra preprocessor defines for reproducer (e.g. -D GGML_USE_HIP). "
    "May be specified multiple times.",
)
@click.option(
    "--timeout",
    default=300,
    type=int,
    help="Maximum seconds to wait for the application (default: 300).",
)
def extract(kernel_name, cmd, source_dir, output, language, dispatch, defines, timeout):
    """Extract a kernel into a standalone reproducer.

    KERNEL_NAME is the kernel name (or substring) to capture.
    """
    from kerncap.extract import run_extract

    try:
        result = run_extract(
            kernel_name=kernel_name,
            cmd=cmd,
            source_dir=source_dir,
            output=output,
            language=language,
            dispatch=dispatch,
            defines=list(defines) if defines else None,
            timeout=timeout,
        )
    except Exception as e:
        click.echo(f"Extract failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"  Generated: {', '.join(result.generated_files)}")
    click.echo("\nDone.")


@main.command()
@click.argument("reproducer_dir")
@click.option(
    "--hsaco",
    default=None,
    type=click.Path(exists=True),
    help="Override HSACO file (use recompiled .hsaco).",
)
@click.option("--iterations", "-n", default=1, type=int, help="Number of kernel iterations.")
@click.option("--json", "json_output", is_flag=True, default=False, help="Output results as JSON.")
@click.option(
    "--dump-output",
    is_flag=True,
    default=False,
    help="Dump post-execution memory regions for validation.",
)
@click.option(
    "--hip-launch",
    is_flag=True,
    default=False,
    help="Use HIP runtime for kernel launch (fixes rocprofv3 conflicts).",
)
def replay(reproducer_dir, hsaco, iterations, json_output, dump_output, hip_launch):
    """Replay a captured kernel using VA-faithful HSA dispatch.

    REPRODUCER_DIR is the path to the reproducer project (containing capture/).
    """
    import os
    from kerncap import _get_replay_path

    capture_dir = os.path.join(reproducer_dir, "capture")
    if not os.path.isdir(capture_dir):
        capture_dir = reproducer_dir

    try:
        replay_bin = _get_replay_path()
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    cmd = [replay_bin, capture_dir]
    if hsaco:
        cmd.extend(["--hsaco", hsaco])
    if iterations > 1:
        cmd.extend(["--iterations", str(iterations)])
    if json_output:
        cmd.append("--json")
    if dump_output:
        cmd.append("--dump-output")
    if hip_launch:
        cmd.append("--hip-launch")

    import subprocess

    proc = subprocess.run(cmd, capture_output=not json_output, text=True)

    if not json_output and proc.stdout:
        click.echo(proc.stdout.rstrip())
    if proc.stderr:
        click.echo(proc.stderr.rstrip(), err=True)

    sys.exit(proc.returncode)


@main.command()
@click.argument("reproducer_dir")
@click.option(
    "--tolerance", "-t", default=1e-6, type=float, help="Absolute tolerance for output comparison."
)
@click.option("--rtol", default=1e-5, type=float, help="Relative tolerance for output comparison.")
@click.option(
    "--hsaco",
    default=None,
    type=click.Path(exists=True),
    help="Override HSACO file (validate a recompiled variant).",
)
def validate(reproducer_dir, tolerance, rtol, hsaco):
    """Validate a reproducer by comparing outputs to captured reference.

    REPRODUCER_DIR is the path to the reproducer project.
    """
    from kerncap.validator import validate_reproducer

    click.echo(f"Validating reproducer at {reproducer_dir} ...")
    if hsaco:
        click.echo(f"  Using HSACO: {hsaco}")

    try:
        result = validate_reproducer(
            reproducer_dir=reproducer_dir,
            tolerance=tolerance,
            rtol=rtol,
            hsaco=hsaco,
        )
    except Exception as e:
        click.echo(f"Validation error: {e}", err=True)
        sys.exit(1)

    for detail in result.details:
        click.echo(f"  {detail}")

    import math

    is_smoke_test = any("smoke test" in d for d in result.details)
    if result.passed:
        if is_smoke_test:
            click.echo("\nPASS (smoke test)")
        elif result.max_error == 0.0:
            click.echo("\nPASS")
        else:
            err_str = "nan" if math.isnan(result.max_error) else f"{result.max_error:.2e}"
            click.echo(f"\nPASS (max error: {err_str})")
    else:
        err_str = "nan" if math.isnan(result.max_error) else f"{result.max_error:.2e}"
        if is_smoke_test or result.max_error == 0.0:
            click.echo("\nFAIL")
        else:
            click.echo(f"\nFAIL (max error: {err_str})")
        sys.exit(1)


if __name__ == "__main__":
    main()
