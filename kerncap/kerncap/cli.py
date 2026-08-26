"""kerncap CLI — kernel extraction and isolation tool."""

import json
import logging
import os
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
@click.option(
    "--triton-backend",
    type=click.Choice(["hsa", "python"]),
    default="hsa",
    help=(
        "How to capture Triton kernels.  ``hsa`` (default) loads "
        "libkerncap.so via LD_PRELOAD, parses kernarg layout from the "
        "AMDGPU code-object metadata, and produces a VA-faithful capture "
        "compatible with kerncap-replay -- recommended for all new work.  "
        "``python`` is the legacy path that intercepts JITFunction.run "
        "from a sitecustomize.py hook; kept only for back-compat with "
        "captures taken before the HSA backend existed."
    ),
)
def extract(
    kernel_name, cmd, source_dir, output, language, dispatch, defines, timeout, triton_backend
):
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
            triton_backend=triton_backend,
        )
    except Exception as e:
        click.echo(f"Extract failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"  Generated: {', '.join(result.generated_files)}")
    click.echo("\nDone.")
    _print_next_steps(result)


def _print_next_steps(result) -> None:
    """Print a uniform 3-line ``edit / rebuild / verify`` block.

    Same shape across languages so a HIP cookbook reads the same way as a
    Triton one; values differ by language.
    """
    out = result.output_dir
    edit_file = _pick_edit_file(result)
    if result.language == "triton":
        rebuild = f"cd {out} && python3 reproducer.py"
    else:
        rebuild = f"cd {out} && make recompile"

    click.echo("\nNext steps:")
    if edit_file:
        click.echo(f"  edit:    {os.path.join(out, edit_file)}")
    click.echo(f"  rebuild: {rebuild}")
    click.echo(f"  verify:  kerncap validate {out}")


def _pick_edit_file(result) -> Optional[str]:
    """Choose the file the user should edit, based on what extract wrote."""
    files = result.generated_files or []
    if result.language == "triton":
        # Prefer the cleanly-named ``kernel_variant.py`` (package-source path);
        # fall back to the first non-reproducer/non-dir Python file in the
        # generated list (flat-file path: a copy of the original source).
        if "kernel_variant.py" in files:
            return "kernel_variant.py"
        for f in files:
            if f.endswith(".py") and f != "reproducer.py":
                return f
        return None
    # HIP / HSACO path
    if "kernel_variant.cpp" in files:
        return "kernel_variant.cpp"
    return None


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
    import re
    import subprocess

    from kerncap import _get_replay_path
    from kerncap.validator import format_replay_result

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

    # Capture both streams so we can suppress kerncap-replay's
    # ``Stage 0 / 0.5 / 1 / 2 / 3`` setup chatter (emitted on stderr)
    # unless the user asked for ``-v`` or the binary failed.  This makes
    # the default replay output a clean header + timing block.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    show_stderr = _is_verbose() or proc.returncode != 0

    if json_output:
        # JSON mode: stdout is machine-readable; do not append the
        # Result line and do not strip ``.kd`` (would change byte-exact
        # output downstream parsers depend on).
        if stdout:
            click.echo(stdout.rstrip())
        if stderr and show_stderr:
            click.echo(stderr.rstrip(), err=True)
        sys.exit(proc.returncode)

    if stdout:
        click.echo(_strip_kd(stdout.rstrip()))
    if stderr and show_stderr:
        click.echo(stderr.rstrip(), err=True)

    avg_us = _grep_float(stdout, r"Average GPU time:\s*([\d.eE+\-]+)\s*us")
    min_us = _grep_float(stdout, r"^Min:\s*([\d.eE+\-]+)\s*us", flags=re.MULTILINE)
    max_us = _grep_float(stdout, r"^Max:\s*([\d.eE+\-]+)\s*us", flags=re.MULTILINE)

    click.echo()
    click.echo(
        format_replay_result(
            returncode=proc.returncode,
            iterations=iterations,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
        )
    )
    sys.exit(proc.returncode)


def _grep_float(text: str, pattern: str, flags: int = 0) -> Optional[float]:
    """Pluck the first regex group as a float; return None if no match."""
    import re

    m = re.search(pattern, text, flags=flags)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (ValueError, IndexError):
        return None


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
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Show per-region byte-comparison detail (one line per output region).",
)
def validate(reproducer_dir, tolerance, rtol, hsaco, verbose):
    """Validate a reproducer by comparing outputs to captured reference.

    REPRODUCER_DIR is the path to the reproducer project.
    """
    from kerncap.validator import format_result, validate_reproducer

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
        click.echo(f"  {_strip_kd(detail)}")

    # Per-region/per-arg PASS noise (e.g. ``region_<addr>.bin: PASS (identical)``
    # for a kernel with 200 regions) is only useful on FAIL or when the user
    # explicitly asks for it via ``-v`` (subcommand) or top-level ``kerncap -v``.
    # Failure lines already live in ``details``.
    if result.region_lines and (verbose or _is_verbose() or not result.passed):
        for line in result.region_lines:
            click.echo(f"  {_strip_kd(line)}")

    click.echo()
    click.echo(format_result(result))
    if not result.passed:
        sys.exit(1)


def _is_verbose() -> bool:
    """Whether the user passed ``-v`` to the top-level kerncap command."""
    return logging.getLogger("kerncap").isEnabledFor(logging.DEBUG)


def _strip_kd(text: str) -> str:
    """Drop the HSA ``.kd`` (kernel descriptor) suffix from displayed
    kernel names.

    The on-disk artifacts (``dispatch.json``, ``metadata.json``) keep the
    raw HSA symbol; this is a presentation-only edit so a Triton user
    sees ``triton_poi_fused_relu_0`` instead of
    ``triton_poi_fused_relu_0.kd``.
    """
    import re

    return re.sub(r"(\b[A-Za-z_][A-Za-z0-9_]*)\.kd\b", r"\1", text)


if __name__ == "__main__":
    main()
