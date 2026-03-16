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
            prefix = (f"{self._RED}ERROR{self._RESET}"
                      if self._use_color else "ERROR")
            return f"\n  {prefix}: {msg}"

        if record.levelno >= logging.WARNING:
            prefix = (f"{self._YELLOW}WARNING{self._RESET}"
                      if self._use_color else "WARNING")
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
@click.option("-v", "--verbose", is_flag=True, default=False,
              help="Enable verbose (DEBUG) logging.")
def main(verbose):
    """kerncap — Kernel extraction and isolation tool for HIP and Triton on AMD GPUs."""
    _setup_logging(logging.DEBUG if verbose else logging.INFO)


@main.command()
@click.argument("cmd", nargs=-1, required=True)
@click.option("--output", "-o", default=None, help="Write profile results to JSON file.")
def profile(cmd, output):
    """Profile an application and rank kernels by execution time.

    CMD is the application command to profile (e.g., ./my_app --flag).
    """
    from kerncap.profiler import run_profile

    cmd_list = list(cmd)
    click.echo(f"Profiling: {' '.join(cmd_list)}")

    try:
        kernels = run_profile(cmd_list, output_path=output)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if not kernels:
        click.echo("No kernels found in profile.")
        return

    # Print ranked kernel table
    click.echo(
        f"\n{'Rank':<6}{'Kernel':<60}{'Calls':<8}"
        f"{'Total (ms)':<14}{'Avg (us)':<12}{'%':<8}"
    )
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
@click.option("--language", type=click.Choice(["hip", "triton"]),
              default=None, help="Kernel language (auto-detected if omitted).")
@click.option("--dispatch", default=-1, type=int,
              help="Dispatch index to capture (-1 = first match).")
@click.option("--defines", "-D", multiple=True, default=(),
              help="Extra preprocessor defines for reproducer (e.g. -D GGML_USE_HIP). "
                   "May be specified multiple times.")
def extract(kernel_name, cmd, source_dir, output, language, dispatch, defines):
    """Extract a kernel into a standalone reproducer.

    KERNEL_NAME is the kernel name (or substring) to capture.
    """
    import shlex
    from kerncap.capturer import run_capture
    from kerncap.source_finder import detect_language

    cmd_list = shlex.split(cmd)
    output_dir = output or f"./isolated/{kernel_name}"

    capture_dir = output_dir + "/capture"

    detected_lang = language
    if detected_lang is None and source_dir:
        detected_lang = detect_language(kernel_name, source_dir)
        if detected_lang == "unknown":
            detected_lang = None

    # Step 1: Capture
    click.echo(f"Capturing kernel '{kernel_name}' ...")
    try:
        run_capture(
            kernel_name=kernel_name,
            cmd=cmd_list,
            output_dir=capture_dir,
            dispatch=dispatch,
            language=detected_lang,
        )
    except Exception as e:
        click.echo(f"Capture failed: {e}", err=True)
        sys.exit(1)
    click.echo(f"  Capture complete -> {capture_dir}")

    _extract(
        kernel_name, capture_dir, output_dir, source_dir,
        detected_lang, defines,
    )

    click.echo("\nDone.")


def _extract(kernel_name, capture_dir, output_dir, source_dir,
             language, defines):
    """Extract pipeline — routes to Triton or HSACO reproducer."""
    import json
    import os

    dispatch_path = os.path.join(capture_dir, "dispatch.json")
    meta_path = os.path.join(capture_dir, "metadata.json")

    if os.path.isfile(dispatch_path):
        with open(dispatch_path, "r") as f:
            metadata = json.load(f)
    elif os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    else:
        raise FileNotFoundError(
            f"No dispatch.json or metadata.json found in {capture_dir}"
        )

    effective_lang = language or metadata.get("language")

    if effective_lang == "triton":
        _extract_triton(kernel_name, capture_dir, output_dir, source_dir,
                        language)
    else:
        _extract_hsaco(kernel_name, capture_dir, output_dir, source_dir,
                       language, defines, metadata)


def _extract_triton(kernel_name, capture_dir, output_dir, source_dir,
                    language):
    """Triton extract pipeline — generates a Python reproducer."""
    import os
    from kerncap.reproducer import generate_triton_reproducer

    kernel_src = None
    if source_dir:
        from kerncap.source_finder import find_kernel_source

        click.echo(f"Locating kernel source in {source_dir} ...")
        kernel_src = find_kernel_source(
            kernel_name=kernel_name,
            source_dir=source_dir,
            language=language,
        )
        if kernel_src:
            click.echo(f"  Found: {kernel_src.main_file} ({kernel_src.language})")
        else:
            logger.warning("Kernel source not found.")

    if not kernel_src:
        click.echo("  Error: Triton reproducer requires located kernel source "
                   "(use --source-dir).", err=True)
        sys.exit(1)

    click.echo("Generating Triton reproducer ...")
    generate_triton_reproducer(
        capture_dir=capture_dir,
        kernel_source=kernel_src,
        output_dir=output_dir,
    )
    click.echo(f"  Reproducer -> {output_dir}")

    parts = ["reproducer.py"]
    kernel_dir = os.path.dirname(kernel_src.main_file)
    pkg_init = os.path.join(kernel_dir, "__init__.py")
    if os.path.isfile(pkg_init):
        parts.append(f"{os.path.basename(kernel_dir)}/")
    else:
        parts.append("kernel source files")
    parts.append("capture/")
    click.echo(f"  Generated: {', '.join(parts)}")


def _extract_hsaco(kernel_name, capture_dir, output_dir, source_dir,
                   language, defines, metadata):
    """HIP/HSACO extract pipeline — generates capture dir + Makefile."""
    import os
    from kerncap.reproducer import generate_hsaco_reproducer

    isa_name = metadata.get("isa_name", "")
    if isa_name and "--" in isa_name:
        gpu_arch = isa_name.rsplit("--", 1)[-1]
    elif isa_name and isa_name.startswith("gfx"):
        gpu_arch = isa_name
    else:
        gpu_arch = metadata.get("gpu_arch", "gfx90a")

    hsaco_path = os.path.join(capture_dir, "kernel.hsaco")
    if not os.path.isfile(hsaco_path):
        logger.warning("No kernel.hsaco in capture directory. "
                       "Replay will not work without a .hsaco.")

    kernel_src = None
    mangled_name = metadata.get("mangled_name", "")

    if source_dir:
        from kerncap.source_finder import find_kernel_source

        click.echo(f"Locating kernel source in {source_dir} ...")
        kernel_src = find_kernel_source(
            kernel_name=kernel_name,
            source_dir=source_dir,
            language=language,
            extra_defines=list(defines) if defines else None,
            mangled_name=mangled_name,
        )
        if kernel_src:
            click.echo(f"  Found: {kernel_src.main_file} ({kernel_src.language})")
            if kernel_src.translation_unit:
                click.echo(f"  Translation unit: {kernel_src.translation_unit}")
            if not kernel_src.compile_command:
                logger.warning(
                    "No compile command found (compile_commands.json "
                    "missing or has no entry for this file). "
                    "The 'make recompile' target will not be available."
                )
        else:
            logger.warning("Kernel source not found.")

    click.echo("Generating reproducer ...")
    generate_hsaco_reproducer(
        capture_dir=capture_dir,
        output_dir=output_dir,
        kernel_source=kernel_src,
        metadata=metadata,
    )
    click.echo(f"  Reproducer -> {output_dir}")

    parts = ["capture/", "Makefile"]
    if os.path.isfile(os.path.join(output_dir, "capture", "kernel.hsaco")):
        parts.append("kernel.hsaco")
    if kernel_src:
        parts.append("kernel_variant.cpp")
        parts.append("vfs.yaml")
    click.echo(f"  Generated: {', '.join(parts)}")


@main.command()
@click.argument("reproducer_dir")
@click.option("--hsaco", default=None, type=click.Path(exists=True),
              help="Override HSACO file (use recompiled .hsaco).")
@click.option("--iterations", "-n", default=1, type=int,
              help="Number of kernel iterations.")
@click.option("--json", "json_output", is_flag=True, default=False,
              help="Output results as JSON.")
@click.option("--dump-output", is_flag=True, default=False,
              help="Dump post-execution memory regions for validation.")
@click.option("--hip-launch", is_flag=True, default=False,
              help="Use HIP runtime for kernel launch (fixes rocprofv3 conflicts).")
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
@click.option("--tolerance", "-t", default=1e-6, type=float,
              help="Absolute tolerance for output comparison.")
@click.option("--rtol", default=1e-5, type=float,
              help="Relative tolerance for output comparison.")
@click.option("--hsaco", default=None, type=click.Path(exists=True),
              help="Override HSACO file (validate a recompiled variant).")
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
