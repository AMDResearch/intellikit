"""Kernel extraction pipeline — library interface.

Refactored from cli.py so both the CLI and the Python API can share the
same extraction logic without Click dependencies.
"""

import json
import logging
import os
import shlex
from dataclasses import dataclass, field
from typing import List, Optional

from kerncap.capturer import run_capture
from kerncap.source_finder import KernelSource, detect_language

logger = logging.getLogger(__name__)


@dataclass
class ExtractResult:
    """Result of a kernel extraction."""

    output_dir: str
    capture_dir: str
    language: Optional[str] = None
    has_source: bool = False
    generated_files: List[str] = field(default_factory=list)


def run_extract(
    kernel_name: str,
    cmd: str | list[str],
    source_dir: Optional[str] = None,
    output: Optional[str] = None,
    language: Optional[str] = None,
    dispatch: int = -1,
    defines: Optional[List[str]] = None,
    timeout: int = 300,
) -> ExtractResult:
    """Extract a kernel into a standalone reproducer.

    Runs the full pipeline: capture -> find source -> generate reproducer.

    Parameters
    ----------
    kernel_name : str
        Kernel name (or substring) to capture.
    cmd : str or list[str]
        Application command to run for capture.
    source_dir : str, optional
        Source directory to search for kernel source.
    output : str, optional
        Output directory for reproducer.  Defaults to ``./isolated/<kernel_name>``.
    language : str, optional
        Kernel language ("hip" or "triton").  Auto-detected if omitted.
    dispatch : int
        Dispatch index to capture (-1 = first match).
    defines : list[str], optional
        Extra preprocessor defines for reproducer.
    timeout : int
        Maximum seconds to wait for the application.

    Returns
    -------
    ExtractResult
    """
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = list(cmd)

    defines = defines or []
    output_dir = output or f"./isolated/{kernel_name}"
    capture_dir = os.path.join(output_dir, "capture")

    detected_lang = language
    if detected_lang is None and source_dir:
        detected_lang = detect_language(kernel_name, source_dir)
        if detected_lang == "unknown":
            detected_lang = None

    logger.info("Capturing kernel '%s' ...", kernel_name)
    run_capture(
        kernel_name=kernel_name,
        cmd=cmd_list,
        output_dir=capture_dir,
        dispatch=dispatch,
        language=detected_lang,
        timeout=timeout,
    )
    logger.info("Capture complete -> %s", capture_dir)

    return _generate_reproducer(
        kernel_name,
        capture_dir,
        output_dir,
        source_dir,
        detected_lang,
        defines,
    )


def _generate_reproducer(
    kernel_name: str,
    capture_dir: str,
    output_dir: str,
    source_dir: Optional[str],
    language: Optional[str],
    defines: List[str],
) -> ExtractResult:
    """Route to Triton or HSACO reproducer generation."""
    dispatch_path = os.path.join(capture_dir, "dispatch.json")
    meta_path = os.path.join(capture_dir, "metadata.json")

    if os.path.isfile(dispatch_path):
        with open(dispatch_path) as f:
            metadata = json.load(f)
    elif os.path.isfile(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        raise FileNotFoundError(f"No dispatch.json or metadata.json found in {capture_dir}")

    effective_lang = language or metadata.get("language")

    if effective_lang == "triton":
        return _generate_triton(
            kernel_name,
            capture_dir,
            output_dir,
            source_dir,
            language,
        )
    return _generate_hsaco(
        kernel_name,
        capture_dir,
        output_dir,
        source_dir,
        language,
        defines,
        metadata,
    )


def _generate_triton(
    kernel_name: str,
    capture_dir: str,
    output_dir: str,
    source_dir: Optional[str],
    language: Optional[str],
) -> ExtractResult:
    """Triton extract pipeline."""
    from kerncap.reproducer import generate_triton_reproducer
    from kerncap.source_finder import find_kernel_source

    kernel_src = None
    if source_dir:
        logger.info("Locating kernel source in %s ...", source_dir)
        kernel_src = find_kernel_source(
            kernel_name=kernel_name,
            source_dir=source_dir,
            language=language,
        )
        if kernel_src:
            logger.info("Found: %s (%s)", kernel_src.main_file, kernel_src.language)
        else:
            logger.warning("Kernel source not found.")

    if not kernel_src:
        raise RuntimeError("Triton reproducer requires located kernel source (use source_dir).")

    logger.info("Generating Triton reproducer ...")
    generate_triton_reproducer(
        capture_dir=capture_dir,
        kernel_source=kernel_src,
        output_dir=output_dir,
    )
    logger.info("Reproducer -> %s", output_dir)

    generated = ["reproducer.py", "capture/"]
    return ExtractResult(
        output_dir=output_dir,
        capture_dir=capture_dir,
        language="triton",
        has_source=True,
        generated_files=generated,
    )


def _generate_hsaco(
    kernel_name: str,
    capture_dir: str,
    output_dir: str,
    source_dir: Optional[str],
    language: Optional[str],
    defines: List[str],
    metadata: dict,
) -> ExtractResult:
    """HIP/HSACO extract pipeline."""
    from kerncap.reproducer import generate_hsaco_reproducer

    hsaco_path = os.path.join(capture_dir, "kernel.hsaco")
    if not os.path.isfile(hsaco_path):
        logger.warning(
            "No kernel.hsaco in capture directory. Replay will not work without a .hsaco."
        )

    kernel_src = None
    mangled_name = metadata.get("mangled_name", "")

    if source_dir:
        from kerncap.source_finder import find_kernel_source

        logger.info("Locating kernel source in %s ...", source_dir)
        kernel_src = find_kernel_source(
            kernel_name=kernel_name,
            source_dir=source_dir,
            language=language,
            extra_defines=defines if defines else None,
            mangled_name=mangled_name,
        )
        if kernel_src:
            logger.info("Found: %s (%s)", kernel_src.main_file, kernel_src.language)
            if kernel_src.translation_unit:
                logger.info("Translation unit: %s", kernel_src.translation_unit)
            if not kernel_src.compile_command:
                logger.warning(
                    "No compile command found (compile_commands.json "
                    "missing or has no entry for this file). "
                    "The 'make recompile' target will not be available."
                )
        else:
            logger.warning("Kernel source not found.")

    logger.info("Generating reproducer ...")
    generate_hsaco_reproducer(
        capture_dir=capture_dir,
        output_dir=output_dir,
        kernel_source=kernel_src,
        metadata=metadata,
    )
    logger.info("Reproducer -> %s", output_dir)

    generated = ["capture/", "Makefile"]
    if os.path.isfile(os.path.join(output_dir, "capture", "kernel.hsaco")):
        generated.append("kernel.hsaco")
    if kernel_src:
        generated.extend(["kernel_variant.cpp", "vfs.yaml"])

    return ExtractResult(
        output_dir=output_dir,
        capture_dir=capture_dir,
        language=language or "hip",
        has_source=kernel_src is not None,
        generated_files=generated,
    )
