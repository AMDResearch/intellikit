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
    triton_backend: str = "hsa",
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
    triton_backend : str
        Either ``"hsa"`` (default; HSA-layer capture using AMDGPU
        code-object metadata, VA-faithful, replayable via
        ``kerncap-replay``) or ``"python"`` (legacy path that intercepts
        ``JITFunction.run`` from a sitecustomize.py hook -- kept for
        back-compat with pre-HSA captures).  Only consulted when
        *language* (or the auto-detected language) is ``"triton"``.

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
        triton_backend=triton_backend,
    )
    logger.info("Capture complete -> %s", capture_dir)

    return _generate_reproducer(
        kernel_name,
        capture_dir,
        output_dir,
        source_dir,
        detected_lang,
        defines,
        triton_backend=triton_backend,
    )


def _generate_reproducer(
    kernel_name: str,
    capture_dir: str,
    output_dir: str,
    source_dir: Optional[str],
    language: Optional[str],
    defines: List[str],
    triton_backend: str = "hsa",
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

    # ``--triton-backend hsa``: capture is HSA-layer (byte-faithful), but
    # the reproducer is still a Python script that re-JITs through Triton
    # so the user's edit-recompile-validate loop is preserved.  If the
    # editable reproducer cannot be built (e.g. source not located, or the
    # compile shim was bypassed), fall back to the HIP harness so at least
    # ``kerncap-replay`` works on the captured artifacts.
    if effective_lang == "triton" and triton_backend == "hsa":
        try:
            return _generate_triton_from_hsa(
                kernel_name,
                capture_dir,
                output_dir,
                source_dir,
                language,
                metadata,
            )
        except _TritonHsaReproducerUnavailable as e:
            logger.warning(
                "Editable Triton reproducer unavailable (%s); "
                "falling back to HIP harness from the captured HSACO. "
                "`kerncap-replay` will still work; the edit-recompile-validate "
                "loop will not.",
                e,
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


class _TritonHsaReproducerUnavailable(RuntimeError):
    """Raised when the HSA-backed Triton reproducer cannot be generated.

    Triggers a fallback to the HIP harness in ``_generate_reproducer``.
    """


def _generate_triton_from_hsa(
    kernel_name: str,
    capture_dir: str,
    output_dir: str,
    source_dir: Optional[str],
    language: Optional[str],
    metadata: dict,
) -> ExtractResult:
    """HSA-captured + Python-editable Triton reproducer pipeline.

    Capture artifacts come from the HSA layer (byte-faithful kernarg
    buffer, memory regions, HSACO).  The reproducer is a runnable
    Python script that imports the original ``@triton.jit`` function
    and dispatches it with reconstructed args -- editing the source
    file and rerunning triggers Triton's own re-JIT.
    """
    import json as _json

    from kerncap.reproducer import generate_triton_hsa_reproducer
    from kerncap.source_finder import find_kernel_source

    # If the user did not pass --source-dir, fall back to the source
    # snapshots the compile shim wrote to capture/triton_sources/.
    effective_source_dir = source_dir
    snapshot_dir = os.path.join(capture_dir, "triton_sources")
    if not effective_source_dir and os.path.isdir(snapshot_dir):
        effective_source_dir = snapshot_dir
        logger.info("Using compile-shim snapshot at %s as source root", snapshot_dir)

    # Prefer the source_file recorded in name_map.json (the @triton.jit
    # function's actual file) for unambiguous source resolution.
    name_map_path = os.path.join(capture_dir, "name_map.json")
    recorded_source: Optional[str] = None
    if os.path.isfile(name_map_path):
        try:
            with open(name_map_path) as f:
                rows = _json.load(f)
            for row in rows:
                if row.get("user_name") == kernel_name or kernel_name in row.get("user_name", ""):
                    recorded_source = row.get("source_snapshot") or row.get("source_file")
                    if recorded_source:
                        break
        except (OSError, ValueError):
            pass

    if recorded_source and os.path.isfile(recorded_source) and not effective_source_dir:
        effective_source_dir = os.path.dirname(recorded_source)
        logger.info(
            "Using recorded source file %s (no --source-dir passed)",
            recorded_source,
        )

    if not effective_source_dir:
        raise _TritonHsaReproducerUnavailable(
            "no source directory and the compile shim did not snapshot a source file"
        )

    logger.info("Locating kernel source in %s ...", effective_source_dir)
    kernel_src = find_kernel_source(
        kernel_name=kernel_name,
        source_dir=effective_source_dir,
        language=language or "triton",
    )
    if not kernel_src:
        raise _TritonHsaReproducerUnavailable(
            f"kernel source for '{kernel_name}' not found under {effective_source_dir}"
        )
    logger.info("Found: %s (%s)", kernel_src.main_file, kernel_src.language)

    # The HSA-side capture must have written a name_map row for the
    # kernel; otherwise we have no signature/constexprs and cannot
    # author a correct reproducer.
    if not os.path.isfile(name_map_path):
        raise _TritonHsaReproducerUnavailable(
            "no name_map.json in capture (compile shim did not fire -- "
            "likely a Triton/Inductor cache hit; rerun with "
            "KERNCAP_NO_CLEAR_TRITON_CACHE unset)"
        )

    logger.info("Generating editable Triton reproducer ...")
    generate_triton_hsa_reproducer(
        capture_dir=capture_dir,
        kernel_source=kernel_src,
        output_dir=output_dir,
    )
    logger.info("Reproducer -> %s", output_dir)

    # Build the "Generated:" list from what is actually on disk.  The
    # standalone-vs-flat-file branching inside ``generate_triton_hsa_reproducer``
    # writes either ``kernel_variant.py`` (package source) or copies of
    # the original source files (flat-file source); a hardcoded list
    # would lie in the latter case (see audit on plan-fa5cb76b).
    generated = ["reproducer.py", "capture/", "reference_output/"]
    if os.path.isfile(os.path.join(output_dir, "kernel_variant.py")):
        generated.insert(1, "kernel_variant.py")
    else:
        copied_sources = sorted(
            os.path.basename(f)
            for f in (kernel_src.source_files or [])
            if os.path.isfile(os.path.join(output_dir, os.path.basename(f)))
        )
        generated[1:1] = copied_sources

    return ExtractResult(
        output_dir=output_dir,
        capture_dir=capture_dir,
        language="triton",
        has_source=True,
        generated_files=generated,
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
            # 'make recompile' only applies to HIP/C++ sources where a
            # hipcc/clang++ invocation was recorded in compile_commands.json.
            # Triton kernels are JIT-compiled by Triton itself, so there is
            # no compile command to record and no recompile step to expose.
            if kernel_src.language != "triton" and not kernel_src.compile_command:
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
