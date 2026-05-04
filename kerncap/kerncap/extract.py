"""Kernel extraction pipeline — library interface.

Refactored from cli.py so both the CLI and the Python API can share the
same extraction logic without Click dependencies.
"""

import ast
import json
import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from kerncap.capturer import run_capture
from kerncap.source_finder import KernelSource, detect_language

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source-failure classifier
# ---------------------------------------------------------------------------
#
# When ``find_kernel_source`` returns None, we want to tell the user *why* in
# a way that distinguishes user error (wrong --source-dir, wrong --language)
# from a fundamental property of the captured code object (assembly-origin,
# JIT'd, third-party blob — no source trail to follow regardless of where
# they point us).
#
# The single strongest signal is the captured HSACO itself:
#
#   * If ``llvm-dwarfdump --debug-line kernel.hsaco`` yields any
#     user-readable .cpp/.hip/.cu paths, the kernel HAS a source trail and
#     we just failed to walk it from --source-dir. (TYPE A)
#
#   * Else, if ``@triton.jit`` appears in any .py under --source-dir AND a
#     function whose name matches the kernel name carries that decorator,
#     it's a Triton kernel mis-routed via --language hip. (TYPE T)
#
#   * Else, the code object has no source trail by construction. This is
#     normal for Tensile-generated rocBLAS GEMMs, hand-written assembly,
#     JIT'd binaries, or vendor-supplied .hsaco blobs. (TYPE B)
#
# The classifier deliberately does NOT use heuristics that depend on the
# project layout (compile_commands.json existence, ``__global__`` regex
# hits in unrelated files) -- those produce both false positives and false
# negatives. DWARF-on-HSACO is a property of the captured artifact itself,
# not of where the user happened to point --source-dir.

_SOURCE_EXTENSIONS = (".cpp", ".hip", ".cu", ".cxx", ".cc", ".hpp", ".h", ".cuh", ".c")


def _llvm_dwarfdump_paths(hsaco_path: str) -> Optional[List[str]]:
    """Return user-readable source paths referenced by DWARF in *hsaco_path*.

    Returns:
        - ``None`` if dwarfdump could not run (binary missing, etc.).
        - ``[]`` if dwarfdump ran but found no debug info / no source paths.
        - A list of unique source file paths otherwise.
    """
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    candidates = [
        os.path.join(rocm, "llvm", "bin", "llvm-dwarfdump"),
        "/opt/rocm/llvm/bin/llvm-dwarfdump",
        "llvm-dwarfdump",
    ]
    proc = None
    for tool in candidates:
        try:
            proc = subprocess.run(
                [tool, "--debug-line", hsaco_path],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0:
                break
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    if proc is None or proc.returncode != 0:
        return None

    paths: List[str] = []
    seen: set = set()
    for m in re.finditer(r'name:\s*"([^"]+)"', proc.stdout):
        path = m.group(1)
        # Synthetic / non-source entries the AMDGPU back-end emits.
        if path in ("<built-in>", "<command line>") or path.startswith("<"):
            continue
        if not path.endswith(_SOURCE_EXTENSIONS):
            continue
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _hsaco_has_debug_section(hsaco_path: str) -> Optional[bool]:
    """Cheap check: does *hsaco_path* contain any ``.debug_*`` ELF section?

    Returns ``None`` on tooling error, ``True``/``False`` otherwise.
    Used to qualify TYPE B messages: HSACOs assembled without debug
    info look identical to ones stripped after-the-fact, but the
    distinction is sometimes useful context for the user.
    """
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    candidates = [
        os.path.join(rocm, "llvm", "bin", "llvm-readelf"),
        "/opt/rocm/llvm/bin/llvm-readelf",
        "llvm-readelf",
        "readelf",
    ]
    for tool in candidates:
        try:
            proc = subprocess.run(
                [tool, "-S", hsaco_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                return ".debug_" in proc.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return None


def _triton_kernel_in_tree(kernel_name: str, source_dir: str) -> bool:
    """Does *source_dir* contain a ``@triton.jit`` function matching *kernel_name*?

    Used to detect TYPE T (Triton kernel mis-routed via ``--language hip``).
    """
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r") as f:
                    content = f.read()
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
            if "@triton.jit" not in content and "@triton.autotune" not in content:
                continue
            try:
                tree = ast.parse(content, filename=fpath)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                for dec in node.decorator_list:
                    dec_name = ""
                    if isinstance(dec, ast.Name):
                        dec_name = dec.id
                    elif isinstance(dec, ast.Attribute):
                        dec_name = dec.attr
                    elif isinstance(dec, ast.Call):
                        if isinstance(dec.func, ast.Attribute):
                            dec_name = dec.func.attr
                        elif isinstance(dec.func, ast.Name):
                            dec_name = dec.func.id
                    if dec_name in ("jit", "autotune"):
                        if kernel_name in node.name or node.name in kernel_name:
                            return True
    return False


def _explain_source_not_found(
    kernel_name: str,
    capture_dir: str,
    source_dir: Optional[str],
    language: Optional[str],
) -> None:
    """Emit a classification-aware message when source location fails.

    Replaces the ambiguous ``"Kernel source not found."`` warning with one
    of three messages (TYPE A / T / B) based on the captured HSACO itself.
    """
    hsaco_path = os.path.join(capture_dir, "kernel.hsaco")

    if not os.path.isfile(hsaco_path):
        logger.warning(
            "Kernel source not found, and no kernel.hsaco in capture "
            "(cannot classify -- code object missing)."
        )
        return

    dwarf_paths = _llvm_dwarfdump_paths(hsaco_path)

    if dwarf_paths:
        # TYPE A -- source trail exists, discovery failed.
        sample = dwarf_paths[:3]
        logger.warning(
            "Kernel source not found under '%s'.\n"
            "  The captured HSACO contains DWARF debug info pointing at user "
            "source files, so a source trail DOES exist -- we just couldn't "
            "walk it from --source-dir. Try one of these paths:\n"
            "    %s\n"
            "  Or check whether --language matches the kernel's actual "
            "language (currently '%s').",
            source_dir,
            "\n    ".join(sample),
            language or "auto",
        )
        return

    # TYPE T -- check for Triton mis-routing only when user passed
    # --language hip (or omitted it) AND a source dir to scan.
    if source_dir and (language is None or language == "hip"):
        if _triton_kernel_in_tree(kernel_name, source_dir):
            logger.warning(
                "Kernel source not found under '%s'.\n"
                "  The HSACO has no DWARF source mapping, but a "
                "@triton.jit function matching '%s' exists in the source "
                "tree. This kernel is likely Triton, mis-routed via "
                "--language %s. Re-run with --language triton.",
                source_dir,
                kernel_name,
                language or "(auto)",
            )
            return

    # TYPE B -- no source trail by construction.
    has_debug = _hsaco_has_debug_section(hsaco_path)
    qualifier = ""
    if has_debug is True:
        qualifier = (
            "  NOTE: the HSACO contains some .debug_* sections but no source "
            "mapping -- it may have been built with -g but stripped after the "
            "fact, or compiled with aggressive inlining that erased line info."
        )
    elif has_debug is False:
        qualifier = (
            "  NOTE: the HSACO contains no .debug_* sections -- it was "
            "assembled or compiled without any debug info, which is normal "
            "for hand-written assembly and release builds."
        )

    logger.warning(
        "Kernel source not found, and the captured code object has no source "
        "trail to walk:\n"
        "    - no DWARF source mapping in kernel.hsaco\n"
        "    - no @triton.jit function matching '%s' in --source-dir\n"
        "  This is normal for kernels generated by external codegen tools "
        "such as Tensile (rocBLAS), hand-written GCN/MFMA assembly, "
        "JIT-compiled binaries, or third-party .hsaco blobs. There is no "
        "C/C++ source for kerncap to extract.\n"
        "%s\n"
        "  Reproducer is HSACO-only: 'make run' / 'kerncap validate' will "
        "work, but 'make recompile' is not possible because there is no "
        "source to rebuild from.",
        kernel_name,
        qualifier,
    )


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
            _explain_source_not_found(
                kernel_name=kernel_name,
                capture_dir=capture_dir,
                source_dir=source_dir,
                language=language,
            )

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
            _explain_source_not_found(
                kernel_name=kernel_name,
                capture_dir=capture_dir,
                source_dir=source_dir,
                language=language,
            )

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
