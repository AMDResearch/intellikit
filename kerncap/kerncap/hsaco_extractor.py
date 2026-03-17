"""Fallback .hsaco extractor using roc-obj-extract.

Used when the runtime code object interception in libkerncap.so did not
capture the .hsaco blob (e.g., code loaded via a path we don't hook).
Extracts code objects from an ELF binary on disk.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


def extract_hsaco_from_binary(
    binary_path: str,
    kernel_symbol: str,
    gpu_arch: str,
    output_path: str,
) -> bool:
    """Extract a .hsaco from a binary using ``roc-obj-extract``.

    Parameters
    ----------
    binary_path : str
        Path to the compiled ELF binary containing embedded code objects.
    kernel_symbol : str
        Mangled kernel symbol name to search for in the extracted objects.
    gpu_arch : str
        Target GPU architecture (e.g. ``"gfx90a"``).  Used to filter
        code objects when multiple architectures are present.
    output_path : str
        Where to write the extracted ``.hsaco`` file.

    Returns
    -------
    bool
        True if extraction succeeded, False otherwise.
    """
    if not shutil.which("roc-obj-extract"):
        logger.warning("roc-obj-extract not found in PATH")
        return False

    if not os.path.isfile(binary_path):
        logger.warning("Binary not found: %s", binary_path)
        return False

    with tempfile.TemporaryDirectory(prefix="kerncap_hsaco_") as tmpdir:
        # roc-obj-extract writes .co files into the output directory
        try:
            proc = subprocess.run(
                ["roc-obj-extract", "-o", tmpdir, binary_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("roc-obj-extract failed: %s", e)
            return False

        if proc.returncode != 0:
            logger.warning("roc-obj-extract exit %d: %s", proc.returncode, proc.stderr)
            return False

        # Find .co files — prefer ones matching the target arch
        co_files = []
        for fname in os.listdir(tmpdir):
            if fname.endswith((".co", ".hsaco")):
                co_files.append(os.path.join(tmpdir, fname))

        if not co_files:
            logger.warning("roc-obj-extract produced no code objects")
            return False

        # Search for the code object containing our kernel symbol
        best = _find_matching_code_object(
            co_files,
            kernel_symbol,
            gpu_arch,
        )
        if best:
            shutil.copy2(best, output_path)
            logger.info(
                "Extracted .hsaco from binary via roc-obj-extract -> %s",
                output_path,
            )
            return True

        logger.warning(
            "Could not find kernel symbol '%s' in any extracted code object",
            kernel_symbol,
        )
        return False


def _find_matching_code_object(
    co_files: list,
    kernel_symbol: str,
    gpu_arch: str,
) -> Optional[str]:
    """Find the code object file containing the target kernel symbol.

    Uses ``nm`` to inspect symbols in each code object.  Prefers files
    whose name contains the target *gpu_arch*.
    """
    nm_cmd = shutil.which("llvm-nm") or shutil.which("nm")
    if not nm_cmd:
        # Can't inspect symbols — return the first arch-matching file
        for co in co_files:
            if gpu_arch in os.path.basename(co):
                return co
        return co_files[0] if co_files else None

    # Sort: arch-matching files first
    co_files_sorted = sorted(
        co_files,
        key=lambda f: 0 if gpu_arch in os.path.basename(f) else 1,
    )

    for co in co_files_sorted:
        try:
            proc = subprocess.run(
                [nm_cmd, co],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0 and kernel_symbol in proc.stdout:
                return co
        except (subprocess.TimeoutExpired, OSError):
            continue

    # No symbol match — return the first arch-matching file as best guess
    for co in co_files_sorted:
        if gpu_arch in os.path.basename(co):
            return co

    return co_files_sorted[0] if co_files_sorted else None
