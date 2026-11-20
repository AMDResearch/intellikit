"""
Auto-detect GPU architecture
"""

import subprocess
import re


def detect_gpu_arch() -> str:
    """
    Auto-detect AMD GPU architecture using rocminfo

    Returns:
        Architecture string (e.g., "gfx942")

    Raises:
        RuntimeError: If detection fails or no GPU found
    """
    try:
        result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=5)

        if result.returncode != 0:
            raise RuntimeError(f"rocminfo failed: {result.stderr}")

        # Look for "Name: gfx942" or similar
        match = re.search(r"Name:\s+(gfx\w+)", result.stdout)
        if match:
            arch = match.group(1)
            return arch

        raise RuntimeError("No AMD GPU architecture found in rocminfo output")

    except FileNotFoundError:
        raise RuntimeError("rocminfo not found on PATH")
    except subprocess.TimeoutExpired:
        raise RuntimeError("rocminfo timed out")


def detect_or_default(requested_arch: str = None) -> str:
    """
    Use the requested architecture, or detect the GPU's

    Detection failure is an error rather than a guess: profiling a GPU with
    another architecture's counter definitions yields wrong numbers, not none.
    The message is shown by every caller, including metrix profile, which has
    no --arch, so its hint must name steps that work from each of them.

    Args:
        requested_arch: User-requested architecture, or None to auto-detect

    Returns:
        Architecture string

    Raises:
        RuntimeError: If no architecture was requested and detection fails
    """
    if requested_arch:
        return requested_arch

    try:
        return detect_gpu_arch()
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not detect the GPU architecture: {str(exc).strip()}. Metrix will not guess one: "
            "make sure rocminfo runs and lists this GPU, or pass the architecture explicitly "
            "(Metrix(arch=...) in Python, --arch for metrix list and metrix info)."
        ) from exc
