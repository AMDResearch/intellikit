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
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            raise RuntimeError(f"rocminfo failed: {result.stderr}")

        # Look for "Name: gfx942" or similar
        match = re.search(r'Name:\s+(gfx\w+)', result.stdout)
        if match:
            arch = match.group(1)
            return arch

        raise RuntimeError("No AMD GPU architecture found in rocminfo output")

    except FileNotFoundError:
        raise RuntimeError(
            "rocminfo not found. Please install ROCm or specify architecture with --arch"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("rocminfo timed out")


def detect_or_default(requested_arch: str = None) -> str:
    """
    Detect architecture or use provided/default

    Args:
        requested_arch: User-requested architecture, or None to auto-detect

    Returns:
        Architecture string
    """
    if requested_arch:
        return requested_arch

    try:
        return detect_gpu_arch()
    except RuntimeError:
        # Fall back to gfx942 if detection fails
        return "gfx942"

