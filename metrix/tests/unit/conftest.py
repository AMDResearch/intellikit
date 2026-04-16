"""
Shared fixtures for unit tests.

Auto-detects the GPU architecture once and skips tests that request
a backend for an architecture not present on this machine.
"""

import pytest
from metrix.backends.detect import detect_gpu_arch


def _hw_arch():
    try:
        return detect_gpu_arch()
    except RuntimeError:
        return None


HW_ARCH = _hw_arch()


@pytest.fixture(autouse=True)
def skip_arch_mismatch(request):
    """Skip tests parameterized with an arch that doesn't match this GPU."""
    if HW_ARCH is None:
        return
    if "arch" in request.fixturenames:
        arch = request.getfixturevalue("arch")
        if arch != HW_ARCH:
            pytest.skip(f"requires {arch} but this machine has {HW_ARCH}")


def requires_arch(arch: str):
    """Decorator: skip a test unless the machine has the given GPU arch."""
    return pytest.mark.skipif(
        HW_ARCH != arch,
        reason=f"requires {arch} but this machine has {HW_ARCH}",
    )
