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


def _hw_metrics():
    """Get the set of available metrics on the detected hardware."""
    if HW_ARCH is None:
        return set()
    try:
        from metrix.backends import get_backend

        backend = get_backend(HW_ARCH)
        return set(backend.get_available_metrics())
    except (ValueError, RuntimeError):
        return set()


HW_METRICS = _hw_metrics()


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


def requires_metric(*metric_names: str):
    """Decorator: skip a test unless the detected GPU supports the given metric(s).

    Usage:
        @requires_metric("memory.coalescing_efficiency")
        def test_coalescing(self): ...

        @requires_metric("compute.total_flops", "compute.hbm_gflops")
        def test_flops(self): ...
    """
    missing = [m for m in metric_names if m not in HW_METRICS]
    return pytest.mark.skipif(
        HW_ARCH is None or len(missing) > 0,
        reason=(f"requires metric(s) {', '.join(missing)} but {HW_ARCH} does not support them")
        if missing
        else "no GPU detected",
    )
