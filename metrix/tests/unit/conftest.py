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


def requires_cdna():
    """Decorator: skip a test unless the machine has a CDNA GPU (gfx9xx)."""
    return pytest.mark.skipif(
        HW_ARCH is None or not HW_ARCH.startswith("gfx9"),
        reason=f"requires CDNA (gfx9xx) but this machine has {HW_ARCH}",
    )


def requires_metric(*metric_names: str):
    """Decorator: skip a test unless the detected GPU supports the given metric(s).

    Usage:
        @requires_metric("memory.coalescing_efficiency")
        def test_coalescing(self): ...

        @requires_metric("compute.total_flops", "compute.hbm_gflops")
        def test_flops(self): ...
    """
    if HW_ARCH is None:
        return pytest.mark.skipif(True, reason="no GPU detected")
    missing = [m for m in metric_names if m not in HW_METRICS]
    return pytest.mark.skipif(
        len(missing) > 0,
        reason=f"requires metric(s) {', '.join(missing)} but {HW_ARCH} does not support them",
    )


# --------------------------------------------------------------------------
# GPU-free test doubles
# --------------------------------------------------------------------------

DEFAULT_FAKE_METRIC = "memory.hbm_bandwidth_utilization"


def fake_stats(avg: float = 50.0, unit: str = "%"):
    """A real Statistics instance, not a mock."""
    from metrix.backends import Statistics

    return Statistics(min=avg / 2, max=avg * 2, avg=avg, count=3, unit=unit)


class FakeDeviceSpecs:
    def __init__(self, arch: str = "gfx942"):
        self.arch = arch


class FakeBackend:
    """Minimal stand-in for a CounterBackend.

    Implements only the surface the CLI's ``profile_command`` and the
    ``Metrix.profile`` API actually touch, so both can be exercised without a
    GPU.
    """

    def __init__(self, dispatch_keys=None, unsupported=None, available=None, arch="gfx942"):
        self.device_specs = FakeDeviceSpecs(arch)
        self._unsupported_metrics = dict(unsupported or {})
        self._available = list(available) if available is not None else [DEFAULT_FAKE_METRIC]
        self._keys = ["dispatch_1:gemm_kernel"] if dispatch_keys is None else dispatch_keys
        self._aggregated = {
            key: {"duration_us": fake_stats(100.0 + i * 50, "us")}
            for i, key in enumerate(self._keys)
        }
        self.profile_calls = []

    def get_available_metrics(self):
        return list(self._available)

    def profile(self, **kwargs):
        self.profile_calls.append(kwargs)

    def get_dispatch_keys(self):
        return list(self._keys)

    def compute_metric_stats(self, dispatch_key, metric):
        return fake_stats()

    def get_metric_counters(self, metric_name):
        return ["TCC_HIT_sum", "TCC_MISS_sum"]
