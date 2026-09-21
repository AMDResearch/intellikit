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
    """Metrics available on the detected hardware, plus whether probing worked.

    Returns ``(metrics, probe_ok)``. An empty set means two very different
    things and callers have to tell them apart: ``probe_ok=False`` says the
    backend could not be built at all -- no GPU, no ROCm, no hipcc -- which is
    a gap in the environment, while ``probe_ok=True`` with an empty set is the
    architecture's actual answer, and on an arch that counter_defs.yaml
    defines metrics for that is a regression rather than a fact of life.
    """
    if HW_ARCH is None:
        return set(), False
    try:
        from metrix.backends import get_backend

        backend = get_backend(HW_ARCH)
        return set(backend.get_available_metrics()), True
    except (ValueError, RuntimeError):
        return set(), False


HW_METRICS, HW_PROBE_OK = _hw_metrics()


def _archs_with_counter_defs():
    """Architectures that counter_defs.yaml defines at least one metric for.

    Read from the YAML rather than from the backend, and that is the point:
    it is independent of what the backend reports at runtime, so a test can
    tell "this arch is expected to expose no counters" apart from "this arch
    should expose counters but produced none", which is a regression. Every
    definition in the YAML is arch-gated, so an arch absent from every
    ``architectures:`` list genuinely resolves to zero metrics.
    """
    from pathlib import Path

    import yaml

    import metrix.backends as _pkg

    yaml_path = Path(_pkg.__file__).resolve().parent / "counter_defs.yaml"
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return set()

    archs = set()
    for counter in data.get("rocprofiler-sdk", {}).get("counters", []):
        for defn in counter.get("definitions", []):
            archs.update(defn.get("architectures", []))
    return archs


ARCHS_WITH_COUNTER_DEFS = _archs_with_counter_defs()


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


def requires_counter_metrics():
    """Decorator: skip a test unless this GPU is expected to expose counters.

    gfx1103 (Phoenix / Radeon 780M) exposes none -- ROCm ships no hardware
    counter definitions for it -- so every built-in profile is empty there.

    Keyed on the architecture being absent from counter_defs.yaml, never on the
    observed metric set being empty. Keying it on emptiness would be
    self-exempting in the same way the profile invariant was: if metrics
    vanished on an arch the YAML does define, every test guarded by this
    decorator would quietly skip and the regression would never surface.
    An arch with counter definitions is required to produce metrics, so tests
    run there and fail if it does not.

    Still skips when the backend could not be probed at all -- no GPU, no ROCm,
    no hipcc. That is an environmental gap, not a claim about the hardware, and
    turning it into a failure would only mean every GPU test fails together on
    a machine that was never able to run them.
    """
    if not HW_PROBE_OK:
        return pytest.mark.skipif(
            True,
            reason=f"no GPU backend available to probe (detected arch: {HW_ARCH})",
        )
    return pytest.mark.skipif(
        HW_ARCH not in ARCHS_WITH_COUNTER_DEFS,
        reason=f"{HW_ARCH} has no counter definitions in counter_defs.yaml (time-only mode)",
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

    def get_unsupported_metrics(self):
        return dict(self._unsupported_metrics)

    def profile(self, **kwargs):
        self.profile_calls.append(kwargs)

    def get_dispatch_keys(self):
        return list(self._keys)

    def compute_metric_stats(self, dispatch_key, metric):
        return fake_stats()

    def get_metric_counters(self, metric_name):
        return ["TCC_HIT_sum", "TCC_MISS_sum"]
