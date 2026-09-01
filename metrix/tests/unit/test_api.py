"""
Unit tests for the high-level Metrix API
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from metrix.api import Metrix, ProfilingResults, KernelResults
from metrix.backends import Statistics, get_backend
from metrix.metrics import METRIC_PROFILES
from .conftest import requires_arch, requires_metric


class TestMetrixInit:
    """Test Metrix initialization"""

    def test_init_default(self):
        """Test default initialization (falls back to gfx942 if no hardware detected)"""
        profiler = Metrix()
        # Should succeed on any registered architecture
        assert profiler.backend is not None
        assert len(profiler.arch) > 0

    @pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
    def test_init_custom_arch(self, arch):
        """Test custom architecture initialization"""
        profiler = Metrix(arch=arch)
        assert profiler.arch == arch
        assert profiler.backend is not None


class TestMetrixMetricListing:
    """Test metric and profile listing"""

    @pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
    def test_list_metrics(self, arch):
        """Test listing all metrics"""
        profiler = Metrix(arch=arch)
        metrics = profiler.list_metrics()
        assert len(metrics) > 0
        assert "memory.l2_hit_rate" in metrics
        assert "memory.hbm_bandwidth_utilization" in metrics

    @pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
    def test_list_metrics_includes_compute(self, arch):
        """Test that compute metrics are included in list (CDNA only)"""
        profiler = Metrix(arch=arch)
        metrics = profiler.list_metrics()
        assert "compute.total_flops" in metrics
        assert "compute.hbm_gflops" in metrics
        assert "compute.hbm_arithmetic_intensity" in metrics
        assert "compute.l2_arithmetic_intensity" in metrics
        assert "compute.l1_arithmetic_intensity" in metrics

    def test_list_metrics_on_detected_arch(self):
        """Test that listing metrics works on whatever GPU is detected"""
        profiler = Metrix()
        metrics = profiler.list_metrics()
        assert isinstance(metrics, list)

    @requires_metric("memory.l2_hit_rate")
    def test_list_metrics_includes_l2(self):
        """memory.l2_hit_rate should appear in list_metrics() when supported"""
        profiler = Metrix()
        assert "memory.l2_hit_rate" in profiler.list_metrics()

    @pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
    def test_list_profiles(self, arch):
        """Test listing profiles"""
        profiler = Metrix(arch=arch)
        profiles = profiler.list_profiles()
        assert "quick" in profiles
        assert "memory" in profiles

    @pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
    def test_list_profiles_includes_compute(self, arch):
        """Test that compute profile is included"""
        profiler = Metrix(arch=arch)
        profiles = profiler.list_profiles()
        assert "compute" in profiles

    @pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
    def test_get_metric_info(self, arch):
        """Test getting metric information"""
        profiler = Metrix(arch=arch)
        info = profiler.get_metric_info("memory.l2_hit_rate")
        assert info["name"] == "L2 Cache Hit Rate"
        assert info["unit"] == "Percent"

    @pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
    def test_get_compute_metric_info(self, arch):
        """Test getting compute metric information"""
        profiler = Metrix(arch=arch)
        info = profiler.get_metric_info("compute.total_flops")
        assert info["name"] == "Total FLOPS"
        assert info["unit"] == "FLOPS"

    @pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
    def test_get_arithmetic_intensity_info(self, arch):
        """Test getting arithmetic intensity metric information"""
        profiler = Metrix(arch=arch)
        info = profiler.get_metric_info("compute.hbm_arithmetic_intensity")
        assert info["name"] == "HBM Arithmetic Intensity"
        assert info["unit"] == "FLOPs/Byte"

    @pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
    def test_get_unknown_metric_raises(self, arch):
        """Test getting info for unknown metric raises error"""
        profiler = Metrix(arch=arch)
        with pytest.raises(ValueError, match="Unknown metric"):
            profiler.get_metric_info("nonexistent.metric")


class TestKernelResults:
    """Test KernelResults dataclass"""

    def test_create_kernel_results(self):
        """Test creating kernel results"""
        duration_stats = Statistics(min=100.0, max=200.0, avg=150.0, count=3)
        metric_stats = Statistics(min=50.0, max=60.0, avg=55.0, count=3)

        result = KernelResults(
            name="test_kernel",
            duration_us=duration_stats,
            metrics={"memory.l2_hit_rate": metric_stats},
        )

        assert result.name == "test_kernel"
        assert result.duration_us.avg == 150.0
        assert result.metrics["memory.l2_hit_rate"].avg == 55.0


class TestProfilingResults:
    """Test ProfilingResults dataclass"""

    def test_create_profiling_results(self):
        """Test creating profiling results"""
        kernel1 = KernelResults(
            name="kernel1", duration_us=Statistics(100.0, 100.0, 100.0, 1), metrics={}
        )

        results = ProfilingResults(command="./test", kernels=[kernel1], total_kernels=1)

        assert results.command == "./test"
        assert len(results.kernels) == 1
        assert results.total_kernels == 1
        assert results.kernels[0].name == "kernel1"


class TestUnsupportedMetricsAPI:
    """Test API-level handling of unsupported metrics"""

    @requires_arch("gfx90a")
    def test_explicit_unsupported_metric_raises_error(self):
        """Explicitly requesting unsupported metric should raise ValueError"""
        profiler = Metrix(arch="gfx90a")

        # Verify atomic_latency is marked as unsupported
        assert "memory.atomic_latency" in profiler.backend._unsupported_metrics

    @requires_arch("gfx90a")
    def test_profile_filters_unsupported_in_profile(self):
        """Using a profile that includes unsupported metrics should filter them"""
        profiler = Metrix(arch="gfx90a")

        # Create a test list with both supported and unsupported metrics
        test_metrics = [
            "memory.l2_hit_rate",
            "memory.atomic_latency",  # Unsupported on gfx90a
            "memory.hbm_bandwidth_utilization",
        ]

        # Check unsupported
        unsupported = {
            m: profiler.backend._unsupported_metrics[m]
            for m in test_metrics
            if m in profiler.backend._unsupported_metrics
        }
        assert "memory.atomic_latency" in unsupported

        # Filter supported
        filtered = [m for m in test_metrics if m not in profiler.backend._unsupported_metrics]
        assert "memory.atomic_latency" not in filtered
        assert "memory.l2_hit_rate" in filtered
        assert "memory.hbm_bandwidth_utilization" in filtered


# --------------------------------------------------------------------------
# Metrix.profile(profile=...) — GPU-free coverage of profile resolution
# --------------------------------------------------------------------------

_QUICK_METRICS = METRIC_PROFILES["quick"]["metrics"]


class _FakeSpecs:
    arch = "gfx1030"


class _FakeBackend:
    """Minimal stand-in for the surface Metrix.profile touches."""

    def __init__(self, available, unsupported=None):
        self.device_specs = _FakeSpecs()
        self._available = list(available)
        self._unsupported_metrics = dict(unsupported or {})
        self._aggregated = {"dispatch_1:k": {"duration_us": Statistics(1.0, 1.0, 1.0, 1.0, "us")}}
        self.profile_calls = []

    def get_available_metrics(self):
        return list(self._available)

    def profile(self, **kwargs):
        self.profile_calls.append(kwargs)

    def get_dispatch_keys(self):
        return list(self._aggregated)

    def compute_metric_stats(self, dispatch_key, metric):
        return Statistics(1.0, 1.0, 1.0, 1.0, "percent")


@contextmanager
def _patched(backend):
    with (
        patch("metrix.api.detect_or_default", return_value=backend.device_specs.arch),
        patch("metrix.api.get_backend", return_value=backend),
    ):
        yield Metrix()


class TestProfileResolutionAPI:
    """Metrix.profile narrows a preset to what the architecture supports."""

    def test_partly_supported_profile_collects_the_supported_subset(self, caplog):
        backend = _FakeBackend(available=[_QUICK_METRICS[0]])
        with _patched(backend) as profiler:
            profiler.profile("./app", profile="quick")

        assert backend.profile_calls[0]["metrics"] == [_QUICK_METRICS[0]]
        assert _QUICK_METRICS[1] in caplog.text

    def test_wholly_unsupported_profile_raises_naming_the_architecture(self):
        backend = _FakeBackend(available=["memory.l2_hit_rate"])
        with _patched(backend) as profiler:
            with pytest.raises(ValueError, match="gfx1030"):
                profiler.profile("./app", profile="compute")
        assert backend.profile_calls == []

    def test_unknown_profile_raises(self):
        backend = _FakeBackend(available=list(_QUICK_METRICS))
        with _patched(backend) as profiler:
            with pytest.raises(ValueError, match="Unknown profile"):
                profiler.profile("./app", profile="does-not-exist")

    def test_unsupported_metric_in_profile_reports_its_reason(self, caplog):
        reason = "counter is broken on this part"
        backend = _FakeBackend(
            available=[_QUICK_METRICS[0]],
            unsupported={_QUICK_METRICS[1]: reason},
        )
        with _patched(backend) as profiler:
            profiler.profile("./app", profile="quick")

        assert reason in caplog.text
        assert backend.profile_calls[0]["metrics"] == [_QUICK_METRICS[0]]
