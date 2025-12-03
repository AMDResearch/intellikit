"""
Unit tests for the high-level Metrix API
"""

import pytest
from metrix.api import Metrix, ProfilingResults, KernelResults
from metrix.backends import Statistics


class TestMetrixInit:
    """Test Metrix initialization"""

    def test_init_default(self):
        """Test default initialization"""
        profiler = Metrix()
        assert profiler.arch == "gfx942"
        assert profiler.backend is not None

    def test_init_custom_arch(self):
        """Test custom architecture"""
        profiler = Metrix(arch="gfx942")
        assert profiler.arch == "gfx942"
        assert profiler.backend is not None


class TestMetrixMetricListing:
    """Test metric and profile listing"""

    def test_list_metrics(self):
        """Test listing all metrics"""
        profiler = Metrix()
        metrics = profiler.list_metrics()
        assert len(metrics) > 0
        assert "memory.l2_hit_rate" in metrics
        assert "memory.hbm_bandwidth_utilization" in metrics

    def test_list_metrics_includes_compute(self):
        """Test that compute metrics are included in list"""
        profiler = Metrix()
        metrics = profiler.list_metrics()
        assert "compute.total_flops" in metrics
        assert "compute.hbm_gflops" in metrics
        assert "compute.hbm_arithmetic_intensity" in metrics
        assert "compute.l2_arithmetic_intensity" in metrics
        assert "compute.l1_arithmetic_intensity" in metrics

    def test_list_profiles(self):
        """Test listing profiles"""
        profiler = Metrix()
        profiles = profiler.list_profiles()
        assert "quick" in profiles
        assert "memory" in profiles

    def test_list_profiles_includes_compute(self):
        """Test that compute profile is included"""
        profiler = Metrix()
        profiles = profiler.list_profiles()
        assert "compute" in profiles

    def test_get_metric_info(self):
        """Test getting metric information"""
        profiler = Metrix()
        info = profiler.get_metric_info("memory.l2_hit_rate")
        assert info['name'] == "L2 Cache Hit Rate"
        assert info['unit'] == "percent"

    def test_get_compute_metric_info(self):
        """Test getting compute metric information"""
        profiler = Metrix()
        info = profiler.get_metric_info("compute.total_flops")
        assert info['name'] == "Total FLOPS"
        assert info['unit'] == "FLOPS"

    def test_get_arithmetic_intensity_info(self):
        """Test getting arithmetic intensity metric information"""
        profiler = Metrix()
        info = profiler.get_metric_info("compute.hbm_arithmetic_intensity")
        assert info['name'] == "HBM Arithmetic Intensity"
        assert info['unit'] == "FLOP/byte"

    def test_get_unknown_metric_raises(self):
        """Test getting info for unknown metric raises error"""
        profiler = Metrix()
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
            metrics={"memory.l2_hit_rate": metric_stats}
        )

        assert result.name == "test_kernel"
        assert result.duration_us.avg == 150.0
        assert result.metrics["memory.l2_hit_rate"].avg == 55.0


class TestProfilingResults:
    """Test ProfilingResults dataclass"""

    def test_create_profiling_results(self):
        """Test creating profiling results"""
        kernel1 = KernelResults(
            name="kernel1",
            duration_us=Statistics(100.0, 100.0, 100.0, 1),
            metrics={}
        )

        results = ProfilingResults(
            command="./test",
            kernels=[kernel1],
            total_kernels=1
        )

        assert results.command == "./test"
        assert len(results.kernels) == 1
        assert results.total_kernels == 1
        assert results.kernels[0].name == "kernel1"

