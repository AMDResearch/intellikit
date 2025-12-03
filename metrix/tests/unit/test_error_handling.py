"""
Unit tests for error handling - ensuring robustness

Tests various error conditions to ensure the profiler handles them gracefully.
"""

import pytest
import subprocess
from pathlib import Path
from metrix.profiler.rocprof_wrapper import ROCProfV3Wrapper


class TestMissingExecutable:
    """Test handling of missing target executable"""

    def test_missing_executable_error_message(self):
        """Should provide helpful error message for missing executable"""
        wrapper = ROCProfV3Wrapper(timeout=5)

        # rocprofv3 wraps FileNotFoundError in RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            wrapper.profile("nonexistent_binary", counters=[])

        # Error message should mention the executable
        error_msg = str(exc_info.value)
        assert "nonexistent_binary" in error_msg or "FileNotFoundError" in error_msg

    def test_wrong_path_error_message(self):
        """Should provide helpful error when path is incorrect"""
        wrapper = ROCProfV3Wrapper(timeout=5)

        # rocprofv3 wraps FileNotFoundError in RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            wrapper.profile("vector_add", counters=[])  # Missing ./

        error_msg = str(exc_info.value)
        assert "vector_add" in error_msg or "FileNotFoundError" in error_msg


class TestInvalidArguments:
    """Test handling of invalid CLI arguments"""

    def test_invalid_metric_name(self):
        """Should handle invalid metric names gracefully"""
        from metrix.backends import get_backend

        backend = get_backend("gfx942")

        with pytest.raises(ValueError) as exc_info:
            backend.get_required_counters(["invalid.metric.name"])

        assert "invalid.metric.name" in str(exc_info.value)

    def test_invalid_architecture(self):
        """Should handle invalid architecture names"""
        from metrix.backends import get_backend

        with pytest.raises(ValueError) as exc_info:
            get_backend("gfx9999")

        assert "gfx9999" in str(exc_info.value)


class TestTimeoutHandling:
    """Test timeout handling"""

    def test_respects_timeout_setting(self):
        """ROCProfV3Wrapper should respect timeout setting"""
        wrapper = ROCProfV3Wrapper(timeout=1)
        assert wrapper.timeout == 1

        wrapper = ROCProfV3Wrapper(timeout=60)
        assert wrapper.timeout == 60

    def test_default_timeout(self):
        """Should have reasonable default timeout"""
        wrapper = ROCProfV3Wrapper()
        assert wrapper.timeout == 60  # Default from CLI


class TestBackendValidation:
    """Test backend metric validation"""

    def test_get_available_metrics(self):
        """Backend should list all available metrics"""
        from metrix.backends import get_backend

        backend = get_backend("gfx942")
        metrics = backend.get_available_metrics()

        assert len(metrics) > 0
        assert "memory.l2_hit_rate" in metrics
        assert "memory.coalescing_efficiency" in metrics

    def test_get_required_counters(self):
        """Backend should report required counters for metrics"""
        from metrix.backends import get_backend

        backend = get_backend("gfx942")
        counters = backend.get_required_counters(["memory.l2_hit_rate"])

        assert len(counters) > 0
        assert "TCC_HIT_sum" in counters
        assert "TCC_MISS_sum" in counters


class TestMetricComputation:
    """Test metric computation edge cases"""

    def test_division_by_zero_handling(self):
        """Metrics should handle zero denominators gracefully"""
        from metrix.backends import get_backend

        backend = get_backend("gfx942")
        backend._raw_data = {
            'TCC_HIT_sum': 0,
            'TCC_MISS_sum': 0
        }

        # Should return 0.0, not raise ZeroDivisionError
        result = backend._l2_hit_rate()
        assert result == 0.0

    def test_negative_values_handling(self):
        """Metrics should handle negative counter values (shouldn't happen, but...)"""
        from metrix.backends import get_backend

        backend = get_backend("gfx942")
        backend._raw_data = {
            'TCC_HIT_sum': -100,  # Shouldn't happen in practice
            'TCC_MISS_sum': 100
        }

        # Should not crash
        result = backend._l2_hit_rate()
        assert isinstance(result, (int, float))
