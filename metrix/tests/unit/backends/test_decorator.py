"""
Unit tests for @metric decorator
"""

import pytest
from metrix.backends.decorator import metric


def test_metric_decorator_extracts_counters():
    """Test that decorator correctly extracts counter names from function signature"""

    @metric("test.metric")
    def _test_func(self, counter_a, counter_b, counter_c):
        return counter_a + counter_b + counter_c

    # Decorator should attach metadata
    assert hasattr(_test_func, '_metric_name')
    assert hasattr(_test_func, '_metric_counters')

    assert _test_func._metric_name == "test.metric"
    assert _test_func._metric_counters == ['counter_a', 'counter_b', 'counter_c']


def test_metric_decorator_skips_self():
    """Test that 'self' parameter is not included in counters"""

    @metric("test.metric2")
    def _test_func(self, hw_counter_1, hw_counter_2):
        return hw_counter_1 / hw_counter_2

    assert 'self' not in _test_func._metric_counters
    assert _test_func._metric_counters == ['hw_counter_1', 'hw_counter_2']


def test_metric_decorator_wrapper_extracts_from_raw_data():
    """Test that wrapper correctly extracts counter values from _raw_data"""

    class MockBackend:
        def __init__(self):
            self._raw_data = {
                'TCC_HIT_sum': 75,
                'TCC_MISS_sum': 25
            }

        @metric("cache.hit_rate")
        def _hit_rate(self, TCC_HIT_sum, TCC_MISS_sum):
            total = TCC_HIT_sum + TCC_MISS_sum
            return (TCC_HIT_sum / total) * 100 if total > 0 else 0.0

    backend = MockBackend()
    result = backend._hit_rate()

    assert result == 75.0  # (75 / 100) * 100


def test_metric_decorator_handles_missing_counters():
    """Test that missing counters default to 0"""

    class MockBackend:
        def __init__(self):
            self._raw_data = {
                'counter_a': 100
                # counter_b is missing
            }

        @metric("test.sum")
        def _sum(self, counter_a, counter_b):
            return counter_a + counter_b

    backend = MockBackend()
    result = backend._sum()

    assert result == 100  # 100 + 0 (missing defaults to 0)


def test_metric_decorator_preserves_original_function():
    """Test that original function is accessible for debugging"""

    @metric("test.func")
    def _original(self, x, y):
        return x * y

    assert hasattr(_original, '_original_func')
    assert _original._original_func == _original.__wrapped__ if hasattr(_original, '__wrapped__') else _original._original_func

