"""Unit tests for kerncap.profiler — rocprofv3 CSV parsing logic."""

import os
from pathlib import Path

import pytest

from kerncap.profiler import parse_kernel_trace_stats

FIXTURES = Path(__file__).parent / "fixtures"


class TestParseKernelTraceStats:
    """Tests for the rocprofv3 kernel_trace_stats.csv parser."""

    def test_parse_rocprofv3_stats(self):
        """Parse rocprofv3 kernel_trace_stats.csv with all columns."""
        csv_path = str(FIXTURES / "rocprofv3_kernel_stats.csv")
        kernels = parse_kernel_trace_stats(csv_path)

        assert len(kernels) == 5

        # Should be sorted by total duration descending
        assert kernels[0].total_duration_ns == 580000000
        assert "matmul_kernel" in kernels[0].name
        assert kernels[0].calls == 1024
        assert kernels[0].percentage == pytest.approx(42.3)
        assert kernels[0].min_duration_ns == 480000
        assert kernels[0].max_duration_ns == 720000
        assert kernels[0].stddev_ns == pytest.approx(45000.0)

        # Last kernel should have the smallest total
        assert kernels[-1].total_duration_ns == 140000000

    def test_sorted_by_total_duration(self):
        """Results should always be sorted descending by total duration."""
        csv_path = str(FIXTURES / "rocprofv3_kernel_stats.csv")
        kernels = parse_kernel_trace_stats(csv_path)

        durations = [k.total_duration_ns for k in kernels]
        assert durations == sorted(durations, reverse=True)

    def test_avg_duration_populated(self):
        """AverageNs should be parsed from the stats CSV."""
        csv_path = str(FIXTURES / "rocprofv3_kernel_stats.csv")
        kernels = parse_kernel_trace_stats(csv_path)

        for k in kernels:
            assert k.avg_duration_ns > 0

    def test_empty_csv(self, tmp_path):
        """Empty CSV should return empty list."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        kernels = parse_kernel_trace_stats(str(csv_file))
        assert kernels == []

    def test_missing_columns_raises(self, tmp_path):
        """CSV without a name column should raise ValueError."""
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("foo,bar\n1,2\n")
        with pytest.raises(ValueError, match="kernel name column"):
            parse_kernel_trace_stats(str(csv_file))
