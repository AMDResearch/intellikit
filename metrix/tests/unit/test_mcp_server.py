"""
Unit tests for the MCP server tool definitions.

These tests verify that the MCP tools return valid data that matches
the actual metric catalog, preventing regressions like hardcoded
metric names that don't exist.
"""

from metrix.mcp.server import list_available_metrics, profile_metrics
from metrix.metrics import METRIC_CATALOG


class TestListAvailableMetrics:
    """Test the list_available_metrics MCP tool"""

    def test_returns_only_catalog_metrics(self):
        """Every metric returned by list_available_metrics must exist in METRIC_CATALOG"""
        result = list_available_metrics()
        for metric in result["metrics"]:
            assert metric in METRIC_CATALOG, (
                f"list_available_metrics returned '{metric}' which does not exist "
                f"in METRIC_CATALOG. Did you mean one of: "
                f"{[m for m in METRIC_CATALOG if metric.split('.')[-1] in m]}"
            )

    def test_returns_nonempty(self):
        """list_available_metrics must return at least one metric"""
        result = list_available_metrics()
        assert len(result["metrics"]) > 0

    def test_includes_common_metrics(self):
        """list_available_metrics should include well-known metrics"""
        result = list_available_metrics()
        metrics = result["metrics"]
        assert "memory.hbm_bandwidth_utilization" in metrics
        assert "memory.l2_hit_rate" in metrics

    def test_no_bogus_metric_names(self):
        """Explicitly check that previously-hardcoded wrong names are not returned"""
        result = list_available_metrics()
        metrics = result["metrics"]
        # These were the old hardcoded names that don't exist
        assert "memory.l2_cache_hit_rate" not in metrics, (
            "memory.l2_cache_hit_rate is not a real metric — use memory.l2_hit_rate"
        )
        assert "compute.cu_utilization" not in metrics, (
            "compute.cu_utilization does not exist in the metric catalog"
        )
        assert "compute.wave_occupancy" not in metrics, (
            "compute.wave_occupancy does not exist in the metric catalog"
        )

    def test_by_category_grouping(self):
        """list_available_metrics should group metrics by category"""
        result = list_available_metrics()
        assert "by_category" in result
        by_cat = result["by_category"]
        # Should have at least memory and compute categories
        categories = set(by_cat.keys())
        assert "memory_bandwidth" in categories or "memory_cache" in categories
        # Every metric in by_category must also be in the flat list
        flat = set(result["metrics"])
        for cat_metrics in by_cat.values():
            for m in cat_metrics:
                assert m in flat

    def test_metric_count_matches_catalog(self):
        """list_available_metrics should return all metrics from the catalog"""
        result = list_available_metrics()
        assert len(result["metrics"]) == len(METRIC_CATALOG)
