"""
Unit tests for backend metric computations (gfx942, gfx950, and gfx90a)

Tests use MOCK counter data (no hardware counters in test code!)
Tests are parametrized to run on MI300X (gfx942), MI350X (gfx950), and MI200 (gfx90a).
All metrics are loaded from counter_defs.yaml.
"""

import pytest
from unittest.mock import patch
from metrix.backends import get_backend
from metrix.backends.base import DeviceSpecs, Statistics

_TEST_SPECS = {
    "gfx942": DeviceSpecs(
        arch="gfx942",
        name="AMD Instinct MI300X",
        num_cu=304,
        max_waves_per_cu=32,
        wavefront_size=64,
        base_clock_mhz=2100.0,
        hbm_bandwidth_gbs=5300.0,
        l2_size_mb=256.0,
        lds_size_per_cu_kb=64.0,
    ),
    "gfx950": DeviceSpecs(
        arch="gfx950",
        name="AMD Instinct MI350X",
        num_cu=256,
        max_waves_per_cu=32,
        wavefront_size=64,
        base_clock_mhz=2200.0,
        hbm_bandwidth_gbs=6000.0,
        l2_size_mb=256.0,
        lds_size_per_cu_kb=64.0,
    ),
    "gfx90a": DeviceSpecs(
        arch="gfx90a",
        name="AMD Instinct MI210",
        num_cu=104,
        max_waves_per_cu=32,
        wavefront_size=64,
        base_clock_mhz=1700.0,
        hbm_bandwidth_gbs=1600.0,
        l2_size_mb=8.0,
        lds_size_per_cu_kb=64.0,
    ),
}


@pytest.fixture(params=["gfx942", "gfx950", "gfx90a"])
def backend(request):
    """Parametrized fixture that provides gfx942, gfx950, and gfx90a backends"""
    arch = request.param
    patch_target = f"metrix.backends.{arch}.query_device_specs"
    with patch(patch_target, return_value=_TEST_SPECS[arch]):
        return get_backend(arch)


def compute(backend, metric_name):
    """Invoke the YAML-loaded metric compute function"""
    return backend._metrics[metric_name]["compute"]()


def get_arch_counter_names(backend, base_names):
    """
    Map counter names based on backend architecture.

    gfx942/gfx950 (MI300X/MI350X) use TCC_EA0_* naming, gfx90a (MI200) uses TCC_EA_*
    """
    arch = backend.device_specs.arch

    if arch in ("gfx942", "gfx950"):
        mapping = {
            "TCC_EA_RDREQ_sum": "TCC_EA0_RDREQ_sum",
            "TCC_EA_RDREQ_32B_sum": "TCC_EA0_RDREQ_32B_sum",
            "TCC_EA_WRREQ_sum": "TCC_EA0_WRREQ_sum",
            "TCC_EA_WRREQ_64B_sum": "TCC_EA0_WRREQ_64B_sum",
            "TCC_EA_ATOMIC_sum": "TCC_EA0_ATOMIC_sum",
            "TCC_EA_ATOMIC_LEVEL_sum": "TCC_EA0_ATOMIC_LEVEL_sum",
        }
    else:
        mapping = {}

    result = {}
    for base_name, value in base_names.items():
        result[mapping.get(base_name, base_name)] = value

    return result


class TestL2HitRate:
    """Test L2 cache hit rate computation"""

    def test_perfect_hit_rate(self, backend):
        """100% hit rate"""
        backend._raw_data = {"TCC_HIT_sum": 1000, "TCC_MISS_sum": 0}

        result = compute(backend, "memory.l2_hit_rate")
        assert result == 100.0

    def test_zero_hit_rate(self, backend):
        """0% hit rate (all misses)"""
        backend._raw_data = {"TCC_HIT_sum": 0, "TCC_MISS_sum": 1000}

        result = compute(backend, "memory.l2_hit_rate")
        assert result == 0.0

    def test_fifty_percent_hit_rate(self, backend):
        """50% hit rate"""
        backend._raw_data = {"TCC_HIT_sum": 500, "TCC_MISS_sum": 500}

        result = compute(backend, "memory.l2_hit_rate")
        assert result == 50.0

    def test_no_accesses(self, backend):
        """Handle zero total accesses"""
        backend._raw_data = {"TCC_HIT_sum": 0, "TCC_MISS_sum": 0}

        result = compute(backend, "memory.l2_hit_rate")
        assert result == 0.0


class TestCoalescingEfficiency:
    """Test memory coalescing efficiency computation"""

    def test_perfect_coalescing(self, backend):
        """100% coalescing (stride-1 access)"""
        backend._raw_data = {
            "SQ_INSTS_VMEM_RD": 100,
            "SQ_INSTS_VMEM_WR": 0,
            "TCP_TOTAL_ACCESSES_sum": 1600,
        }

        result = compute(backend, "memory.coalescing_efficiency")
        assert result == 100.0

    def test_poor_coalescing(self, backend):
        """25% coalescing (completely uncoalesced float access)"""
        backend._raw_data = {
            "SQ_INSTS_VMEM_RD": 100,
            "SQ_INSTS_VMEM_WR": 0,
            "TCP_TOTAL_ACCESSES_sum": 6400,
        }

        result = compute(backend, "memory.coalescing_efficiency")
        assert result == 25.0

    def test_mixed_read_write(self, backend):
        """Coalescing with both reads and writes"""
        backend._raw_data = {
            "SQ_INSTS_VMEM_RD": 50,
            "SQ_INSTS_VMEM_WR": 50,
            "TCP_TOTAL_ACCESSES_sum": 1600,
        }

        result = compute(backend, "memory.coalescing_efficiency")
        assert result == 100.0

    def test_no_memory_instructions(self, backend):
        """Handle zero memory instructions"""
        backend._raw_data = {
            "SQ_INSTS_VMEM_RD": 0,
            "SQ_INSTS_VMEM_WR": 0,
            "TCP_TOTAL_ACCESSES_sum": 1000,
        }

        result = compute(backend, "memory.coalescing_efficiency")
        assert result == 0.0


class TestLDSBankConflicts:
    """Test LDS bank conflict computation"""

    def test_no_conflicts(self, backend):
        """Perfect LDS access pattern"""
        backend._raw_data = {"SQ_LDS_BANK_CONFLICT": 0, "SQ_INSTS_LDS": 1000}

        result = compute(backend, "memory.lds_bank_conflicts")
        assert result == 0.0

    def test_high_conflicts(self, backend):
        """2 conflicts per instruction"""
        backend._raw_data = {"SQ_LDS_BANK_CONFLICT": 2000, "SQ_INSTS_LDS": 1000}

        result = compute(backend, "memory.lds_bank_conflicts")
        assert result == 2.0

    def test_no_lds_instructions(self, backend):
        """Handle zero LDS instructions"""
        backend._raw_data = {"SQ_LDS_BANK_CONFLICT": 100, "SQ_INSTS_LDS": 0}

        result = compute(backend, "memory.lds_bank_conflicts")
        assert result == 0.0


class TestBandwidthMetrics:
    """Test HBM bandwidth computations with 32B/64B/128B request granularity"""

    def test_hbm_read_bandwidth_64b_only(self, backend):
        """Test read bandwidth with only 64B requests"""
        arch = backend.device_specs.arch
        # Use clock_mhz * 1000 as active cycles → active_time = 0.001s for any arch
        active_cycles = int(backend.device_specs.base_clock_mhz * 1000)

        if arch in ("gfx942", "gfx950"):
            counters = {
                "TCC_EA_RDREQ_sum": 1000,
                "TCC_EA_RDREQ_32B_sum": 0,
                "TCC_BUBBLE_sum": 0,
                "GRBM_GUI_ACTIVE": active_cycles,
            }
        else:
            counters = {
                "TCC_EA_RDREQ_sum": 1000,
                "TCC_EA_RDREQ_32B_sum": 0,
                "GRBM_GUI_ACTIVE": active_cycles,
            }

        backend._raw_data = get_arch_counter_names(backend, counters)
        result = compute(backend, "memory.hbm_read_bandwidth")
        assert 0.06 < result < 0.07

    def test_hbm_read_bandwidth_mixed_sizes(self, backend):
        """Test read bandwidth with mixed request sizes"""
        arch = backend.device_specs.arch
        active_cycles = int(backend.device_specs.base_clock_mhz * 1000)

        if arch in ("gfx942", "gfx950"):
            counters = {
                "TCC_EA_RDREQ_sum": 1000,
                "TCC_EA_RDREQ_32B_sum": 200,
                "TCC_BUBBLE_sum": 300,
                "GRBM_GUI_ACTIVE": active_cycles,
            }
            expected_min, expected_max = 0.07, 0.08
        else:
            counters = {
                "TCC_EA_RDREQ_sum": 1000,
                "TCC_EA_RDREQ_32B_sum": 400,
                "GRBM_GUI_ACTIVE": active_cycles,
            }
            expected_min, expected_max = 0.05, 0.06

        backend._raw_data = get_arch_counter_names(backend, counters)
        result = compute(backend, "memory.hbm_read_bandwidth")
        assert expected_min < result < expected_max

    def test_hbm_write_bandwidth_64b_only(self, backend):
        """Test write bandwidth with only 64B requests"""
        active_cycles = int(backend.device_specs.base_clock_mhz * 1000)

        counters = {
            "TCC_EA_WRREQ_sum": 1000,
            "TCC_EA_WRREQ_64B_sum": 1000,
            "GRBM_GUI_ACTIVE": active_cycles,
        }

        backend._raw_data = get_arch_counter_names(backend, counters)
        result = compute(backend, "memory.hbm_write_bandwidth")
        assert 0.06 < result < 0.07

    def test_hbm_write_bandwidth_mixed_sizes(self, backend):
        """Test write bandwidth with mixed 32B and 64B requests"""
        active_cycles = int(backend.device_specs.base_clock_mhz * 1000)

        counters = {
            "TCC_EA_WRREQ_sum": 1000,
            "TCC_EA_WRREQ_64B_sum": 600,
            "GRBM_GUI_ACTIVE": active_cycles,
        }

        backend._raw_data = get_arch_counter_names(backend, counters)
        result = compute(backend, "memory.hbm_write_bandwidth")
        assert 0.05 < result < 0.06

    def test_zero_active_cycles(self, backend):
        """Handle zero active cycles"""
        counters = {"TCC_EA_RDREQ_sum": 1000, "TCC_EA_RDREQ_32B_sum": 0, "GRBM_GUI_ACTIVE": 0}

        if backend.device_specs.arch in ("gfx942", "gfx950"):
            counters["TCC_BUBBLE_sum"] = 0

        backend._raw_data = get_arch_counter_names(backend, counters)
        result = compute(backend, "memory.hbm_read_bandwidth")
        assert result == 0.0


class TestAtomicLatency:
    """Test L2 cache atomic operation latency computation"""

    @pytest.fixture(params=["gfx942", "gfx950"])
    def atomic_backend(self, request):
        """gfx942 and gfx950 support atomic_latency (broken on gfx90a)"""
        arch = request.param
        with patch(
            f"metrix.backends.{arch}.query_device_specs",
            return_value=_TEST_SPECS[arch],
        ):
            return get_backend(arch)

    def test_low_latency(self, atomic_backend):
        """10 cycles per atomic operation"""
        counters = {"TCC_EA_ATOMIC_sum": 1000, "TCC_EA_ATOMIC_LEVEL_sum": 10000}

        atomic_backend._raw_data = get_arch_counter_names(atomic_backend, counters)
        result = compute(atomic_backend, "memory.atomic_latency")
        assert result == 10.0

    def test_high_latency(self, atomic_backend):
        """1000 cycles per atomic (contention)"""
        counters = {"TCC_EA_ATOMIC_sum": 100, "TCC_EA_ATOMIC_LEVEL_sum": 100000}

        atomic_backend._raw_data = get_arch_counter_names(atomic_backend, counters)
        result = compute(atomic_backend, "memory.atomic_latency")
        assert result == 1000.0

    def test_no_atomics(self, atomic_backend):
        """Handle zero atomic instructions"""
        counters = {"TCC_EA_ATOMIC_sum": 0, "TCC_EA_ATOMIC_LEVEL_sum": 5000}

        atomic_backend._raw_data = get_arch_counter_names(atomic_backend, counters)
        result = compute(atomic_backend, "memory.atomic_latency")
        assert result == 0.0


class TestMetricDiscovery:
    """Test backend auto-discovers metrics"""

    def test_discovers_core_metrics(self, backend):
        """Backend should discover core YAML-defined metrics for CDNA"""
        metrics = backend.get_available_metrics()

        # These are available on all CDNA architectures
        assert "memory.l2_hit_rate" in metrics
        assert "memory.coalescing_efficiency" in metrics
        assert "memory.lds_bank_conflicts" in metrics
        assert "memory.hbm_read_bandwidth" in metrics

        if backend.device_specs.arch == "gfx90a":
            assert "memory.atomic_latency" not in metrics
            assert "memory.atomic_latency" in backend._unsupported_metrics
        else:
            assert "memory.atomic_latency" in metrics

    def test_get_required_counters(self, backend):
        """Backend should correctly report required counters for a metric"""
        counters = backend.get_required_counters(["memory.l2_hit_rate"])

        # CDNA backends use TCC counters for L2
        assert "TCC_HIT_sum" in counters
        assert "TCC_MISS_sum" in counters
        assert len(counters) == 2

    def test_discovers_compute_metrics(self, backend):
        """CDNA backend should discover all compute metrics"""
        metrics = backend.get_available_metrics()

        assert "compute.total_flops" in metrics
        assert "compute.hbm_gflops" in metrics
        assert "compute.hbm_arithmetic_intensity" in metrics
        assert "compute.l2_arithmetic_intensity" in metrics
        assert "compute.l1_arithmetic_intensity" in metrics

    def test_yaml_units_loaded(self, backend):
        """YAML unit strings are loaded into backend._metrics"""
        # Only check metrics that are actually available on this backend
        expected_units = {
            "memory.l2_hit_rate": "Percent",
            "memory.l2_bandwidth": "GB/s",
            "memory.hbm_read_bandwidth": "GB/s",
            "memory.hbm_write_bandwidth": "GB/s",
            "memory.hbm_bandwidth_utilization": "Percent",
            "memory.bytes_transferred_l2": "Bytes",
            "memory.bytes_transferred_hbm": "Bytes",
            "memory.lds_bank_conflicts": "Conflicts per Access",
        }
        for metric_name, expected_unit in expected_units.items():
            if metric_name not in backend._metrics:
                continue  # Skip metrics not available on this arch
            actual_unit = backend._metrics[metric_name].get("unit", "")
            assert actual_unit == expected_unit, (
                f"{metric_name}: expected unit '{expected_unit}', got '{actual_unit}'"
            )

    def test_compute_metric_stats_returns_unit(self, backend):
        """compute_metric_stats() returns Statistics with correct unit"""
        dispatch_key = "test_kernel"
        backend._aggregated = {
            dispatch_key: {
                "TCC_HIT_sum": Statistics(min=800, max=800, avg=800, count=1),
                "TCC_MISS_sum": Statistics(min=200, max=200, avg=200, count=1),
            }
        }

        stats = backend.compute_metric_stats(dispatch_key, "memory.l2_hit_rate")
        assert stats.unit == "Percent"
        assert stats.avg == 80.0


class TestComputeMetrics:
    """Test compute metric computations (FLOPS, arithmetic intensity)"""

    def _get_zero_flops_counters(self):
        """Helper: return counter dict with all FLOPS counters set to 0"""
        return {
            "SQ_INSTS_VALU_ADD_F16": 0,
            "SQ_INSTS_VALU_MUL_F16": 0,
            "SQ_INSTS_VALU_TRANS_F16": 0,
            "SQ_INSTS_VALU_FMA_F16": 0,
            "SQ_INSTS_VALU_ADD_F32": 0,
            "SQ_INSTS_VALU_MUL_F32": 0,
            "SQ_INSTS_VALU_TRANS_F32": 0,
            "SQ_INSTS_VALU_FMA_F32": 0,
            "SQ_INSTS_VALU_ADD_F64": 0,
            "SQ_INSTS_VALU_MUL_F64": 0,
            "SQ_INSTS_VALU_TRANS_F64": 0,
            "SQ_INSTS_VALU_FMA_F64": 0,
            "SQ_INSTS_VALU_MFMA_MOPS_F16": 0,
            "SQ_INSTS_VALU_MFMA_MOPS_BF16": 0,
            "SQ_INSTS_VALU_MFMA_MOPS_F32": 0,
            "SQ_INSTS_VALU_MFMA_MOPS_F64": 0,
        }

    def test_total_flops_fp32_add(self, backend):
        """Test FLOPS calculation with FP32 add instructions"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 100

        result = compute(backend, "compute.total_flops")
        assert result == 6400

    def test_total_flops_fma_counts_double(self, backend):
        """Test that FMA instructions count as 2 operations"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_FMA_F32"] = 100

        result = compute(backend, "compute.total_flops")
        assert result == 12800

    def test_total_flops_mfma_high_throughput(self, backend):
        """Test MFMA instructions produce 512 operations each"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_MFMA_MOPS_F32"] = 10

        result = compute(backend, "compute.total_flops")
        assert result == 5120

    def test_total_flops_mixed_precision(self, backend):
        """Test FLOPS with mixed precision operations"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F16"] = 100
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 50
        backend._raw_data["SQ_INSTS_VALU_ADD_F64"] = 25

        result = compute(backend, "compute.total_flops")
        assert result == 6400 + 3200 + 1600

    def test_total_flops_zero(self, backend):
        """Handle zero FLOPS gracefully"""
        backend._raw_data = self._get_zero_flops_counters()

        result = compute(backend, "compute.total_flops")
        assert result == 0

    def test_hbm_gflops_zero_time(self, backend):
        """Handle zero duration"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 1000
        backend._current_duration_us = 0.0

        result = compute(backend, "compute.hbm_gflops")
        assert result == 0.0

    def test_hbm_arithmetic_intensity(self, backend):
        """Test HBM arithmetic intensity calculation"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 1000

        counters = {
            "TCC_EA_RDREQ_sum": 1000,
            "TCC_EA_RDREQ_32B_sum": 0,
            "TCC_EA_WRREQ_sum": 0,
            "TCC_EA_WRREQ_64B_sum": 0,
        }

        if backend.device_specs.arch in ("gfx942", "gfx950"):
            counters["TCC_BUBBLE_sum"] = 0

        backend._raw_data.update(get_arch_counter_names(backend, counters))

        result = compute(backend, "compute.hbm_arithmetic_intensity")
        assert result == 1.0

    def test_hbm_arithmetic_intensity_zero_bytes(self, backend):
        """Handle zero HBM bytes transferred"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 1000

        counters = {
            "TCC_EA_RDREQ_sum": 0,
            "TCC_EA_RDREQ_32B_sum": 0,
            "TCC_EA_WRREQ_sum": 0,
            "TCC_EA_WRREQ_64B_sum": 0,
        }

        if backend.device_specs.arch in ("gfx942", "gfx950"):
            counters["TCC_BUBBLE_sum"] = 0

        backend._raw_data.update(get_arch_counter_names(backend, counters))

        result = compute(backend, "compute.hbm_arithmetic_intensity")
        assert result == 0.0

    def test_l2_arithmetic_intensity(self, backend):
        """Test L2 arithmetic intensity calculation"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 1000
        backend._raw_data["TCC_REQ_sum"] = 500

        result = compute(backend, "compute.l2_arithmetic_intensity")
        assert result == 1.0

    def test_l2_arithmetic_intensity_zero_bytes(self, backend):
        """Handle zero L2 bytes"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 1000
        backend._raw_data["TCC_REQ_sum"] = 0

        result = compute(backend, "compute.l2_arithmetic_intensity")
        assert result == 0.0

    def test_l1_arithmetic_intensity(self, backend):
        """Test L1 arithmetic intensity calculation"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 1000

        if backend.device_specs.arch in ("gfx942", "gfx950"):
            # 128B L1 cache line: 1000 FP32 FLOPS / (500 * 128B) = 1.0
            backend._raw_data["TCP_TOTAL_CACHE_ACCESSES_sum"] = 500
        else:
            # gfx90a: 64B L1 cache line: 1000 FP32 FLOPS / (1000 * 64B) = 1.0
            backend._raw_data["TCP_TOTAL_CACHE_ACCESSES_sum"] = 1000

        result = compute(backend, "compute.l1_arithmetic_intensity")
        assert result == 1.0

    def test_l1_arithmetic_intensity_zero_bytes(self, backend):
        """Handle zero L1 bytes"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 1000
        backend._raw_data["TCP_TOTAL_CACHE_ACCESSES_sum"] = 0

        result = compute(backend, "compute.l1_arithmetic_intensity")
        assert result == 0.0

    def test_high_arithmetic_intensity_compute_bound(self, backend):
        """Test high AI indicates compute-bound kernel"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_MFMA_MOPS_F32"] = 1000

        counters = {
            "TCC_EA_RDREQ_sum": 100,
            "TCC_EA_RDREQ_32B_sum": 0,
            "TCC_EA_WRREQ_sum": 0,
            "TCC_EA_WRREQ_64B_sum": 0,
        }

        if backend.device_specs.arch in ("gfx942", "gfx950"):
            counters["TCC_BUBBLE_sum"] = 0

        backend._raw_data.update(get_arch_counter_names(backend, counters))

        result = compute(backend, "compute.hbm_arithmetic_intensity")
        assert result == 80.0

    def test_low_arithmetic_intensity_memory_bound(self, backend):
        """Test low AI indicates memory-bound kernel"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 100

        counters = {
            "TCC_EA_RDREQ_sum": 10000,
            "TCC_EA_RDREQ_32B_sum": 0,
            "TCC_EA_WRREQ_sum": 0,
            "TCC_EA_WRREQ_64B_sum": 0,
        }

        if backend.device_specs.arch in ("gfx942", "gfx950"):
            counters["TCC_BUBBLE_sum"] = 0

        backend._raw_data.update(get_arch_counter_names(backend, counters))

        result = compute(backend, "compute.hbm_arithmetic_intensity")
        assert result == 0.01
