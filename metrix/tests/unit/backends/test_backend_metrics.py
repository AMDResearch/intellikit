"""
Unit tests for backend metric computations (gfx942 and gfx90a)

Tests use MOCK counter data (no hardware counters in test code!)
Tests are parametrized to run on both MI300X (gfx942) and MI200 (gfx90a)
"""

import pytest
from metrix.backends import get_backend


@pytest.fixture(params=["gfx942", "gfx90a"])
def backend(request):
    """Parametrized fixture that provides both gfx942 and gfx90a backends"""
    return get_backend(request.param)


def get_arch_counter_names(backend, base_names):
    """
    Map counter names based on backend architecture.

    gfx942 (MI300X) uses TCC_EA0_* naming, gfx90a (MI200) uses TCC_EA_*
    """
    arch = backend.device_specs.arch

    if arch == "gfx942":
        # MI300X counter mapping
        mapping = {
            "TCC_EA_RDREQ_sum": "TCC_EA0_RDREQ_sum",
            "TCC_EA_RDREQ_32B_sum": "TCC_EA0_RDREQ_32B_sum",
            "TCC_EA_WRREQ_sum": "TCC_EA0_WRREQ_sum",
            "TCC_EA_WRREQ_64B_sum": "TCC_EA0_WRREQ_64B_sum",
            "TCC_EA_ATOMIC_sum": "TCC_EA0_ATOMIC_sum",
            "TCC_EA_ATOMIC_LEVEL_sum": "TCC_EA0_ATOMIC_LEVEL_sum",
        }
    else:  # gfx90a
        # MI200 uses base names as-is
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

        result = backend._l2_hit_rate()
        assert result == 100.0

    def test_zero_hit_rate(self, backend):
        """0% hit rate (all misses)"""
        backend._raw_data = {"TCC_HIT_sum": 0, "TCC_MISS_sum": 1000}

        result = backend._l2_hit_rate()
        assert result == 0.0

    def test_fifty_percent_hit_rate(self, backend):
        """50% hit rate"""
        backend._raw_data = {"TCC_HIT_sum": 500, "TCC_MISS_sum": 500}

        result = backend._l2_hit_rate()
        assert result == 50.0

    def test_no_accesses(self, backend):
        """Handle zero total accesses"""
        backend._raw_data = {"TCC_HIT_sum": 0, "TCC_MISS_sum": 0}

        result = backend._l2_hit_rate()
        assert result == 0.0


class TestCoalescingEfficiency:
    """Test memory coalescing efficiency computation"""

    def test_perfect_coalescing(self, backend):
        """100% coalescing (stride-1 access)"""
        backend._raw_data = {
            "SQ_INSTS_VMEM_RD": 100,
            "SQ_INSTS_VMEM_WR": 0,
            "TCP_TOTAL_ACCESSES_sum": 1600,  # 100 * 16
        }

        result = backend._coalescing_efficiency()
        assert result == 100.0

    def test_poor_coalescing(self, backend):
        """25% coalescing (completely uncoalesced float access)"""
        backend._raw_data = {
            "SQ_INSTS_VMEM_RD": 100,
            "SQ_INSTS_VMEM_WR": 0,
            "TCP_TOTAL_ACCESSES_sum": 6400,  # 4x more accesses
        }

        result = backend._coalescing_efficiency()
        assert result == 25.0

    def test_mixed_read_write(self, backend):
        """Coalescing with both reads and writes"""
        backend._raw_data = {
            "SQ_INSTS_VMEM_RD": 50,
            "SQ_INSTS_VMEM_WR": 50,
            "TCP_TOTAL_ACCESSES_sum": 1600,  # (50 + 50) * 16
        }

        result = backend._coalescing_efficiency()
        assert result == 100.0

    def test_no_memory_instructions(self, backend):
        """Handle zero memory instructions"""
        backend._raw_data = {
            "SQ_INSTS_VMEM_RD": 0,
            "SQ_INSTS_VMEM_WR": 0,
            "TCP_TOTAL_ACCESSES_sum": 1000,
        }

        result = backend._coalescing_efficiency()
        assert result == 0.0


class TestLDSBankConflicts:
    """Test LDS bank conflict computation"""

    def test_no_conflicts(self, backend):
        """Perfect LDS access pattern"""
        backend._raw_data = {"SQ_LDS_BANK_CONFLICT": 0, "SQ_INSTS_LDS": 1000}

        result = backend._lds_bank_conflicts()
        assert result == 0.0

    def test_high_conflicts(self, backend):
        """2 conflicts per instruction"""
        backend._raw_data = {"SQ_LDS_BANK_CONFLICT": 2000, "SQ_INSTS_LDS": 1000}

        result = backend._lds_bank_conflicts()
        assert result == 2.0

    def test_no_lds_instructions(self, backend):
        """Handle zero LDS instructions"""
        backend._raw_data = {"SQ_LDS_BANK_CONFLICT": 100, "SQ_INSTS_LDS": 0}

        result = backend._lds_bank_conflicts()
        assert result == 0.0


class TestBandwidthMetrics:
    """Test HBM bandwidth computations with 32B/64B/128B request granularity"""

    def test_hbm_read_bandwidth_64b_only(self, backend):
        """Test read bandwidth with only 64B requests"""
        arch = backend.device_specs.arch
        clock_mhz = backend.device_specs.base_clock_mhz

        # Time calculation based on architecture clock speed
        if arch == "gfx942":
            active_cycles = 2100000  # 1 ms at 2.1 GHz
            counters = {
                "TCC_EA_RDREQ_sum": 1000,
                "TCC_EA_RDREQ_32B_sum": 0,
                "TCC_BUBBLE_sum": 0,  # gfx942 has this counter
                "GRBM_GUI_ACTIVE": active_cycles,
            }
        else:  # gfx90a
            active_cycles = 1700000  # 1 ms at 1.7 GHz
            counters = {
                "TCC_EA_RDREQ_sum": 1000,
                "TCC_EA_RDREQ_32B_sum": 0,
                "GRBM_GUI_ACTIVE": active_cycles,
            }

        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._hbm_read_bandwidth()
        # (1000 * 64 bytes) / 0.001 seconds = 64 MB/s = 0.064 GB/s
        assert 0.06 < result < 0.07

    def test_hbm_read_bandwidth_mixed_sizes(self, backend):
        """Test read bandwidth with mixed request sizes"""
        arch = backend.device_specs.arch

        if arch == "gfx942":
            # MI300X with 128B bubble requests
            active_cycles = 2100000  # 1 ms at 2.1 GHz
            counters = {
                "TCC_EA_RDREQ_sum": 1000,
                "TCC_EA_RDREQ_32B_sum": 200,
                "TCC_BUBBLE_sum": 300,  # 300 × 128B = 38400 bytes
                "GRBM_GUI_ACTIVE": active_cycles,
            }
            # Remaining: 1000 - 200 - 300 = 500 × 64B = 32000 bytes
            # Total: 6400 + 38400 + 32000 = 76800 bytes
            expected_min, expected_max = 0.07, 0.08
        else:  # gfx90a
            # MI200 without 128B counter (all 64B or 32B)
            active_cycles = 1700000  # 1 ms at 1.7 GHz
            counters = {
                "TCC_EA_RDREQ_sum": 1000,
                "TCC_EA_RDREQ_32B_sum": 400,
                "GRBM_GUI_ACTIVE": active_cycles,
            }
            # 400 × 32B = 12800, 600 × 64B = 38400, Total = 51200 bytes
            expected_min, expected_max = 0.05, 0.06

        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._hbm_read_bandwidth()
        assert expected_min < result < expected_max

    def test_hbm_write_bandwidth_64b_only(self, backend):
        """Test write bandwidth with only 64B requests"""
        arch = backend.device_specs.arch

        if arch == "gfx942":
            active_cycles = 2100000  # 1 ms at 2.1 GHz
        else:  # gfx90a
            active_cycles = 1700000  # 1 ms at 1.7 GHz

        counters = {
            "TCC_EA_WRREQ_sum": 1000,
            "TCC_EA_WRREQ_64B_sum": 1000,
            "GRBM_GUI_ACTIVE": active_cycles,
        }

        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._hbm_write_bandwidth()
        # (1000 * 64 bytes) / 0.001 seconds = 64 MB/s = 0.064 GB/s
        assert 0.06 < result < 0.07

    def test_hbm_write_bandwidth_mixed_sizes(self, backend):
        """Test write bandwidth with mixed 32B and 64B requests"""
        arch = backend.device_specs.arch

        if arch == "gfx942":
            active_cycles = 2100000  # 1 ms at 2.1 GHz
        else:  # gfx90a
            active_cycles = 1700000  # 1 ms at 1.7 GHz

        counters = {
            "TCC_EA_WRREQ_sum": 1000,
            "TCC_EA_WRREQ_64B_sum": 600,
            "GRBM_GUI_ACTIVE": active_cycles,
        }
        # 600 × 64B = 38400, 400 × 32B = 12800, Total = 51200 bytes

        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._hbm_write_bandwidth()
        # 51200 / 1e9 / 0.001 = 0.0512 GB/s
        assert 0.05 < result < 0.06

    def test_zero_active_cycles(self, backend):
        """Handle zero active cycles"""
        counters = {"TCC_EA_RDREQ_sum": 1000, "TCC_EA_RDREQ_32B_sum": 0, "GRBM_GUI_ACTIVE": 0}

        # Add TCC_BUBBLE for gfx942
        if backend.device_specs.arch == "gfx942":
            counters["TCC_BUBBLE_sum"] = 0

        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._hbm_read_bandwidth()
        assert result == 0.0


class TestAtomicLatency:
    """Test L2 cache atomic operation latency computation"""

    def test_low_latency(self, backend):
        """10 cycles per atomic operation"""
        counters = {"TCC_EA_ATOMIC_sum": 1000, "TCC_EA_ATOMIC_LEVEL_sum": 10000}

        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._atomic_latency()
        # 10000 / 1000 = 10 cycles per atomic
        assert result == 10.0

    def test_high_latency(self, backend):
        """1000 cycles per atomic (contention)"""
        counters = {"TCC_EA_ATOMIC_sum": 100, "TCC_EA_ATOMIC_LEVEL_sum": 100000}

        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._atomic_latency()
        # 100000 / 100 = 1000 cycles per atomic
        assert result == 1000.0

    def test_no_atomics(self, backend):
        """Handle zero atomic instructions"""
        counters = {"TCC_EA_ATOMIC_sum": 0, "TCC_EA_ATOMIC_LEVEL_sum": 5000}

        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._atomic_latency()
        assert result == 0.0


class TestMetricDiscovery:
    """Test backend auto-discovers metrics"""

    def test_discovers_all_metrics(self, backend):
        """Backend should auto-discover all @metric decorated methods"""
        metrics = backend.get_available_metrics()

        # Should have all the metrics we defined
        assert "memory.l2_hit_rate" in metrics
        assert "memory.coalescing_efficiency" in metrics
        assert "memory.lds_bank_conflicts" in metrics
        assert "memory.hbm_read_bandwidth" in metrics

        # atomic_latency is architecture-specific
        if backend.device_specs.arch == "gfx90a":
            # On MI200, atomic_latency is unsupported (broken counter)
            assert "memory.atomic_latency" not in metrics
            assert "memory.atomic_latency" in backend._unsupported_metrics
        else:
            # On other architectures (gfx942, etc), it's supported
            assert "memory.atomic_latency" in metrics

    def test_get_required_counters(self, backend):
        """Backend should correctly report required counters for a metric"""
        counters = backend.get_required_counters(["memory.l2_hit_rate"])

        # Should require TCC_HIT_sum and TCC_MISS_sum (counter names appear in function signature)
        assert "TCC_HIT_sum" in counters
        assert "TCC_MISS_sum" in counters
        assert len(counters) == 2

    def test_discovers_compute_metrics(self, backend):
        """Backend should discover all compute metrics"""
        metrics = backend.get_available_metrics()

        assert "compute.total_flops" in metrics
        assert "compute.hbm_gflops" in metrics
        assert "compute.hbm_arithmetic_intensity" in metrics
        assert "compute.l2_arithmetic_intensity" in metrics
        assert "compute.l1_arithmetic_intensity" in metrics


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

        result = backend._total_flops()
        # 64 threads per wave * 100 instructions = 6400 FLOPS
        assert result == 6400

    def test_total_flops_fma_counts_double(self, backend):
        """Test that FMA instructions count as 2 operations"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_FMA_F32"] = 100

        result = backend._total_flops()
        # 64 threads * 100 FMA * 2 ops = 12800 FLOPS
        assert result == 12800

    def test_total_flops_mfma_high_throughput(self, backend):
        """Test MFMA instructions produce 512 operations each"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_MFMA_MOPS_F32"] = 10

        result = backend._total_flops()
        # 512 ops * 10 instructions = 5120 FLOPS
        assert result == 5120

    def test_total_flops_mixed_precision(self, backend):
        """Test FLOPS with mixed precision operations"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F16"] = 100  # 6400 FLOPS
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 50  # 3200 FLOPS
        backend._raw_data["SQ_INSTS_VALU_ADD_F64"] = 25  # 1600 FLOPS

        result = backend._total_flops()
        assert result == 6400 + 3200 + 1600

    def test_total_flops_zero(self, backend):
        """Handle zero FLOPS gracefully"""
        backend._raw_data = self._get_zero_flops_counters()

        result = backend._total_flops()
        assert result == 0

    def test_hbm_gflops_zero_time(self, backend):
        """Handle zero active cycles"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 1000
        backend._raw_data["GRBM_GUI_ACTIVE"] = 0

        result = backend._hbm_gflops()
        assert result == 0.0

    def test_hbm_arithmetic_intensity(self, backend):
        """Test HBM arithmetic intensity calculation"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 1000  # 64000 FLOPS

        counters = {
            "TCC_EA_RDREQ_sum": 1000,
            "TCC_EA_RDREQ_32B_sum": 0,
            "TCC_EA_WRREQ_sum": 0,
            "TCC_EA_WRREQ_64B_sum": 0,
        }

        # Add TCC_BUBBLE for gfx942
        if backend.device_specs.arch == "gfx942":
            counters["TCC_BUBBLE_sum"] = 0

        backend._raw_data.update(get_arch_counter_names(backend, counters))

        result = backend._hbm_arithmetic_intensity()
        # 64000 FLOPS / (1000 * 64 bytes) = 64000 / 64000 = 1.0 FLOP/byte
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

        if backend.device_specs.arch == "gfx942":
            counters["TCC_BUBBLE_sum"] = 0

        backend._raw_data.update(get_arch_counter_names(backend, counters))

        result = backend._hbm_arithmetic_intensity()
        assert result == 0.0

    def test_l2_arithmetic_intensity(self, backend):
        """Test L2 arithmetic intensity calculation"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 1000  # 64000 FLOPS
        backend._raw_data["TCC_REQ_sum"] = 500  # 500 * 128 = 64000 bytes

        result = backend._l2_arithmetic_intensity()
        # 64000 FLOPS / 64000 bytes = 1.0 FLOP/byte
        assert result == 1.0

    def test_l2_arithmetic_intensity_zero_bytes(self, backend):
        """Handle zero L2 bytes"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 1000
        backend._raw_data["TCC_REQ_sum"] = 0

        result = backend._l2_arithmetic_intensity()
        assert result == 0.0

    def test_l1_arithmetic_intensity(self, backend):
        """Test L1 arithmetic intensity calculation"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 1000  # 64000 FLOPS

        # L1 cache line size differs by architecture:
        # gfx942 (MI300X): 128 bytes
        # gfx90a (MI200): 64 bytes
        if backend.device_specs.arch == "gfx942":
            backend._raw_data["TCP_TOTAL_CACHE_ACCESSES_sum"] = 500  # 500 * 128 = 64000 bytes
        else:  # gfx90a
            backend._raw_data["TCP_TOTAL_CACHE_ACCESSES_sum"] = 1000  # 1000 * 64 = 64000 bytes

        result = backend._l1_arithmetic_intensity()
        # 64000 FLOPS / 64000 bytes = 1.0 FLOP/byte
        assert result == 1.0

    def test_l1_arithmetic_intensity_zero_bytes(self, backend):
        """Handle zero L1 bytes"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 1000
        backend._raw_data["TCP_TOTAL_CACHE_ACCESSES_sum"] = 0

        result = backend._l1_arithmetic_intensity()
        assert result == 0.0

    def test_high_arithmetic_intensity_compute_bound(self, backend):
        """Test high AI indicates compute-bound kernel"""
        backend._raw_data = self._get_zero_flops_counters()
        # Lots of compute, little memory
        backend._raw_data["SQ_INSTS_VALU_MFMA_MOPS_F32"] = 1000  # 512000 FLOPS

        counters = {
            "TCC_EA_RDREQ_sum": 100,
            "TCC_EA_RDREQ_32B_sum": 0,
            "TCC_EA_WRREQ_sum": 0,
            "TCC_EA_WRREQ_64B_sum": 0,
        }

        if backend.device_specs.arch == "gfx942":
            counters["TCC_BUBBLE_sum"] = 0

        backend._raw_data.update(get_arch_counter_names(backend, counters))

        result = backend._hbm_arithmetic_intensity()
        # 512000 / 6400 = 80 FLOP/byte (very compute-bound)
        assert result == 80.0

    def test_low_arithmetic_intensity_memory_bound(self, backend):
        """Test low AI indicates memory-bound kernel"""
        backend._raw_data = self._get_zero_flops_counters()
        # Little compute, lots of memory
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 100  # 6400 FLOPS

        counters = {
            "TCC_EA_RDREQ_sum": 10000,
            "TCC_EA_RDREQ_32B_sum": 0,
            "TCC_EA_WRREQ_sum": 0,
            "TCC_EA_WRREQ_64B_sum": 0,
        }

        if backend.device_specs.arch == "gfx942":
            counters["TCC_BUBBLE_sum"] = 0

        backend._raw_data.update(get_arch_counter_names(backend, counters))

        result = backend._hbm_arithmetic_intensity()
        # 6400 / 640000 = 0.01 FLOP/byte (very memory-bound)
        assert result == 0.01


# ---------------------------------------------------------------------------
# Microbenchmark-inspired validation tests
#
# These tests simulate analytically predictable counter patterns that mirror
# real GPU microbenchmarks (sequential read, strided access, pure FMA, etc.)
# and verify that derived metrics match expected values.
# ---------------------------------------------------------------------------


class TestSequentialReadPattern:
    """Simulate coalesced sequential read: stride-1, large array >> L2 cache"""

    def test_coalescing_is_perfect(self, backend):
        """Stride-1 read: each wavefront VMEM instruction -> 16 cache accesses"""
        backend._raw_data = {
            "SQ_INSTS_VMEM_RD": 1000,
            "SQ_INSTS_VMEM_WR": 0,
            "TCP_TOTAL_ACCESSES_sum": 16000,  # 1000 * 16 = perfectly coalesced
        }
        result = backend._coalescing_efficiency()
        assert result == 100.0

    def test_load_efficiency_coalesced(self, backend):
        """Coalesced reads: each VMEM_RD generates exactly 1 TCP read request"""
        backend._raw_data = {
            "SQ_INSTS_VMEM_RD": 1000,
            "TCP_TCC_READ_REQ_sum": 2000,  # 2 L2 requests per instruction
        }
        result = backend._global_load_efficiency()
        assert result == 50.0

    def test_l2_low_hit_rate_streaming(self, backend):
        """Array >> L2 cache: most accesses miss L2"""
        backend._raw_data = {"TCC_HIT_sum": 100, "TCC_MISS_sum": 900}
        result = backend._l2_hit_rate()
        assert result == 10.0

    def test_bytes_transferred_hbm_read_only(self, backend):
        """Read-only: bytes = read requests * request_size, zero writes"""
        arch = backend.device_specs.arch
        if arch == "gfx942":
            # All 64B reads, no 32B or 128B
            counters = {
                "TCC_EA_RDREQ_sum": 10000,
                "TCC_EA_RDREQ_32B_sum": 0,
                "TCC_BUBBLE_sum": 0,
                "TCC_EA_WRREQ_sum": 0,
                "TCC_EA_WRREQ_64B_sum": 0,
            }
        else:
            counters = {
                "TCC_EA_RDREQ_sum": 10000,
                "TCC_EA_RDREQ_32B_sum": 0,
                "TCC_EA_WRREQ_sum": 0,
                "TCC_EA_WRREQ_64B_sum": 0,
            }
        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._bytes_transferred_hbm()
        # 10000 * 64 = 640000 bytes
        assert result == 640000


class TestSequentialWritePattern:
    """Simulate coalesced sequential write: stride-1, write-only"""

    def test_store_efficiency(self, backend):
        """Each VMEM_WR generates multiple TCP write requests on gfx942"""
        backend._raw_data = {
            "SQ_INSTS_VMEM_WR": 1000,
            "TCP_TCC_WRITE_REQ_sum": 4000,  # 4 requests per instruction
        }
        result = backend._global_store_efficiency()
        assert result == 25.0

    def test_write_bandwidth_all_64b(self, backend):
        """Write-only with 64B requests"""
        arch = backend.device_specs.arch
        if arch == "gfx942":
            active_cycles = 2100000  # 1 ms
        else:
            active_cycles = 1700000
        counters = {
            "TCC_EA_WRREQ_sum": 10000,
            "TCC_EA_WRREQ_64B_sum": 10000,
            "GRBM_GUI_ACTIVE": active_cycles,
        }
        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._hbm_write_bandwidth()
        # 10000 * 64 = 640000 bytes / 0.001 s = 0.64 GB/s
        assert 0.63 < result < 0.65


class TestCopyPattern:
    """Simulate read-write copy: c[i] = a[i]"""

    def test_bytes_transferred_read_plus_write(self, backend):
        """Copy transfers bytes for both read and write"""
        arch = backend.device_specs.arch
        if arch == "gfx942":
            counters = {
                "TCC_EA_RDREQ_sum": 5000,
                "TCC_EA_RDREQ_32B_sum": 0,
                "TCC_BUBBLE_sum": 0,
                "TCC_EA_WRREQ_sum": 5000,
                "TCC_EA_WRREQ_64B_sum": 5000,
            }
        else:
            counters = {
                "TCC_EA_RDREQ_sum": 5000,
                "TCC_EA_RDREQ_32B_sum": 0,
                "TCC_EA_WRREQ_sum": 5000,
                "TCC_EA_WRREQ_64B_sum": 5000,
            }
        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._bytes_transferred_hbm()
        # read: 5000 * 64 = 320000, write: 5000 * 64 = 320000, total = 640000
        assert result == 640000

    def test_l1_hit_rate_with_writes(self, backend):
        """Copy kernel: L1 hit rate only accounts for read misses"""
        backend._raw_data = {
            "TCP_TCC_READ_REQ_sum": 250,
            "TCP_TOTAL_CACHE_ACCESSES_sum": 1000,
        }
        result = backend._l1_hit_rate()
        # (1000 - 250) / 1000 * 100 = 75%
        assert result == 75.0

    def test_bandwidth_utilization(self, backend):
        """Bandwidth utilization = actual_bw / peak_bw * 100"""
        arch = backend.device_specs.arch
        if arch == "gfx942":
            active_cycles = 2100000  # 1 ms
            peak_bw = 5300.0
            counters = {
                "TCC_EA_RDREQ_sum": 10000,
                "TCC_EA_RDREQ_32B_sum": 0,
                "TCC_BUBBLE_sum": 0,
                "TCC_EA_WRREQ_sum": 10000,
                "TCC_EA_WRREQ_64B_sum": 10000,
                "GRBM_GUI_ACTIVE": active_cycles,
            }
        else:
            active_cycles = 1700000
            peak_bw = 3200.0
            counters = {
                "TCC_EA_RDREQ_sum": 10000,
                "TCC_EA_RDREQ_32B_sum": 0,
                "TCC_EA_WRREQ_sum": 10000,
                "TCC_EA_WRREQ_64B_sum": 10000,
                "GRBM_GUI_ACTIVE": active_cycles,
            }
        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._hbm_bandwidth_utilization()
        # total = 20000 * 64 = 1.28 MB / 0.001 s = 1.28 GB/s
        # utilization = 1.28 / peak_bw * 100
        expected = (1.28 / peak_bw) * 100
        assert abs(result - expected) < 0.01


class TestStridedAccessPattern:
    """Simulate strided access at various strides"""

    @pytest.mark.parametrize(
        "stride, total_accesses, expected_efficiency",
        [
            (1, 16000, 100.0),   # stride-1: perfectly coalesced
            (2, 32000, 50.0),    # stride-2: 2x the cache accesses
            (4, 64000, 25.0),    # stride-4: 4x the cache accesses
        ],
    )
    def test_coalescing_varies_with_stride(
        self, backend, stride, total_accesses, expected_efficiency
    ):
        """Coalescing efficiency inversely proportional to stride"""
        backend._raw_data = {
            "SQ_INSTS_VMEM_RD": 1000,
            "SQ_INSTS_VMEM_WR": 0,
            "TCP_TOTAL_ACCESSES_sum": total_accesses,
        }
        result = backend._coalescing_efficiency()
        assert result == expected_efficiency


class TestL2ResidentPattern:
    """Simulate L2-resident array: small array iterated many times"""

    def test_high_l2_hit_rate(self, backend):
        """Array fits in L2, iterated many times: mostly hits"""
        backend._raw_data = {"TCC_HIT_sum": 9900, "TCC_MISS_sum": 100}
        result = backend._l2_hit_rate()
        assert result == 99.0

    def test_l2_bandwidth_from_hit_miss(self, backend):
        """L2 BW computed from (hits + misses) * 128 bytes / time"""
        arch = backend.device_specs.arch
        if arch == "gfx942":
            active_cycles = 2100000  # 1 ms
        else:
            active_cycles = 1700000
        backend._raw_data = {
            "TCC_HIT_sum": 9000,
            "TCC_MISS_sum": 1000,
            "GRBM_GUI_ACTIVE": active_cycles,
        }
        result = backend._l2_bandwidth()
        # (10000 * 128) / 1e9 / 0.001 = 1.28 GB/s
        assert 1.27 < result < 1.29


class TestL1ResidentPattern:
    """Simulate L1-resident array: tiny array per workgroup, many iterations"""

    def test_high_l1_hit_rate(self, backend):
        """8KB per WG in L1, iterated 200 times: ~99.5% hit rate"""
        # First pass misses, remaining 199 hit
        backend._raw_data = {
            "TCP_TCC_READ_REQ_sum": 100,     # misses (first pass only)
            "TCP_TOTAL_CACHE_ACCESSES_sum": 20000,  # 200 passes * 100 accesses
        }
        result = backend._l1_hit_rate()
        # (20000 - 100) / 20000 * 100 = 99.5%
        assert result == 99.5


class TestLDSConflictPattern:
    """Simulate LDS bank conflict scenarios"""

    def test_conflict_free_vs_high_conflict(self, backend):
        """Conflict-free should be 0, high-conflict should be >> 0"""
        # Conflict-free
        backend._raw_data = {"SQ_LDS_BANK_CONFLICT": 0, "SQ_INSTS_LDS": 10000}
        conflict_free = backend._lds_bank_conflicts()

        # High conflict (stride-32: all threads hit same bank)
        backend._raw_data = {"SQ_LDS_BANK_CONFLICT": 36000, "SQ_INSTS_LDS": 10000}
        high_conflict = backend._lds_bank_conflicts()

        assert conflict_free == 0.0
        assert high_conflict == 3.6
        assert high_conflict > conflict_free


class TestPureFMAPattern:
    """Simulate pure FP32 FMA loop in registers (no memory)"""

    def _get_zero_flops_counters(self):
        """Helper: all FLOPS counters zeroed"""
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

    def test_flops_exact_for_fma_loop(self, backend):
        """N FMA instructions per wave = N * 2 * 64 FLOPS"""
        backend._raw_data = self._get_zero_flops_counters()
        # Simulate 100K FMA instructions counted by rocprofv3
        # (counter is per-wave total, summed across all waves)
        num_waves = 1216  # 304 CUs * 4 waves/CU
        fma_per_wave = 100000
        backend._raw_data["SQ_INSTS_VALU_FMA_F32"] = num_waves * fma_per_wave

        result = backend._total_flops()
        expected = num_waves * fma_per_wave * 2 * 64
        assert result == expected

    def test_high_arithmetic_intensity_pure_compute(self, backend):
        """Pure compute kernel: very high AI (near-zero memory)"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_FMA_F32"] = 100000  # lots of compute

        # Minimal memory (just the output store)
        counters = {
            "TCC_EA_RDREQ_sum": 1,
            "TCC_EA_RDREQ_32B_sum": 0,
            "TCC_EA_WRREQ_sum": 1,
            "TCC_EA_WRREQ_64B_sum": 1,
        }
        if backend.device_specs.arch == "gfx942":
            counters["TCC_BUBBLE_sum"] = 0
        backend._raw_data.update(get_arch_counter_names(backend, counters))

        result = backend._hbm_arithmetic_intensity()
        # FLOPS = 100000 * 2 * 64 = 12800000
        # bytes = 1*64 + 1*64 = 128
        # AI = 12800000 / 128 = 100000
        assert result == 100000.0

    def test_gflops_with_duration(self, backend):
        """GFLOPS = total_flops / duration"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_FMA_F32"] = 1000000  # many FMAs

        # Set duration (used by hbm_gflops)
        backend._current_duration_us = 1000.0  # 1 ms

        result = backend._hbm_gflops()
        # flops = 1000000 * 2 * 64 = 128000000
        # gflops = 128000000 / 1e9 / 0.001 = 128.0
        assert result == 128.0


class TestMFMAPattern:
    """Simulate MFMA instructions"""

    def _get_zero_flops_counters(self):
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

    def test_mfma_512_ops_per_instruction(self, backend):
        """Each MFMA instruction produces 512 FLOPS"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_MFMA_MOPS_F32"] = 1000
        result = backend._total_flops()
        assert result == 1000 * 512

    def test_mfma_plus_valu_combined(self, backend):
        """MFMA and VALU FLOPS are additive"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_MFMA_MOPS_F32"] = 100  # 51200 FLOPS
        backend._raw_data["SQ_INSTS_VALU_ADD_F32"] = 200  # 12800 FLOPS
        result = backend._total_flops()
        assert result == 100 * 512 + 200 * 64

    def test_mfma_all_precisions(self, backend):
        """MFMA across F16, BF16, F32, F64 all count at 512 ops"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_MFMA_MOPS_F16"] = 10
        backend._raw_data["SQ_INSTS_VALU_MFMA_MOPS_BF16"] = 20
        backend._raw_data["SQ_INSTS_VALU_MFMA_MOPS_F32"] = 30
        backend._raw_data["SQ_INSTS_VALU_MFMA_MOPS_F64"] = 40
        result = backend._total_flops()
        assert result == (10 + 20 + 30 + 40) * 512


class TestAtomicContentionPattern:
    """Simulate high-contention global atomics"""

    def test_high_contention_high_latency(self, backend):
        """All threads atomic to one address: latency >> 1"""
        counters = {
            "TCC_EA_ATOMIC_sum": 1000,
            "TCC_EA_ATOMIC_LEVEL_sum": 22000000,  # ~22K cycles per atomic
        }
        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._atomic_latency()
        assert result == 22000.0

    def test_low_contention_low_latency(self, backend):
        """Per-WG atomics (no contention): latency much lower"""
        counters = {
            "TCC_EA_ATOMIC_sum": 1000,
            "TCC_EA_ATOMIC_LEVEL_sum": 50000,  # ~50 cycles per atomic
        }
        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._atomic_latency()
        assert result == 50.0


class TestMixedComputeMemoryPattern:
    """Simulate tunable arithmetic intensity: K FMAs per load"""

    def _get_zero_flops_counters(self):
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

    @pytest.mark.parametrize(
        "K, expected_ai",
        [
            (1, 0.25),    # AI = K/4
            (10, 2.5),
            (100, 25.0),
            (1000, 250.0),
        ],
    )
    def test_arithmetic_intensity_scales_with_k(self, backend, K, expected_ai):
        """AI = K * 2 * 64 / (2 * 64 * 4) = K / 4 FLOP/byte"""
        num_elements = 1024  # must be multiple of 64 for exact wave count
        # Each thread: 1 read (4 bytes) + K FMAs + 1 write (4 bytes)
        # Per wave (64 threads): 1 VMEM_RD + K FMAs (rocprofv3 counts per-wave)
        num_waves = num_elements // 64

        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data["SQ_INSTS_VALU_FMA_F32"] = num_waves * K

        # HBM counters: 1 read + 1 write per thread, all 64B requests
        # Total threads = num_waves * 64 = num_elements
        # Each thread reads 4 bytes, writes 4 bytes
        # At 64B granularity: (num_elements * 4) / 64 requests each
        rd_requests = (num_elements * 4) // 64
        wr_requests = (num_elements * 4) // 64

        counters = {
            "TCC_EA_RDREQ_sum": rd_requests,
            "TCC_EA_RDREQ_32B_sum": 0,
            "TCC_EA_WRREQ_sum": wr_requests,
            "TCC_EA_WRREQ_64B_sum": wr_requests,
        }
        if backend.device_specs.arch == "gfx942":
            counters["TCC_BUBBLE_sum"] = 0
        backend._raw_data.update(get_arch_counter_names(backend, counters))

        result = backend._hbm_arithmetic_intensity()
        # flops = num_waves * K * 2 * 64 = num_elements * K * 2
        # bytes = (rd_requests + wr_requests) * 64 = num_elements * 8
        # AI = num_elements * K * 2 / (num_elements * 8) = K / 4
        assert abs(result - expected_ai) / max(expected_ai, 1e-9) < 0.05

    def test_bytes_transferred_l2_scales_with_accesses(self, backend):
        """L2 bytes = TCC_REQ * 128"""
        backend._raw_data = {"TCC_REQ_sum": 10000}
        result = backend._bytes_transferred_l2()
        assert result == 10000 * 128

    def test_bytes_transferred_l1_gfx942(self, backend):
        """L1 bytes = TCP_TOTAL_CACHE_ACCESSES * cache_line_size"""
        backend._raw_data = {"TCP_TOTAL_CACHE_ACCESSES_sum": 10000}
        result = backend._bytes_transferred_l1()
        if backend.device_specs.arch == "gfx942":
            assert result == 10000 * 128  # 128B cache lines on gfx942
        else:
            assert result == 10000 * 64  # 64B cache lines on gfx90a
