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
        backend._raw_data = {
            'TCC_HIT_sum': 1000,
            'TCC_MISS_sum': 0
        }

        result = backend._l2_hit_rate()
        assert result == 100.0

    def test_zero_hit_rate(self, backend):
        """0% hit rate (all misses)"""
        backend._raw_data = {
            'TCC_HIT_sum': 0,
            'TCC_MISS_sum': 1000
        }

        result = backend._l2_hit_rate()
        assert result == 0.0

    def test_fifty_percent_hit_rate(self, backend):
        """50% hit rate"""
        backend._raw_data = {
            'TCC_HIT_sum': 500,
            'TCC_MISS_sum': 500
        }

        result = backend._l2_hit_rate()
        assert result == 50.0

    def test_no_accesses(self, backend):
        """Handle zero total accesses"""
        backend._raw_data = {
            'TCC_HIT_sum': 0,
            'TCC_MISS_sum': 0
        }

        result = backend._l2_hit_rate()
        assert result == 0.0


class TestCoalescingEfficiency:
    """Test memory coalescing efficiency computation"""

    def test_perfect_coalescing(self, backend):
        """100% coalescing (stride-1 access)"""
        backend._raw_data = {
            'SQ_INSTS_VMEM_RD': 100,
            'SQ_INSTS_VMEM_WR': 0,
            'TCP_TOTAL_ACCESSES_sum': 1600  # 100 * 16
        }

        result = backend._coalescing_efficiency()
        assert result == 100.0

    def test_poor_coalescing(self, backend):
        """25% coalescing (completely uncoalesced float access)"""
        backend._raw_data = {
            'SQ_INSTS_VMEM_RD': 100,
            'SQ_INSTS_VMEM_WR': 0,
            'TCP_TOTAL_ACCESSES_sum': 6400  # 4x more accesses
        }

        result = backend._coalescing_efficiency()
        assert result == 25.0

    def test_mixed_read_write(self, backend):
        """Coalescing with both reads and writes"""
        backend._raw_data = {
            'SQ_INSTS_VMEM_RD': 50,
            'SQ_INSTS_VMEM_WR': 50,
            'TCP_TOTAL_ACCESSES_sum': 1600  # (50 + 50) * 16
        }

        result = backend._coalescing_efficiency()
        assert result == 100.0

    def test_no_memory_instructions(self, backend):
        """Handle zero memory instructions"""
        backend._raw_data = {
            'SQ_INSTS_VMEM_RD': 0,
            'SQ_INSTS_VMEM_WR': 0,
            'TCP_TOTAL_ACCESSES_sum': 1000
        }

        result = backend._coalescing_efficiency()
        assert result == 0.0


class TestLDSBankConflicts:
    """Test LDS bank conflict computation"""

    def test_no_conflicts(self, backend):
        """Perfect LDS access pattern"""
        backend._raw_data = {
            'SQ_LDS_BANK_CONFLICT': 0,
            'SQ_INSTS_LDS': 1000
        }

        result = backend._lds_bank_conflicts()
        assert result == 0.0

    def test_high_conflicts(self, backend):
        """2 conflicts per instruction"""
        backend._raw_data = {
            'SQ_LDS_BANK_CONFLICT': 2000,
            'SQ_INSTS_LDS': 1000
        }

        result = backend._lds_bank_conflicts()
        assert result == 2.0

    def test_no_lds_instructions(self, backend):
        """Handle zero LDS instructions"""
        backend._raw_data = {
            'SQ_LDS_BANK_CONFLICT': 100,
            'SQ_INSTS_LDS': 0
        }

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
                'TCC_EA_RDREQ_sum': 1000,
                'TCC_EA_RDREQ_32B_sum': 0,
                'TCC_BUBBLE_sum': 0,  # gfx942 has this counter
                'GRBM_GUI_ACTIVE': active_cycles
            }
        else:  # gfx90a
            active_cycles = 1700000  # 1 ms at 1.7 GHz
            counters = {
                'TCC_EA_RDREQ_sum': 1000,
                'TCC_EA_RDREQ_32B_sum': 0,
                'GRBM_GUI_ACTIVE': active_cycles
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
                'TCC_EA_RDREQ_sum': 1000,
                'TCC_EA_RDREQ_32B_sum': 200,
                'TCC_BUBBLE_sum': 300,  # 300 × 128B = 38400 bytes
                'GRBM_GUI_ACTIVE': active_cycles
            }
            # Remaining: 1000 - 200 - 300 = 500 × 64B = 32000 bytes
            # Total: 6400 + 38400 + 32000 = 76800 bytes
            expected_min, expected_max = 0.07, 0.08
        else:  # gfx90a
            # MI200 without 128B counter (all 64B or 32B)
            active_cycles = 1700000  # 1 ms at 1.7 GHz
            counters = {
                'TCC_EA_RDREQ_sum': 1000,
                'TCC_EA_RDREQ_32B_sum': 400,
                'GRBM_GUI_ACTIVE': active_cycles
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
            'TCC_EA_WRREQ_sum': 1000,
            'TCC_EA_WRREQ_64B_sum': 1000,
            'GRBM_GUI_ACTIVE': active_cycles
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
            'TCC_EA_WRREQ_sum': 1000,
            'TCC_EA_WRREQ_64B_sum': 600,
            'GRBM_GUI_ACTIVE': active_cycles
        }
        # 600 × 64B = 38400, 400 × 32B = 12800, Total = 51200 bytes
        
        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._hbm_write_bandwidth()
        # 51200 / 1e9 / 0.001 = 0.0512 GB/s
        assert 0.05 < result < 0.06

    def test_zero_active_cycles(self, backend):
        """Handle zero active cycles"""
        counters = {
            'TCC_EA_RDREQ_sum': 1000,
            'TCC_EA_RDREQ_32B_sum': 0,
            'GRBM_GUI_ACTIVE': 0
        }
        
        # Add TCC_BUBBLE for gfx942
        if backend.device_specs.arch == "gfx942":
            counters['TCC_BUBBLE_sum'] = 0
        
        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._hbm_read_bandwidth()
        assert result == 0.0


class TestAtomicLatency:
    """Test L2 cache atomic operation latency computation"""

    def test_low_latency(self, backend):
        """10 cycles per atomic operation"""
        counters = {
            'TCC_EA_ATOMIC_sum': 1000,
            'TCC_EA_ATOMIC_LEVEL_sum': 10000
        }
        
        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._atomic_latency()
        # 10000 / 1000 = 10 cycles per atomic
        assert result == 10.0

    def test_high_latency(self, backend):
        """1000 cycles per atomic (contention)"""
        counters = {
            'TCC_EA_ATOMIC_sum': 100,
            'TCC_EA_ATOMIC_LEVEL_sum': 100000
        }
        
        backend._raw_data = get_arch_counter_names(backend, counters)
        result = backend._atomic_latency()
        # 100000 / 100 = 1000 cycles per atomic
        assert result == 1000.0

    def test_no_atomics(self, backend):
        """Handle zero atomic instructions"""
        counters = {
            'TCC_EA_ATOMIC_sum': 0,
            'TCC_EA_ATOMIC_LEVEL_sum': 5000
        }
        
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
            'SQ_INSTS_VALU_ADD_F16': 0, 'SQ_INSTS_VALU_MUL_F16': 0,
            'SQ_INSTS_VALU_TRANS_F16': 0, 'SQ_INSTS_VALU_FMA_F16': 0,
            'SQ_INSTS_VALU_ADD_F32': 0, 'SQ_INSTS_VALU_MUL_F32': 0,
            'SQ_INSTS_VALU_TRANS_F32': 0, 'SQ_INSTS_VALU_FMA_F32': 0,
            'SQ_INSTS_VALU_ADD_F64': 0, 'SQ_INSTS_VALU_MUL_F64': 0,
            'SQ_INSTS_VALU_TRANS_F64': 0, 'SQ_INSTS_VALU_FMA_F64': 0,
            'SQ_INSTS_VALU_MFMA_MOPS_F16': 0, 'SQ_INSTS_VALU_MFMA_MOPS_BF16': 0,
            'SQ_INSTS_VALU_MFMA_MOPS_F32': 0, 'SQ_INSTS_VALU_MFMA_MOPS_F64': 0,
        }

    def test_total_flops_fp32_add(self, backend):
        """Test FLOPS calculation with FP32 add instructions"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data['SQ_INSTS_VALU_ADD_F32'] = 100

        result = backend._total_flops()
        # 64 threads per wave * 100 instructions = 6400 FLOPS
        assert result == 6400

    def test_total_flops_fma_counts_double(self, backend):
        """Test that FMA instructions count as 2 operations"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data['SQ_INSTS_VALU_FMA_F32'] = 100

        result = backend._total_flops()
        # 64 threads * 100 FMA * 2 ops = 12800 FLOPS
        assert result == 12800

    def test_total_flops_mfma_high_throughput(self, backend):
        """Test MFMA instructions produce 512 operations each"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data['SQ_INSTS_VALU_MFMA_MOPS_F32'] = 10

        result = backend._total_flops()
        # 512 ops * 10 instructions = 5120 FLOPS
        assert result == 5120

    def test_total_flops_mixed_precision(self, backend):
        """Test FLOPS with mixed precision operations"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data['SQ_INSTS_VALU_ADD_F16'] = 100  # 6400 FLOPS
        backend._raw_data['SQ_INSTS_VALU_ADD_F32'] = 50   # 3200 FLOPS
        backend._raw_data['SQ_INSTS_VALU_ADD_F64'] = 25   # 1600 FLOPS

        result = backend._total_flops()
        assert result == 6400 + 3200 + 1600

    def test_total_flops_zero(self, backend):
        """Handle zero FLOPS gracefully"""
        backend._raw_data = self._get_zero_flops_counters()

        result = backend._total_flops()
        assert result == 0

    def test_hbm_gflops_calculation(self, backend):
        """Test GFLOPS calculation with timing"""
        arch = backend.device_specs.arch
        
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data['SQ_INSTS_VALU_ADD_F32'] = 1000000  # 64M FLOPS
        
        if arch == "gfx942":
            backend._raw_data['GRBM_GUI_ACTIVE'] = 2100000  # 1 ms at 2.1 GHz
        else:  # gfx90a
            backend._raw_data['GRBM_GUI_ACTIVE'] = 1700000  # 1 ms at 1.7 GHz

        result = backend._hbm_gflops()
        # 64M FLOPS / 0.001 seconds = 64 GFLOPS
        assert 60 < result < 70

    def test_hbm_gflops_zero_time(self, backend):
        """Handle zero active cycles"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data['SQ_INSTS_VALU_ADD_F32'] = 1000
        backend._raw_data['GRBM_GUI_ACTIVE'] = 0

        result = backend._hbm_gflops()
        assert result == 0.0

    def test_hbm_arithmetic_intensity(self, backend):
        """Test HBM arithmetic intensity calculation"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data['SQ_INSTS_VALU_ADD_F32'] = 1000  # 64000 FLOPS
        
        counters = {
            'TCC_EA_RDREQ_sum': 1000,
            'TCC_EA_RDREQ_32B_sum': 0,
            'TCC_EA_WRREQ_sum': 0,
            'TCC_EA_WRREQ_64B_sum': 0
        }
        
        # Add TCC_BUBBLE for gfx942
        if backend.device_specs.arch == "gfx942":
            counters['TCC_BUBBLE_sum'] = 0
        
        backend._raw_data.update(get_arch_counter_names(backend, counters))

        result = backend._hbm_arithmetic_intensity()
        # 64000 FLOPS / (1000 * 64 bytes) = 64000 / 64000 = 1.0 FLOP/byte
        assert result == 1.0

    def test_hbm_arithmetic_intensity_zero_bytes(self, backend):
        """Handle zero HBM bytes transferred"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data['SQ_INSTS_VALU_ADD_F32'] = 1000
        
        counters = {
            'TCC_EA_RDREQ_sum': 0,
            'TCC_EA_RDREQ_32B_sum': 0,
            'TCC_EA_WRREQ_sum': 0,
            'TCC_EA_WRREQ_64B_sum': 0
        }
        
        if backend.device_specs.arch == "gfx942":
            counters['TCC_BUBBLE_sum'] = 0
        
        backend._raw_data.update(get_arch_counter_names(backend, counters))

        result = backend._hbm_arithmetic_intensity()
        assert result == 0.0

    def test_l2_arithmetic_intensity(self, backend):
        """Test L2 arithmetic intensity calculation"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data['SQ_INSTS_VALU_ADD_F32'] = 1000  # 64000 FLOPS
        backend._raw_data['TCC_REQ_sum'] = 500  # 500 * 128 = 64000 bytes

        result = backend._l2_arithmetic_intensity()
        # 64000 FLOPS / 64000 bytes = 1.0 FLOP/byte
        assert result == 1.0

    def test_l2_arithmetic_intensity_zero_bytes(self, backend):
        """Handle zero L2 bytes"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data['SQ_INSTS_VALU_ADD_F32'] = 1000
        backend._raw_data['TCC_REQ_sum'] = 0

        result = backend._l2_arithmetic_intensity()
        assert result == 0.0

    def test_l1_arithmetic_intensity(self, backend):
        """Test L1 arithmetic intensity calculation"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data['SQ_INSTS_VALU_ADD_F32'] = 1000  # 64000 FLOPS
        
        # L1 cache line size differs by architecture:
        # gfx942 (MI300X): 128 bytes
        # gfx90a (MI200): 64 bytes
        if backend.device_specs.arch == "gfx942":
            backend._raw_data['TCP_TOTAL_CACHE_ACCESSES_sum'] = 500  # 500 * 128 = 64000 bytes
        else:  # gfx90a
            backend._raw_data['TCP_TOTAL_CACHE_ACCESSES_sum'] = 1000  # 1000 * 64 = 64000 bytes

        result = backend._l1_arithmetic_intensity()
        # 64000 FLOPS / 64000 bytes = 1.0 FLOP/byte
        assert result == 1.0

    def test_l1_arithmetic_intensity_zero_bytes(self, backend):
        """Handle zero L1 bytes"""
        backend._raw_data = self._get_zero_flops_counters()
        backend._raw_data['SQ_INSTS_VALU_ADD_F32'] = 1000
        backend._raw_data['TCP_TOTAL_CACHE_ACCESSES_sum'] = 0

        result = backend._l1_arithmetic_intensity()
        assert result == 0.0

    def test_high_arithmetic_intensity_compute_bound(self, backend):
        """Test high AI indicates compute-bound kernel"""
        backend._raw_data = self._get_zero_flops_counters()
        # Lots of compute, little memory
        backend._raw_data['SQ_INSTS_VALU_MFMA_MOPS_F32'] = 1000  # 512000 FLOPS
        
        counters = {
            'TCC_EA_RDREQ_sum': 100,
            'TCC_EA_RDREQ_32B_sum': 0,
            'TCC_EA_WRREQ_sum': 0,
            'TCC_EA_WRREQ_64B_sum': 0
        }
        
        if backend.device_specs.arch == "gfx942":
            counters['TCC_BUBBLE_sum'] = 0
        
        backend._raw_data.update(get_arch_counter_names(backend, counters))

        result = backend._hbm_arithmetic_intensity()
        # 512000 / 6400 = 80 FLOP/byte (very compute-bound)
        assert result == 80.0

    def test_low_arithmetic_intensity_memory_bound(self, backend):
        """Test low AI indicates memory-bound kernel"""
        backend._raw_data = self._get_zero_flops_counters()
        # Little compute, lots of memory
        backend._raw_data['SQ_INSTS_VALU_ADD_F32'] = 100  # 6400 FLOPS
        
        counters = {
            'TCC_EA_RDREQ_sum': 10000,
            'TCC_EA_RDREQ_32B_sum': 0,
            'TCC_EA_WRREQ_sum': 0,
            'TCC_EA_WRREQ_64B_sum': 0
        }
        
        if backend.device_specs.arch == "gfx942":
            counters['TCC_BUBBLE_sum'] = 0
        
        backend._raw_data.update(get_arch_counter_names(backend, counters))

        result = backend._hbm_arithmetic_intensity()
        # 6400 / 640000 = 0.01 FLOP/byte (very memory-bound)
        assert result == 0.01



