"""
GFX1201 (RDNA3) Backend

Each metric is defined with @metric decorator.
"""

from .base import CounterBackend, DeviceSpecs, ProfileResult
from ..utils.common import split_counters_into_passes
from .decorator import metric
from ..profiler.rocprof_wrapper import ROCProfV3Wrapper
from typing import List, Optional, Dict


class GFX1201Backend(CounterBackend):
    """
    AMD RDNA (gfx1201) counter backend

    All metrics are defined with @metric decorator.
    Hardware counter names appear ONLY as function parameter names.
    """

    def _get_device_specs(self) -> DeviceSpecs:
        """RDNA3 (gfx1201) specifications - AMD Radeon Graphics"""
        return DeviceSpecs(
            arch="gfx1201",
            name="AMD Radeon Graphics (RDNA3)",
            num_cu=64,  # From rocminfo
            max_waves_per_cu=32,  # RDNA3 standard
            wavefront_size=32,  # RDNA3 uses wave32 by default
            base_clock_mhz=2420.0,  # From rocminfo: Max Clock Freq
            hbm_bandwidth_gbs=664.6,
            l2_bandwidth_gbs=2000.0,  # TODO: Fix this value
            l2_size_mb=8.0,  # From rocminfo: L2 = 8192 KB
            lds_size_per_cu_kb=64.0,  # TODO: Fix this value
            fp32_tflops=39.0,  # Estimated: 64 CU * 2 SIMD * 32 lanes * 2.42 GHz * 2 (FMA)
            fp64_tflops=1.129,
            int8_tops=312.0,  # TODO: Fix this value
            boost_clock_mhz=2420,  # From rocminfo
        )

    def _get_counter_groups(self, counters: List[str]) -> List[List[str]]:
        """
        Group counters into passes using RDNA3-specific block limits.

        This keeps the hardware-specific knowledge (block limits and naming)
        in the gfx1201 backend while reusing the generic helper from
        `common.py` for the actual bin-packing.
        """
        from ..logger import logger

        block_limits = self._get_counter_block_limits()
        return split_counters_into_passes(
            counters,
            block_limits=block_limits,
            get_counter_block=self._get_counter_block,
            logger=logger,
        )

    def _get_counter_block_limits(self) -> Dict[str, int]:
        """
        Return per-hardware-block counter limits for gfx1201 (RDNA3).

        These limits define how many performance counters can be simultaneously
        collected from each hardware block in a single profiling pass.

        Hardware blocks on RDNA3:
        - SQ (Shader Queue): Instruction counters, wave management
        - SQC (Shader Queue Cache): LDS operations, instruction cache
        - TA (Texture Addresser): Texture address operations
        - TCP (Texture Cache per Pipe): L0 vector cache
        - GL2C (Global L2 Cache): L2 cache and memory controller
        - GRBM (Graphics Register Bus Manager): Global GPU activity

        Returns:
            Dict mapping block_name -> max_counters_per_pass
        """
        return {
            "SQ": 8,  # Shader Queue - wave and instruction counters
            "SQC": 4,  # Shader Queue Cache - LDS and instruction cache
            "TA": 2,  # Texture Addresser
            "TCP": 4,  # L0 Cache (Texture Cache per Pipe)
            "GL2C": 4,  # L2 Cache / Memory Controller
            "GRBM": 2,  # Graphics Register Bus Manager
        }

    def _run_rocprof(
        self,
        command: str,
        counters: List[str],
        kernel_filter: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout_seconds: Optional[int] = 0,
    ) -> List[ProfileResult]:
        """Run rocprofv3 and return results"""
        wrapper = ROCProfV3Wrapper(timeout_seconds=timeout_seconds)
        return wrapper.profile(command, counters, kernel_filter=kernel_filter, cwd=cwd)

    # Memory bandwidth metrics

    @metric("memory.hbm_read_bandwidth")
    def _hbm_read_bandwidth(self, GL2C_MISS_sum, GRBM_GUI_ACTIVE):
        """
        Memory read bandwidth in GB/s (estimated from L2 misses)

        Formula: (L2_misses * 128 bytes) / time_seconds

        Counters:
        - GL2C_MISS_sum: L2 cache misses (go to memory)
        - GRBM_GUI_ACTIVE: Number of active GPU cycles

        Note: GL2C_EA_RDREQ counters return zero on gfx1201 (RDNA3).
        Using L2 misses as proxy - misses trigger memory reads.
        This underestimates true bandwidth as it doesn't include writes.
        """
        # L2 misses go to memory (128-byte cache lines)
        bytes_read = GL2C_MISS_sum * 128

        if GRBM_GUI_ACTIVE == 0:
            return 0.0

        time_seconds = GRBM_GUI_ACTIVE / (self.device_specs.base_clock_mhz * 1e6)
        return (bytes_read / 1e9) / time_seconds if time_seconds > 0 else 0.0

    @metric(
        "memory.hbm_write_bandwidth",
        unsupported_reason="GL2C_EA_WRREQ counters return zero on gfx1201 (RDNA3)",
    )
    def _hbm_write_bandwidth(self):
        """
        Memory write bandwidth in GB/s

        Note: GL2C_EA_WRREQ_64B_sum returns zero on gfx1201 (RDNA3).
        No reliable proxy available from L2 cache counters for write bandwidth.
        """
        return 0.0

    @metric("memory.hbm_bandwidth_utilization")
    def _hbm_bandwidth_utilization(self, GL2C_MISS_sum, GRBM_GUI_ACTIVE):
        """
        Memory bandwidth utilization as percentage of peak (estimated from L2 misses)

        Formula: ((L2_miss_traffic) / peak_bandwidth) * 100

        Counters:
        - GL2C_MISS_sum: L2 cache misses (go to memory)
        - GRBM_GUI_ACTIVE: Number of active GPU cycles

        Note: GL2C_EA read/write request counters return zero on gfx1201 (RDNA3).
        Using L2 misses as proxy - each miss transfers 128 bytes from memory.
        This provides a lower bound estimate (doesn't include write-backs).
        """
        # L2 misses trigger memory reads (128-byte cache lines)
        bytes_from_memory = GL2C_MISS_sum * 128

        if GRBM_GUI_ACTIVE == 0:
            return 0.0

        time_seconds = GRBM_GUI_ACTIVE / (self.device_specs.base_clock_mhz * 1e6)
        actual_bandwidth = (bytes_from_memory / 1e9) / time_seconds if time_seconds > 0 else 0.0

        return (actual_bandwidth / self.device_specs.hbm_bandwidth_gbs) * 100

    @metric("memory.bytes_transferred_hbm")
    def _bytes_transferred_hbm(self, GL2C_MISS_sum):
        """
        Total bytes transferred from memory (estimated from L2 misses)

        Formula: L2_misses * 128 bytes

        Counters:
        - GL2C_MISS_sum: L2 cache misses (each triggers 128-byte read from memory)

        Note: GL2C_EA read/write request counters return zero on gfx1201 (RDNA3).
        Using L2 misses as proxy. This counts read traffic only (misses).
        """
        return GL2C_MISS_sum * 128

    @metric("memory.bytes_transferred_l2")
    def _bytes_transferred_l2(self, GL2C_HIT_sum, GL2C_MISS_sum):
        """
        Total bytes transferred through L2 cache

        Formula: (GL2C_HIT + GL2C_MISS) * 128 (L2 cache line size is 128 bytes)

        Counters:
        - GL2C_HIT_sum: L2 cache hits
        - GL2C_MISS_sum: L2 cache misses
        """
        total_accesses = GL2C_HIT_sum + GL2C_MISS_sum
        return total_accesses * 128  # 128-byte cache lines

    @metric("memory.bytes_transferred_l1")
    def _bytes_transferred_l1(self, TCP_REQ_sum):
        """
        Total bytes transferred through L1 (L0) cache

        Formula: TCP_REQ_sum * 64 (typical cache line size for TCP)

        Counters:
        - TCP_REQ_sum: Total TCP requests

        Note: RDNA3 TCP (L0 vector cache) typically uses 64-byte granularity
        """
        return TCP_REQ_sum * 64

    # Cache metrics

    @metric("memory.l2_hit_rate")
    def _l2_hit_rate(self, GL2C_HIT_sum, GL2C_MISS_sum):
        """
        L2 cache hit rate as percentage

        Formula: (hits / (hits + misses)) * 100

        Counters:
        - GL2C_HIT_sum: Number of cache hits (sum over GL2C instances)
        - GL2C_MISS_sum: Number of cache misses (sum over GL2C instances)
        """
        total = GL2C_HIT_sum + GL2C_MISS_sum
        return (GL2C_HIT_sum / total) * 100 if total > 0 else 0.0

    @metric("memory.l1_hit_rate")
    def _l1_hit_rate(self, TCP_REQ_sum, TCP_REQ_MISS_sum):
        """
        L1 (L0) cache hit rate as percentage

        Formula: ((total_accesses - l1_misses) / total_accesses) * 100

        Counters:
        - TCP_REQ_sum: Total TCP requests (sum over TCP instances)
        - TCP_REQ_MISS_sum: TCP request misses (sum over TCP instances)

        Note: On RDNA3, TCP is the L0 vector cache (equivalent to L1 in older architectures)
        """
        if TCP_REQ_sum == 0:
            return 0.0
        return ((TCP_REQ_sum - TCP_REQ_MISS_sum) / TCP_REQ_sum) * 100

    @metric("memory.l2_bandwidth")
    def _l2_bandwidth(self, GL2C_HIT_sum, GL2C_MISS_sum, GRBM_GUI_ACTIVE):
        """
        L2 cache bandwidth in GB/s

        Formula: (total_accesses * 128 bytes) / time

        Counters:
        - GL2C_HIT_sum: L2 cache hits
        - GL2C_MISS_sum: L2 cache misses
        - GRBM_GUI_ACTIVE: Number of active GPU cycles

        Note: L2 cacheline is 128 bytes
        """
        total_accesses = GL2C_HIT_sum + GL2C_MISS_sum
        bytes_transferred = total_accesses * 128

        if GRBM_GUI_ACTIVE == 0:
            return 0.0

        time_seconds = GRBM_GUI_ACTIVE / (self.device_specs.base_clock_mhz * 1e6)
        return (bytes_transferred / 1e9) / time_seconds if time_seconds > 0 else 0.0

    # Coalescing metrics

    @metric("memory.coalescing_efficiency")
    def _coalescing_efficiency(
        self, TA_TOTAL_WAVEFRONTS_sum, TCP_REQ_READ_sum, TCP_REQ_NON_READ_sum
    ):
        """
        Memory coalescing efficiency as percentage

        Formula: (TA_TOTAL_WAVEFRONTS / TCP_TOTAL_ACCESSES) * 100

        Counters:
        - TA_TOTAL_WAVEFRONTS_sum: Total vec32 packets processed by TA
        - TCP_REQ_READ_sum: Total TCP read requests
        - TCP_REQ_NON_READ_sum: Total TCP non-read requests

        Physical meaning:
        - 100% = perfectly coalesced (1 cache access per wavefront)
        - 3.125% = completely uncoalesced (32 cache accesses per wavefront)

        Note: RDNA3 uses wave32, so perfect coalescing = 1 access per wave
        """
        tcp_total_accesses = TCP_REQ_READ_sum + TCP_REQ_NON_READ_sum
        if tcp_total_accesses == 0:
            return 0.0
        return (TA_TOTAL_WAVEFRONTS_sum / tcp_total_accesses) * 100

    @metric(
        "memory.global_load_efficiency",
        unsupported_reason="Requires instruction-level counters not available in current YAML configs",
    )
    def _global_load_efficiency(self):
        """
        Global load efficiency - ratio of requested vs fetched memory

        Formula: (read_instructions * 64 bytes / read_requests * 64 bytes) * 100
        Simplifies to: (read_instructions / read_requests) * 100

        Note: Currently unsupported - requires SQ_INSTS_VMEM_RD counter
        """
        return 0.0

    @metric(
        "memory.global_store_efficiency",
        unsupported_reason="Requires instruction-level counters not available in current YAML configs",
    )
    def _global_store_efficiency(self):
        """
        Global store efficiency - ratio of requested vs written memory

        Formula: (write_instructions / write_requests) * 100

        Note: Currently unsupported - requires SQ_INSTS_VMEM_WR counter
        """
        return 0.0

    # LDS metrics

    @metric("memory.lds_bank_conflicts")
    def _lds_bank_conflicts(self, SQC_LDS_BANK_CONFLICT_sum, SQC_LDS_IDX_ACTIVE_sum):
        """
        LDS bank conflicts as percentage

        Formula: (SQC_LDS_BANK_CONFLICT / SQC_LDS_IDX_ACTIVE) * 100

        Counters:
        - SQC_LDS_BANK_CONFLICT_sum: Number of cycles LDS is stalled by bank conflicts
        - SQC_LDS_IDX_ACTIVE_sum: Number of cycles LDS is used for indexed operations

        Value range: 0% (optimal) to 100% (bad)
        Thresholds: <5% good, >20% bad
        """
        if SQC_LDS_IDX_ACTIVE_sum == 0:
            return 0.0
        return (SQC_LDS_BANK_CONFLICT_sum / SQC_LDS_IDX_ACTIVE_sum) * 100

    # Atomic metrics

    @metric("memory.atomic_latency")
    def _atomic_latency(self, SQ_WAIT_ANY_sum, SQ_WAVES_sum):
        """
        Wait cycles per wave (atomic contention indicator)

        Formula: (SQ_WAIT_ANY * 4) / SQ_WAVES

        Counters:
        - SQ_WAIT_ANY_sum: Number of wave-cycles spent waiting (in quad-cycles)
        - SQ_WAVES_sum: Total number of waves

        Thresholds:
        - <10K: LOW contention (good)
        - 10K-100K: MODERATE contention
        - 100K-1M: HIGH contention
        - >1M: EXTREME contention (bad)

        Note: SQ_WAIT_ANY is in quad-cycles (4 cycles), so multiply by 4
        """
        if SQ_WAVES_sum == 0:
            return 0.0
        return (SQ_WAIT_ANY_sum * 4) / SQ_WAVES_sum

    # Occupancy metrics

    @metric("compute.occupancy_percent")
    def _occupancy_percent(self, SQ_WAVE_CYCLES_sum, GRBM_GUI_ACTIVE):
        """
        GPU Occupancy as percentage of maximum

        Formula: (SQ_WAVE_CYCLES / GRBM_GUI_ACTIVE / CU_NUM / max_waves_per_cu) * 100

        Counters:
        - SQ_WAVE_CYCLES_sum: Cycles spent executing waves (in quad-cycles)
        - GRBM_GUI_ACTIVE: Total active GPU cycles

        Thresholds:
        - >50%: Good occupancy
        - 25-50%: Moderate occupancy
        - <25%: Poor occupancy (bad)

        Note: SQ_WAVE_CYCLES is in quad-cycles (4 cycles), so multiply by 4
        """
        if GRBM_GUI_ACTIVE == 0:
            return 0.0

        # Convert quad-cycles to actual cycles
        wave_cycles_actual = SQ_WAVE_CYCLES_sum * 4

        # Maximum possible wave-cycles = active_cycles * num_CUs * max_waves_per_CU
        max_wave_cycles = (
            GRBM_GUI_ACTIVE * self.device_specs.num_cu * self.device_specs.max_waves_per_cu
        )

        return (wave_cycles_actual / max_wave_cycles) * 100 if max_wave_cycles > 0 else 0.0

    # Compute metrics

    @metric("compute.total_flops")
    def _total_flops(self):
        """
        Total floating-point operations performed by the kernel

        Formula: 64 * (FP16 + FP32 + FP64) + 512 * MFMA
        """
        return 0.0

    @metric("compute.hbm_gflops")
    def _hbm_gflops(self):
        """
        Compute throughput (GFLOPS) normalized by kernel execution time

        Formula: (total_flops / 1e9) / time_seconds
        """
        return 0.0

    @metric("compute.hbm_arithmetic_intensity")
    def _hbm_arithmetic_intensity(self):
        """
        HBM Arithmetic Intensity: ratio of floating-point operations to HBM bytes transferred (FLOP/byte)

        Formula: total_flops / hbm_bytes
        """
        return 0.0

    @metric("compute.l2_arithmetic_intensity")
    def _l2_arithmetic_intensity(self):
        """
        L2 Arithmetic Intensity: ratio of floating-point operations to L2 cache bytes accessed (FLOP/byte)

        Formula: total_flops / l2_bytes
        """
        return 0.0

    @metric("compute.l1_arithmetic_intensity")
    def _l1_arithmetic_intensity(self):
        """
        L1 Arithmetic Intensity: ratio of floating-point operations to L1 cache bytes accessed (FLOP/byte)

        Formula: total_flops / l1_bytes
        """
        return 0.0
