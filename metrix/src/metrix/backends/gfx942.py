"""
GFX942 (MI300X) Backend

Each metric is defined with @metric decorator.
Counter names appear EXACTLY ONCE - as function parameters.
"""

from .base import CounterBackend, DeviceSpecs, ProfileResult
from ..utils.common import split_counters_into_passes
from .decorator import metric
from ..profiler.rocprof_wrapper import ROCProfV3Wrapper
from typing import List, Optional, Dict


class GFX942Backend(CounterBackend):
    """
    AMD MI300X (gfx942) counter backend

    All metrics are defined with @metric decorator.
    Hardware counter names appear ONLY as function parameter names.
    """

    def _get_device_specs(self) -> DeviceSpecs:
        """MI300X specifications"""
        return DeviceSpecs(
            arch="gfx942",
            name="AMD Instinct MI300X",
            num_cu=304,
            max_waves_per_cu=32,
            wavefront_size=64,
            base_clock_mhz=2100.0,
            hbm_bandwidth_gbs=5300.0,
            l2_bandwidth_gbs=11000.0,
            l2_size_mb=256.0,
            lds_size_per_cu_kb=64.0,
            fp32_tflops=163.4,
            fp64_tflops=81.7,
            int8_tops=1307.4,
            boost_clock_mhz=2100,
        )

    def _get_counter_groups(self, counters: List[str]) -> List[List[str]]:
        """
        Group counters into passes using MI300X-specific block limits.

        This keeps the hardware-specific knowledge (block limits and naming)
        in the gfx942 backend while reusing the generic helper from
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
        Return per-hardware-block counter limits for gfx942 (MI300X).

        These limits define how many performance counters can be simultaneously
        collected from each hardware block in a single profiling pass.

        Hardware blocks on MI300X:
        - SQ (Shader): Instruction counters (VALU, LDS, VMEM, etc.)
        - TA (Texture Addresser): Texture address operations
        - TD (Texture Data): Texture data fetch operations
        - TCP (Texture Cache per Pipe): L1 vector cache
        - TCC (Texture Cache Channel): L2 cache and memory controller
        - CPC (Command Processor - Compute): Compute command processing
        - CPF (Command Processor - Fetch): Command fetch operations
        - SPI (Shader Processor Input): Wavefront dispatch and scheduling
        - GRBM (Graphics Register Bus Manager): Global GPU activity
        - GDS (Global Data Share): Inter-workgroup communication

        Returns:
            Dict mapping block_name -> max_counters_per_pass
        """
        return {
            "SQ": 8,  # Shader - instruction counters
            "TA": 2,  # Texture Addresser
            "TD": 2,  # Texture Data
            "TCP": 4,  # L1 Cache (Texture Cache per Pipe)
            "TCC": 4,  # L2 Cache / Memory Controller
            "CPC": 2,  # Command Processor - Compute
            "CPF": 2,  # Command Processor - Fetch
            "SPI": 6,  # Shader Processor Input
            "GRBM": 2,  # Graphics Register Bus Manager
            "GDS": 4,  # Global Data Share
        }

    def _run_rocprof(
        self, command: str, counters: List[str], kernel_filter: Optional[str] = None, cwd: Optional[str] = None, timeout_seconds: Optional[int] = 0
    ) -> List[ProfileResult]:
        """Run rocprofv3 and return results (single pass only - base class handles multi-pass)"""
        wrapper = ROCProfV3Wrapper(timeout_seconds=timeout_seconds)
        return wrapper.profile(command, counters, kernel_filter=kernel_filter, cwd=cwd)

    # Memory bandwidth metrics

    @metric("memory.hbm_read_bandwidth")
    def _hbm_read_bandwidth(self, TCC_EA0_RDREQ_sum, TCC_EA0_RDREQ_32B_sum, TCC_BUBBLE_sum, GRBM_GUI_ACTIVE):
        """
        HBM read bandwidth in GB/s

        Formula: (128B_requests * 128 + 64B_requests * 64 + 32B_requests * 32) / (active_cycles / clock_freq)

        Note: TCC_EA0_RDREQ_sum aggregates across all memory controllers on MI300
              TCC_BUBBLE_sum counts 128B read requests
        """
        # Calculate bytes with 32B/64B/128B distinction
        bytes_read_128B = TCC_BUBBLE_sum * 128
        bytes_read_64B = (TCC_EA0_RDREQ_sum - TCC_BUBBLE_sum - TCC_EA0_RDREQ_32B_sum) * 64
        bytes_read_32B = TCC_EA0_RDREQ_32B_sum * 32
        bytes_read = bytes_read_128B + bytes_read_64B + bytes_read_32B

        if GRBM_GUI_ACTIVE == 0:
            return 0.0

        time_seconds = GRBM_GUI_ACTIVE / (self.device_specs.base_clock_mhz * 1e6)
        return (bytes_read / 1e9) / time_seconds if time_seconds > 0 else 0.0

    @metric("memory.hbm_write_bandwidth")
    def _hbm_write_bandwidth(self, TCC_EA0_WRREQ_sum, TCC_EA0_WRREQ_64B_sum, GRBM_GUI_ACTIVE):
        """
        HBM write bandwidth in GB/s (with 32B/64B request granularity)

        Formula: (64B_requests * 64 + 32B_requests * 32) / (active_cycles / clock_freq)

        Note: TCC_EA0_WRREQ_sum aggregates across all memory controllers on MI300
        """
        # Calculate bytes with 32B/64B distinction
        bytes_written_64B = TCC_EA0_WRREQ_64B_sum * 64
        bytes_written_32B = (TCC_EA0_WRREQ_sum - TCC_EA0_WRREQ_64B_sum) * 32
        bytes_written = bytes_written_64B + bytes_written_32B

        if GRBM_GUI_ACTIVE == 0:
            return 0.0

        time_seconds = GRBM_GUI_ACTIVE / (self.device_specs.base_clock_mhz * 1e6)
        return (bytes_written / 1e9) / time_seconds if time_seconds > 0 else 0.0

    @metric("memory.hbm_bandwidth_utilization")
    def _hbm_bandwidth_utilization(
        self,
        TCC_EA0_RDREQ_sum,
        TCC_EA0_RDREQ_32B_sum,
        TCC_BUBBLE_sum,
        TCC_EA0_WRREQ_sum,
        TCC_EA0_WRREQ_64B_sum,
        GRBM_GUI_ACTIVE,
    ):
        """
        HBM bandwidth utilization as percentage of peak

        Formula: (actual_bandwidth / peak_bandwidth) * 100

        Note: TCC_EA0_* counters aggregate across all memory controllers on MI300
              TCC_BUBBLE_sum counts 128B read requests
        """
        # Calculate bytes with 32B/64B/128B distinction
        bytes_read = (
            TCC_BUBBLE_sum * 128
            + (TCC_EA0_RDREQ_sum - TCC_BUBBLE_sum - TCC_EA0_RDREQ_32B_sum) * 64
            + TCC_EA0_RDREQ_32B_sum * 32
        )
        bytes_written = TCC_EA0_WRREQ_64B_sum * 64 + (TCC_EA0_WRREQ_sum - TCC_EA0_WRREQ_64B_sum) * 32
        total_bytes = bytes_read + bytes_written

        if GRBM_GUI_ACTIVE == 0:
            return 0.0

        time_seconds = GRBM_GUI_ACTIVE / (self.device_specs.base_clock_mhz * 1e6)
        actual_bw_gbs = (total_bytes / 1e9) / time_seconds if time_seconds > 0 else 0.0

        return (actual_bw_gbs / self.device_specs.hbm_bandwidth_gbs) * 100

    @metric("memory.bytes_transferred_hbm")
    def _bytes_transferred_hbm(
        self, TCC_EA0_RDREQ_sum, TCC_EA0_RDREQ_32B_sum, TCC_BUBBLE_sum, TCC_EA0_WRREQ_sum, TCC_EA0_WRREQ_64B_sum
    ):
        """
        Total bytes transferred through HBM

        Formula: (128B_read_requests * 128 + 64B_read_requests * 64 + 32B_read_requests * 32 +
                  64B_write_requests * 64 + 32B_write_requests * 32)

        Note: TCC_EA0_* counters aggregate across all memory controllers on MI300
              TCC_BUBBLE_sum counts 128B read requests
        """
        bytes_read = (
            TCC_BUBBLE_sum * 128
            + (TCC_EA0_RDREQ_sum - TCC_BUBBLE_sum - TCC_EA0_RDREQ_32B_sum) * 64
            + TCC_EA0_RDREQ_32B_sum * 32
        )
        bytes_written = TCC_EA0_WRREQ_64B_sum * 64 + (TCC_EA0_WRREQ_sum - TCC_EA0_WRREQ_64B_sum) * 32
        return bytes_read + bytes_written

    @metric("memory.bytes_transferred_l2")
    def _bytes_transferred_l2(self, TCC_REQ_sum):
        """
        Total bytes transferred through L2 cache

        Formula: TCC_REQ_sum * 128 (L2 cache line size is 128 bytes)
        """
        return TCC_REQ_sum * 128

    @metric("memory.bytes_transferred_l1")
    def _bytes_transferred_l1(self, TCP_TOTAL_CACHE_ACCESSES_sum):
        """
        Total bytes transferred through L1 cache

        Formula: TCP_TOTAL_CACHE_ACCESSES_sum * 128 (L1 cache line size is 128 bytes)
        """
        return TCP_TOTAL_CACHE_ACCESSES_sum * 128

    # Cache metrics

    @metric("memory.l2_hit_rate")
    def _l2_hit_rate(self, TCC_HIT_sum, TCC_MISS_sum):
        """
        L2 cache hit rate as percentage

        Formula: (hits / (hits + misses)) * 100
        """
        total = TCC_HIT_sum + TCC_MISS_sum
        return (TCC_HIT_sum / total) * 100 if total > 0 else 0.0

    @metric("memory.l1_hit_rate")
    def _l1_hit_rate(self, TCP_TCC_READ_REQ_sum, TCP_TOTAL_CACHE_ACCESSES_sum):
        """
        L1 cache hit rate as percentage

        Formula: ((total_accesses - l1_misses) / total_accesses) * 100
        L1 misses go to L2 (TCC), so misses = TCP_TCC_READ_REQ
        """
        if TCP_TOTAL_CACHE_ACCESSES_sum == 0:
            return 0.0

        l1_hits = TCP_TOTAL_CACHE_ACCESSES_sum - TCP_TCC_READ_REQ_sum
        return (l1_hits / TCP_TOTAL_CACHE_ACCESSES_sum) * 100

    @metric("memory.l2_bandwidth")
    def _l2_bandwidth(self, TCC_HIT_sum, TCC_MISS_sum, GRBM_GUI_ACTIVE):
        """
        L2 cache bandwidth in GB/s

        Formula: (total_accesses * 128 bytes) / time
        Note: L2 cacheline is 128 bytes
        """
        total_accesses = TCC_HIT_sum + TCC_MISS_sum
        bytes_accessed = total_accesses * 128  # L2 cacheline size

        if GRBM_GUI_ACTIVE == 0:
            return 0.0

        time_seconds = GRBM_GUI_ACTIVE / (self.device_specs.base_clock_mhz * 1e6)
        return (bytes_accessed / 1e9) / time_seconds if time_seconds > 0 else 0.0

    # Coalescing metrics

    @metric("memory.coalescing_efficiency")
    def _coalescing_efficiency(self, SQ_INSTS_VMEM_RD, SQ_INSTS_VMEM_WR, TCP_TOTAL_ACCESSES_sum):
        """
        Memory coalescing efficiency as percentage

        Formula: (total_memory_instructions * 16 / total_cache_accesses) * 100

        Physical meaning:
        - Perfect coalescing (stride=1): 100% (minimal cache accesses)
        - Poor coalescing (stride>1): 25% for float, 50% for double

        This represents actual bandwidth efficiency, not rescaled.
        """
        total_instructions = SQ_INSTS_VMEM_RD + SQ_INSTS_VMEM_WR

        if TCP_TOTAL_ACCESSES_sum == 0:
            return 0.0

        # 16 = 64 threads per wavefront / 4 threads per cacheline
        efficiency = (total_instructions * 16 / TCP_TOTAL_ACCESSES_sum) * 100

        # Cap at 100% (can happen due to prefetching)
        return min(efficiency, 100.0)

    @metric("memory.global_load_efficiency")
    def _global_load_efficiency(self, SQ_INSTS_VMEM_RD, TCP_TCC_READ_REQ_sum):
        """
        Global load efficiency - ratio of requested vs fetched memory

        Formula: (read_instructions * 64 bytes / read_requests * 64 bytes) * 100
        Simplifies to: (read_instructions / read_requests) * 100
        """
        if TCP_TCC_READ_REQ_sum == 0:
            return 0.0

        return min((SQ_INSTS_VMEM_RD / TCP_TCC_READ_REQ_sum) * 100, 100.0)

    @metric("memory.global_store_efficiency")
    def _global_store_efficiency(self, SQ_INSTS_VMEM_WR, TCP_TCC_WRITE_REQ_sum):
        """
        Global store efficiency - ratio of requested vs written memory

        Formula: (write_instructions / write_requests) * 100
        """
        if TCP_TCC_WRITE_REQ_sum == 0:
            return 0.0

        return min((SQ_INSTS_VMEM_WR / TCP_TCC_WRITE_REQ_sum) * 100, 100.0)

    # LDS metrics

    @metric("memory.lds_bank_conflicts")
    def _lds_bank_conflicts(self, SQ_LDS_BANK_CONFLICT, SQ_INSTS_LDS):
        """
        LDS bank conflicts per instruction

        Formula: total_conflicts / total_lds_instructions
        """
        if SQ_INSTS_LDS == 0:
            return 0.0

        return SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS

    # Atomic metrics

    @metric("memory.atomic_latency")
    def _atomic_latency(self, TCC_EA0_ATOMIC_LEVEL_sum, TCC_EA0_ATOMIC_sum):
        """
        Average atomic operation latency in cycles (L2 cache atomic latency)

        Formula: TCC_EA0_ATOMIC_LEVEL_sum / TCC_EA0_ATOMIC_sum (MI300/MI350 counters)

        Note: This measures atomic operations to/from L2 cache, not GDS operations.
        GDS (Global Data Share) is a special feature rarely used by most kernels.
        """
        if TCC_EA0_ATOMIC_sum == 0:
            return 0.0

        return TCC_EA0_ATOMIC_LEVEL_sum / TCC_EA0_ATOMIC_sum

    # Compute metrics

    @metric("compute.total_flops")
    def _total_flops(
        self,
        SQ_INSTS_VALU_ADD_F16,
        SQ_INSTS_VALU_MUL_F16,
        SQ_INSTS_VALU_TRANS_F16,
        SQ_INSTS_VALU_FMA_F16,
        SQ_INSTS_VALU_ADD_F32,
        SQ_INSTS_VALU_MUL_F32,
        SQ_INSTS_VALU_TRANS_F32,
        SQ_INSTS_VALU_FMA_F32,
        SQ_INSTS_VALU_ADD_F64,
        SQ_INSTS_VALU_MUL_F64,
        SQ_INSTS_VALU_TRANS_F64,
        SQ_INSTS_VALU_FMA_F64,
        SQ_INSTS_VALU_MFMA_MOPS_F16,
        SQ_INSTS_VALU_MFMA_MOPS_BF16,
        SQ_INSTS_VALU_MFMA_MOPS_F32,
        SQ_INSTS_VALU_MFMA_MOPS_F64,
    ):
        """
        Total floating-point operations performed by the kernel

        Formula: 64 * (FP16 + FP32 + FP64) + 512 * MFMA
        - 64 operations per wave (wavefront size = 64)
        - FMA counts as 2 operations (multiply + add)
        - MFMA instructions produce 512 operations per instruction
        """
        fops = 64 * (
            (SQ_INSTS_VALU_ADD_F16 + SQ_INSTS_VALU_MUL_F16 + SQ_INSTS_VALU_TRANS_F16 + SQ_INSTS_VALU_FMA_F16 * 2)
            + (SQ_INSTS_VALU_ADD_F32 + SQ_INSTS_VALU_MUL_F32 + SQ_INSTS_VALU_TRANS_F32 + SQ_INSTS_VALU_FMA_F32 * 2)
            + (SQ_INSTS_VALU_ADD_F64 + SQ_INSTS_VALU_MUL_F64 + SQ_INSTS_VALU_TRANS_F64 + SQ_INSTS_VALU_FMA_F64 * 2)
        ) + 512 * (
            SQ_INSTS_VALU_MFMA_MOPS_F16
            + SQ_INSTS_VALU_MFMA_MOPS_BF16
            + SQ_INSTS_VALU_MFMA_MOPS_F32
            + SQ_INSTS_VALU_MFMA_MOPS_F64
        )

        return fops

    @metric("compute.hbm_gflops")
    def _hbm_gflops(
        self,
        SQ_INSTS_VALU_ADD_F16,
        SQ_INSTS_VALU_MUL_F16,
        SQ_INSTS_VALU_TRANS_F16,
        SQ_INSTS_VALU_FMA_F16,
        SQ_INSTS_VALU_ADD_F32,
        SQ_INSTS_VALU_MUL_F32,
        SQ_INSTS_VALU_TRANS_F32,
        SQ_INSTS_VALU_FMA_F32,
        SQ_INSTS_VALU_ADD_F64,
        SQ_INSTS_VALU_MUL_F64,
        SQ_INSTS_VALU_TRANS_F64,
        SQ_INSTS_VALU_FMA_F64,
        SQ_INSTS_VALU_MFMA_MOPS_F16,
        SQ_INSTS_VALU_MFMA_MOPS_BF16,
        SQ_INSTS_VALU_MFMA_MOPS_F32,
        SQ_INSTS_VALU_MFMA_MOPS_F64,
        GRBM_GUI_ACTIVE,
    ):
        """
        Compute throughput (GFLOPS) normalized by kernel execution time

        Formula: (total_flops / 1e9) / time_seconds
        """
        # Calculate total FLOPS (same as compute.total_flops)
        fops = 64 * (
            (SQ_INSTS_VALU_ADD_F16 + SQ_INSTS_VALU_MUL_F16 + SQ_INSTS_VALU_TRANS_F16 + SQ_INSTS_VALU_FMA_F16 * 2)
            + (SQ_INSTS_VALU_ADD_F32 + SQ_INSTS_VALU_MUL_F32 + SQ_INSTS_VALU_TRANS_F32 + SQ_INSTS_VALU_FMA_F32 * 2)
            + (SQ_INSTS_VALU_ADD_F64 + SQ_INSTS_VALU_MUL_F64 + SQ_INSTS_VALU_TRANS_F64 + SQ_INSTS_VALU_FMA_F64 * 2)
        ) + 512 * (
            SQ_INSTS_VALU_MFMA_MOPS_F16
            + SQ_INSTS_VALU_MFMA_MOPS_BF16
            + SQ_INSTS_VALU_MFMA_MOPS_F32
            + SQ_INSTS_VALU_MFMA_MOPS_F64
        )

        if GRBM_GUI_ACTIVE == 0:
            return 0.0

        time_seconds = GRBM_GUI_ACTIVE / (self.device_specs.base_clock_mhz * 1e6)
        gflops = (fops / 1e9) / time_seconds if time_seconds > 0 else 0.0

        return gflops

    @metric("compute.hbm_arithmetic_intensity")
    def _hbm_arithmetic_intensity(
        self,
        SQ_INSTS_VALU_ADD_F16,
        SQ_INSTS_VALU_MUL_F16,
        SQ_INSTS_VALU_TRANS_F16,
        SQ_INSTS_VALU_FMA_F16,
        SQ_INSTS_VALU_ADD_F32,
        SQ_INSTS_VALU_MUL_F32,
        SQ_INSTS_VALU_TRANS_F32,
        SQ_INSTS_VALU_FMA_F32,
        SQ_INSTS_VALU_ADD_F64,
        SQ_INSTS_VALU_MUL_F64,
        SQ_INSTS_VALU_TRANS_F64,
        SQ_INSTS_VALU_FMA_F64,
        SQ_INSTS_VALU_MFMA_MOPS_F16,
        SQ_INSTS_VALU_MFMA_MOPS_BF16,
        SQ_INSTS_VALU_MFMA_MOPS_F32,
        SQ_INSTS_VALU_MFMA_MOPS_F64,
        TCC_EA0_RDREQ_sum,
        TCC_EA0_RDREQ_32B_sum,
        TCC_BUBBLE_sum,
        TCC_EA0_WRREQ_sum,
        TCC_EA0_WRREQ_64B_sum,
    ):
        """
        HBM Arithmetic Intensity: ratio of floating-point operations to HBM bytes transferred (FLOP/byte)

        Formula: total_flops / hbm_bytes
        """
        # Calculate total FLOPS
        fops = 64 * (
            (SQ_INSTS_VALU_ADD_F16 + SQ_INSTS_VALU_MUL_F16 + SQ_INSTS_VALU_TRANS_F16 + SQ_INSTS_VALU_FMA_F16 * 2)
            + (SQ_INSTS_VALU_ADD_F32 + SQ_INSTS_VALU_MUL_F32 + SQ_INSTS_VALU_TRANS_F32 + SQ_INSTS_VALU_FMA_F32 * 2)
            + (SQ_INSTS_VALU_ADD_F64 + SQ_INSTS_VALU_MUL_F64 + SQ_INSTS_VALU_TRANS_F64 + SQ_INSTS_VALU_FMA_F64 * 2)
        ) + 512 * (
            SQ_INSTS_VALU_MFMA_MOPS_F16
            + SQ_INSTS_VALU_MFMA_MOPS_BF16
            + SQ_INSTS_VALU_MFMA_MOPS_F32
            + SQ_INSTS_VALU_MFMA_MOPS_F64
        )

        # Calculate HBM bytes (with 32B/64B/128B distinction)
        hbm_rd = (
            TCC_BUBBLE_sum * 128
            + (TCC_EA0_RDREQ_sum - TCC_BUBBLE_sum - TCC_EA0_RDREQ_32B_sum) * 64
            + TCC_EA0_RDREQ_32B_sum * 32
        )
        hbm_wr = TCC_EA0_WRREQ_64B_sum * 64 + (TCC_EA0_WRREQ_sum - TCC_EA0_WRREQ_64B_sum) * 32
        hbm_bytes = hbm_rd + hbm_wr

        # Arithmetic intensity = FLOP / byte
        ai_hbm = fops / hbm_bytes if hbm_bytes > 0 else 0.0

        return ai_hbm

    @metric("compute.l2_arithmetic_intensity")
    def _l2_arithmetic_intensity(
        self,
        SQ_INSTS_VALU_ADD_F16,
        SQ_INSTS_VALU_MUL_F16,
        SQ_INSTS_VALU_TRANS_F16,
        SQ_INSTS_VALU_FMA_F16,
        SQ_INSTS_VALU_ADD_F32,
        SQ_INSTS_VALU_MUL_F32,
        SQ_INSTS_VALU_TRANS_F32,
        SQ_INSTS_VALU_FMA_F32,
        SQ_INSTS_VALU_ADD_F64,
        SQ_INSTS_VALU_MUL_F64,
        SQ_INSTS_VALU_TRANS_F64,
        SQ_INSTS_VALU_FMA_F64,
        SQ_INSTS_VALU_MFMA_MOPS_F16,
        SQ_INSTS_VALU_MFMA_MOPS_BF16,
        SQ_INSTS_VALU_MFMA_MOPS_F32,
        SQ_INSTS_VALU_MFMA_MOPS_F64,
        TCC_REQ_sum,
    ):
        """
        L2 Arithmetic Intensity: ratio of floating-point operations to L2 cache bytes accessed (FLOP/byte)

        Formula: total_flops / l2_bytes
        """
        # Calculate total FLOPS
        fops = 64 * (
            (SQ_INSTS_VALU_ADD_F16 + SQ_INSTS_VALU_MUL_F16 + SQ_INSTS_VALU_TRANS_F16 + SQ_INSTS_VALU_FMA_F16 * 2)
            + (SQ_INSTS_VALU_ADD_F32 + SQ_INSTS_VALU_MUL_F32 + SQ_INSTS_VALU_TRANS_F32 + SQ_INSTS_VALU_FMA_F32 * 2)
            + (SQ_INSTS_VALU_ADD_F64 + SQ_INSTS_VALU_MUL_F64 + SQ_INSTS_VALU_TRANS_F64 + SQ_INSTS_VALU_FMA_F64 * 2)
        ) + 512 * (
            SQ_INSTS_VALU_MFMA_MOPS_F16
            + SQ_INSTS_VALU_MFMA_MOPS_BF16
            + SQ_INSTS_VALU_MFMA_MOPS_F32
            + SQ_INSTS_VALU_MFMA_MOPS_F64
        )

        # Calculate L2 bytes (L2 cache line is 128 bytes)
        l2_bytes = TCC_REQ_sum * 128

        # Arithmetic intensity = FLOP / byte
        ai_l2 = fops / l2_bytes if l2_bytes > 0 else 0.0

        return ai_l2

    @metric("compute.l1_arithmetic_intensity")
    def _l1_arithmetic_intensity(
        self,
        SQ_INSTS_VALU_ADD_F16,
        SQ_INSTS_VALU_MUL_F16,
        SQ_INSTS_VALU_TRANS_F16,
        SQ_INSTS_VALU_FMA_F16,
        SQ_INSTS_VALU_ADD_F32,
        SQ_INSTS_VALU_MUL_F32,
        SQ_INSTS_VALU_TRANS_F32,
        SQ_INSTS_VALU_FMA_F32,
        SQ_INSTS_VALU_ADD_F64,
        SQ_INSTS_VALU_MUL_F64,
        SQ_INSTS_VALU_TRANS_F64,
        SQ_INSTS_VALU_FMA_F64,
        SQ_INSTS_VALU_MFMA_MOPS_F16,
        SQ_INSTS_VALU_MFMA_MOPS_BF16,
        SQ_INSTS_VALU_MFMA_MOPS_F32,
        SQ_INSTS_VALU_MFMA_MOPS_F64,
        TCP_TOTAL_CACHE_ACCESSES_sum,
    ):
        """
        L1 Arithmetic Intensity: ratio of floating-point operations to L1 cache bytes accessed (FLOP/byte)

        Formula: total_flops / l1_bytes
        """
        # Calculate total FLOPS
        fops = 64 * (
            (SQ_INSTS_VALU_ADD_F16 + SQ_INSTS_VALU_MUL_F16 + SQ_INSTS_VALU_TRANS_F16 + SQ_INSTS_VALU_FMA_F16 * 2)
            + (SQ_INSTS_VALU_ADD_F32 + SQ_INSTS_VALU_MUL_F32 + SQ_INSTS_VALU_TRANS_F32 + SQ_INSTS_VALU_FMA_F32 * 2)
            + (SQ_INSTS_VALU_ADD_F64 + SQ_INSTS_VALU_MUL_F64 + SQ_INSTS_VALU_TRANS_F64 + SQ_INSTS_VALU_FMA_F64 * 2)
        ) + 512 * (
            SQ_INSTS_VALU_MFMA_MOPS_F16
            + SQ_INSTS_VALU_MFMA_MOPS_BF16
            + SQ_INSTS_VALU_MFMA_MOPS_F32
            + SQ_INSTS_VALU_MFMA_MOPS_F64
        )

        # Calculate L1 bytes (L1 cache line is 128 bytes on gfx942)
        l1_bytes = TCP_TOTAL_CACHE_ACCESSES_sum * 128

        # Arithmetic intensity = FLOP / byte
        ai_l1 = fops / l1_bytes if l1_bytes > 0 else 0.0

        return ai_l1
