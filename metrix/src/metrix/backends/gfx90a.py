"""
GFX90a (MI200) Backend

Metrics are loaded from counter_defs.yaml.
This file provides architecture-specific infrastructure only.
Device specs are queried from hipGetDeviceProperties at runtime.
"""

from .base import CounterBackend, DeviceSpecs, ProfileResult
from .device_info import query_device_specs
from ..utils.common import split_counters_into_passes
from ..profiler.rocprof_wrapper import ROCProfV3Wrapper
from typing import List, Optional, Dict


class GFX90aBackend(CounterBackend):
    """
    AMD MI200 (gfx90a) counter backend

    Metric definitions live in counter_defs.yaml.
    """

    def _get_device_specs(self) -> DeviceSpecs:
        return query_device_specs("gfx90a")

    def _get_counter_groups(self, counters: List[str]) -> List[List[str]]:
        """
        Group counters into passes using MI200-specific block limits.

        This keeps the hardware-specific knowledge (block limits and naming)
        in the gfx90a backend while reusing the generic helper from
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
        Return per-hardware-block counter limits for gfx90a (MI200).

        These limits define how many performance counters can be simultaneously
        collected from each hardware block in a single profiling pass.
        """
        return {
            "SQ": 8,  # Shader Sequencer — instruction issue & scheduling
            "TA": 2,  # Texture Addresser — coalesces memory requests
            "TD": 2,  # Texture Data — routes cache data back to SIMDs
            "TCP": 4,  # Texture Cache per Pipe — L1 vector cache
            "TCC": 4,  # Texture Cache Channel — L2 cache / memory controller
            "CPC": 2,  # Command Processor Compute — decodes dispatches
            "CPF": 2,  # Command Processor Fetch — fetches commands from memory
            "SPI": 6,  # Shader Processor Input — workgroup manager, schedules waves to CUs
            "GRBM": 2,  # Graphics Register Bus Manager — top-level GPU activity counters
            "GDS": 4,  # Global Data Share — chip-wide shared memory
        }

    def _run_rocprof(
        self,
        command: str,
        counters: List[str],
        kernel_filter: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout_seconds: Optional[int] = 0,
        kernel_iteration_range: Optional[str] = None,
    ) -> List[ProfileResult]:
        """Run rocprofv3 and return results (single pass only - base class handles multi-pass)"""
        wrapper = ROCProfV3Wrapper(timeout_seconds=timeout_seconds)
        return wrapper.profile(
            command,
            counters,
            kernel_filter=kernel_filter,
            cwd=cwd,
            kernel_iteration_range=kernel_iteration_range,
        )
    ):
        """
        HBM bandwidth utilization as percentage of peak

        Formula: (actual_bandwidth / peak_bandwidth) * 100
        """
        # Calculate bytes with 32B/64B distinction
        bytes_read = (TCC_EA_RDREQ_sum - TCC_EA_RDREQ_32B_sum) * 64 + TCC_EA_RDREQ_32B_sum * 32
        bytes_written = TCC_EA_WRREQ_64B_sum * 64 + (TCC_EA_WRREQ_sum - TCC_EA_WRREQ_64B_sum) * 32
        total_bytes = bytes_read + bytes_written

        if GRBM_GUI_ACTIVE == 0:
            return 0.0

        time_seconds = GRBM_GUI_ACTIVE / (self.device_specs.base_clock_mhz * 1e6)
        actual_bw_gbs = (total_bytes / 1e9) / time_seconds if time_seconds > 0 else 0.0

        return (actual_bw_gbs / self.device_specs.hbm_bandwidth_gbs) * 100

    @metric("memory.bytes_transferred_hbm")
    def _bytes_transferred_hbm(
        self, TCC_EA_RDREQ_sum, TCC_EA_RDREQ_32B_sum, TCC_EA_WRREQ_sum, TCC_EA_WRREQ_64B_sum
    ):
        """
        Total bytes transferred through HBM

        Formula: (64B_read_requests * 64 + 32B_read_requests * 32 +
                  64B_write_requests * 64 + 32B_write_requests * 32)
        """
        bytes_read = (TCC_EA_RDREQ_sum - TCC_EA_RDREQ_32B_sum) * 64 + TCC_EA_RDREQ_32B_sum * 32
        bytes_written = TCC_EA_WRREQ_64B_sum * 64 + (TCC_EA_WRREQ_sum - TCC_EA_WRREQ_64B_sum) * 32
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

        Formula: TCP_TOTAL_CACHE_ACCESSES_sum * 64 (L1 cache line size is 64 bytes)
        """
        return TCP_TOTAL_CACHE_ACCESSES_sum * 64

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

    @metric(
        "memory.atomic_latency",
        unsupported_reason="TCC_EA_ATOMIC_LEVEL_sum counter is broken on MI200 (gfx90a). "
        "This metric only works correctly on MI300X (gfx942) and newer GPUs.",
    )
    def _atomic_latency(self, TCC_EA_ATOMIC_LEVEL_sum, TCC_EA_ATOMIC_sum):
        """
        Average atomic operation latency in cycles (L2 cache atomic latency)

        Formula: TCC_EA_ATOMIC_LEVEL_sum / TCC_EA_ATOMIC_sum (MI200 counters)

        Note: This measures atomic operations to/from L2 cache, not GDS operations.
        GDS (Global Data Share) is a special feature rarely used by most kernels.
        """
        if TCC_EA_ATOMIC_sum == 0:
            return 0.0

        return TCC_EA_ATOMIC_LEVEL_sum / TCC_EA_ATOMIC_sum

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
            (
                SQ_INSTS_VALU_ADD_F16
                + SQ_INSTS_VALU_MUL_F16
                + SQ_INSTS_VALU_TRANS_F16
                + SQ_INSTS_VALU_FMA_F16 * 2
            )
            + (
                SQ_INSTS_VALU_ADD_F32
                + SQ_INSTS_VALU_MUL_F32
                + SQ_INSTS_VALU_TRANS_F32
                + SQ_INSTS_VALU_FMA_F32 * 2
            )
            + (
                SQ_INSTS_VALU_ADD_F64
                + SQ_INSTS_VALU_MUL_F64
                + SQ_INSTS_VALU_TRANS_F64
                + SQ_INSTS_VALU_FMA_F64 * 2
            )
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
    ):
        """
        Compute throughput (GFLOPS) using profiler kernel duration.

        Formula: (total_flops / 1e9) / (duration_us / 1e6)
        Duration is set by the base class from profiler timestamps before calling.
        """
        # Calculate total FLOPS (same as compute.total_flops)
        fops = 64 * (
            (
                SQ_INSTS_VALU_ADD_F16
                + SQ_INSTS_VALU_MUL_F16
                + SQ_INSTS_VALU_TRANS_F16
                + SQ_INSTS_VALU_FMA_F16 * 2
            )
            + (
                SQ_INSTS_VALU_ADD_F32
                + SQ_INSTS_VALU_MUL_F32
                + SQ_INSTS_VALU_TRANS_F32
                + SQ_INSTS_VALU_FMA_F32 * 2
            )
            + (
                SQ_INSTS_VALU_ADD_F64
                + SQ_INSTS_VALU_MUL_F64
                + SQ_INSTS_VALU_TRANS_F64
                + SQ_INSTS_VALU_FMA_F64 * 2
            )
        ) + 512 * (
            SQ_INSTS_VALU_MFMA_MOPS_F16
            + SQ_INSTS_VALU_MFMA_MOPS_BF16
            + SQ_INSTS_VALU_MFMA_MOPS_F32
            + SQ_INSTS_VALU_MFMA_MOPS_F64
        )

        duration_us = getattr(self, "_current_duration_us", 0.0)
        if duration_us <= 0:
            return 0.0

        time_seconds = duration_us / 1e6
        gflops = (fops / 1e9) / time_seconds

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
        TCC_EA_RDREQ_sum,
        TCC_EA_RDREQ_32B_sum,
        TCC_EA_WRREQ_sum,
        TCC_EA_WRREQ_64B_sum,
    ):
        """
        HBM Arithmetic Intensity: ratio of floating-point operations to HBM bytes transferred (FLOP/byte)

        Formula: total_flops / hbm_bytes
        """
        # Calculate total FLOPS
        fops = 64 * (
            (
                SQ_INSTS_VALU_ADD_F16
                + SQ_INSTS_VALU_MUL_F16
                + SQ_INSTS_VALU_TRANS_F16
                + SQ_INSTS_VALU_FMA_F16 * 2
            )
            + (
                SQ_INSTS_VALU_ADD_F32
                + SQ_INSTS_VALU_MUL_F32
                + SQ_INSTS_VALU_TRANS_F32
                + SQ_INSTS_VALU_FMA_F32 * 2
            )
            + (
                SQ_INSTS_VALU_ADD_F64
                + SQ_INSTS_VALU_MUL_F64
                + SQ_INSTS_VALU_TRANS_F64
                + SQ_INSTS_VALU_FMA_F64 * 2
            )
        ) + 512 * (
            SQ_INSTS_VALU_MFMA_MOPS_F16
            + SQ_INSTS_VALU_MFMA_MOPS_BF16
            + SQ_INSTS_VALU_MFMA_MOPS_F32
            + SQ_INSTS_VALU_MFMA_MOPS_F64
        )

        # Calculate HBM bytes (with 32B/64B/128B distinction)
        hbm_rd = (TCC_EA_RDREQ_sum - TCC_EA_RDREQ_32B_sum) * 64 + TCC_EA_RDREQ_32B_sum * 32
        hbm_wr = TCC_EA_WRREQ_64B_sum * 64 + (TCC_EA_WRREQ_sum - TCC_EA_WRREQ_64B_sum) * 32
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
            (
                SQ_INSTS_VALU_ADD_F16
                + SQ_INSTS_VALU_MUL_F16
                + SQ_INSTS_VALU_TRANS_F16
                + SQ_INSTS_VALU_FMA_F16 * 2
            )
            + (
                SQ_INSTS_VALU_ADD_F32
                + SQ_INSTS_VALU_MUL_F32
                + SQ_INSTS_VALU_TRANS_F32
                + SQ_INSTS_VALU_FMA_F32 * 2
            )
            + (
                SQ_INSTS_VALU_ADD_F64
                + SQ_INSTS_VALU_MUL_F64
                + SQ_INSTS_VALU_TRANS_F64
                + SQ_INSTS_VALU_FMA_F64 * 2
            )
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
            (
                SQ_INSTS_VALU_ADD_F16
                + SQ_INSTS_VALU_MUL_F16
                + SQ_INSTS_VALU_TRANS_F16
                + SQ_INSTS_VALU_FMA_F16 * 2
            )
            + (
                SQ_INSTS_VALU_ADD_F32
                + SQ_INSTS_VALU_MUL_F32
                + SQ_INSTS_VALU_TRANS_F32
                + SQ_INSTS_VALU_FMA_F32 * 2
            )
            + (
                SQ_INSTS_VALU_ADD_F64
                + SQ_INSTS_VALU_MUL_F64
                + SQ_INSTS_VALU_TRANS_F64
                + SQ_INSTS_VALU_FMA_F64 * 2
            )
        ) + 512 * (
            SQ_INSTS_VALU_MFMA_MOPS_F16
            + SQ_INSTS_VALU_MFMA_MOPS_BF16
            + SQ_INSTS_VALU_MFMA_MOPS_F32
            + SQ_INSTS_VALU_MFMA_MOPS_F64
        )

        # Calculate L1 bytes (L1 cache line is 64 bytes on gfx90a)
        l1_bytes = TCP_TOTAL_CACHE_ACCESSES_sum * 64

        # Arithmetic intensity = FLOP / byte
        ai_l1 = fops / l1_bytes if l1_bytes > 0 else 0.0

        return ai_l1
