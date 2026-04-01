"""
GFX1201 (RDNA4) Backend

Metrics are loaded from counter_defs.yaml.
This file provides architecture-specific infrastructure only.
Device specs are queried from hipGetDeviceProperties at runtime.
"""

from .base import CounterBackend, ProfileResult
from .device_info import query_device_specs
from ..utils.common import split_counters_into_passes
from ..profiler.rocprof_wrapper import ROCProfV3Wrapper
from pathlib import Path
from typing import List, Dict, Optional


class GFX1201Backend(CounterBackend):
    """AMD RDNA4 (gfx1201) backend."""

    def _get_device_specs(self):
        return query_device_specs("gfx1201")

    def _get_counter_groups(self, counters: List[str]) -> List[List[str]]:
        """
        Group counters into passes using GFX12-specific block limits.

        Reuses the generic bin-packing helper from `common.py` with
        GFX12 block limits and counter-block mapping.
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
        Return per-hardware-block counter limits for GFX12 (RDNA4).

        These limits define how many performance counters can be simultaneously
        collected from each hardware block in a single profiling pass.

        Source: ROCm/rocm-systems aqlprofile gfxip/gfx12/gfx12_block_info.h
                (NumCounters constants, derived from AMD internal chip_offset_byte.h)
                https://github.com/ROCm/rocm-systems/blob/121fafc82c961b9a9f32eda63d2422bf7a0be817/projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h
        """
        return {
            "SQG": 8,  # Shader (Graphics) - instruction counters
            "SQC": 16,  # Shader (Compute) - instruction counters
            "SPI": 6,  # Shader Processor Input
            "TA": 2,  # Texture Addresser
            "TD": 2,  # Texture Data
            "TCP": 4,  # L1 Cache (Texture Cache per Pipe)
            "GL1A": 4,  # Global L1 Arbiter
            "GL1C": 4,  # Global L1 Cache
            "GL2A": 4,  # Global L2 Arbiter
            "GL2C": 4,  # Global L2 Cache
            "CPC": 2,  # Command Processor - Compute
            "CPF": 2,  # Command Processor - Fetch
            "CPG": 2,  # Command Processor - Graphics
            "GRBM": 2,  # Graphics Register Bus Manager
            "GRBMH": 2,  # GRBM (per-SE)
            "GCR": 2,  # Global Cache Router
            "RPB": 4,  # Read Path Buffer
            "SDMA": 2,  # System DMA
            "RLC": 2,  # Run List Controller
            "CHA": 4,  # Coherency Hub Agent
            "CHC": 4,  # Coherency Hub Client
            "UTCL1": 4,  # Unified Translation Cache L1
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
        wrapper = ROCProfV3Wrapper(timeout_seconds=timeout_seconds)
        extra_counters_path = Path(__file__).parent / "counter_defs.yaml"

        return wrapper.profile(
            command=command,
            counters=counters,
            kernel_filter=kernel_filter,
            cwd=cwd,
            kernel_iteration_range=kernel_iteration_range,
            extra_counters_path=extra_counters_path if extra_counters_path.exists() else None,
            arch=self.device_specs.arch,
        )
