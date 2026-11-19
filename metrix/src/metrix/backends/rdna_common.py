"""
Shared RDNA backend infrastructure

RDNABackend holds what every RDNA backend (gfx10, gfx11, gfx12) does the same
way: block-aware counter grouping and the rocprofv3 invocation that passes
counter_defs.yaml as extra counter definitions. GFX11Backend adds the gfx11
per-block counter limits shared by the gfx11 parts.

Metrics themselves come from counter_defs.yaml, gated on DeviceSpecs.arch;
nothing here makes one RDNA generation's metrics apply to another.
"""

from .base import CounterBackend, ProfileResult
from ..utils.common import split_counters_into_passes
from ..profiler.rocprof_wrapper import ROCProfV3Wrapper
from pathlib import Path
from typing import List, Dict, Optional


class RDNABackend(CounterBackend):
    """Common base for RDNA backends; subclasses supply device specs and block limits."""

    def _get_counter_groups(self, counters: List[str]) -> List[List[str]]:
        """
        Group counters into passes using the backend's block limits.

        Reuses the generic bin-packing helper from `common.py` with the
        backend's block limits and counter-block mapping.
        """
        from ..logger import logger

        block_limits = self._get_counter_block_limits()
        return split_counters_into_passes(
            counters,
            block_limits=block_limits,
            get_counter_block=self._get_counter_block,
            logger=logger,
        )

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


class GFX11Backend(RDNABackend):
    """Common base for gfx11 (RDNA3 and RDNA 3.5) backends."""

    def _get_counter_block_limits(self) -> Dict[str, int]:
        """
        Return per-hardware-block counter limits for GFX11 (RDNA3).

        Source: ROCm/rocm-systems aqlprofile gfxip/gfx11/gfx11_block_info.h
                (NumCounters constants, derived from AMD internal chip_offset_byte.h)
                https://github.com/ROCm/rocm-systems/blob/121fafc82c961b9a9f32eda63d2422bf7a0be817/projects/aqlprofile/gfxip/gfx11/gfx11_block_info.h
        """
        return {
            "SQ": 8,  # Shader - instruction counters
            "SPI": 6,  # Shader Processor Input
            "TA": 2,  # Texture Addresser
            "TD": 2,  # Texture Data
            "TCP": 4,  # L1 Cache (Texture Cache per Pipe)
            "TCC": 4,  # L2 Cache / Memory Controller
            "TCA": 4,  # L2 Cache Arbiter
            "CPC": 2,  # Command Processor - Compute
            "CPF": 2,  # Command Processor - Fetch
            "CPG": 2,  # Command Processor - Graphics
            "GRBM": 2,  # Graphics Register Bus Manager
            "GDS": 4,  # Global Data Share
            "GL1A": 4,  # Global L1 Arbiter
            "GL1C": 4,  # Global L1 Cache
            "GL2A": 4,  # Global L2 Arbiter
            "GL2C": 4,  # Global L2 Cache
            "GCR": 2,  # Global Cache Router
            "GUS": 2,  # Graphics Utility Shader
            "CB": 4,  # Color Buffer
            "DB": 4,  # Depth Buffer
            "RMI": 4,  # Resource Management Interface
            "RPB": 4,  # Read Path Buffer
            "SDMA": 2,  # System DMA
        }
