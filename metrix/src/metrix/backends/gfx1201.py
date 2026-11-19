"""
GFX1201 (RDNA4) Backend

Metrics are loaded from counter_defs.yaml.
This file provides architecture-specific infrastructure only.
Device specs are queried from hipGetDeviceProperties at runtime.
Counter grouping and the rocprofv3 invocation are shared with the other RDNA
backends through RDNABackend.
"""

from .device_info import query_device_specs
from .rdna_common import RDNABackend
from typing import Dict


class GFX1201Backend(RDNABackend):
    """AMD RDNA4 (gfx1201) backend."""

    def _get_device_specs(self):
        return query_device_specs("gfx1201")

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
