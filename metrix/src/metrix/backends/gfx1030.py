"""
GFX1030 (RDNA2) Backend

Counter-limits-only backend for AMD RDNA2 (RX 6000 series).
Inherits metric computation from GFX1201Backend (YAML-based metrics via counter_defs.yaml).
Only overrides device specs and counter block limits.
"""

from .device_info import query_device_specs
from .gfx1201 import GFX1201Backend
from typing import Dict


class GFX1030Backend(GFX1201Backend):
    """AMD RDNA2 (gfx1030) backend."""

    def _get_device_specs(self):
        return query_device_specs("gfx1030")

    def _get_counter_block_limits(self) -> Dict[str, int]:
        """
        Return per-hardware-block counter limits for GFX10 (RDNA2).

        Source: ROCm/rocm-systems aqlprofile gfxip/gfx10/gfx10_block_info.h
                (NumCounters constants, derived from AMD internal chip_offset_byte.h)
                https://github.com/ROCm/rocm-systems/blob/121fafc82c961b9a9f32eda63d2422bf7a0be817/projects/aqlprofile/gfxip/gfx10/gfx10_block_info.h
        """
        return {
            "SQ": 8,     # Shader - instruction counters
            "SPI": 6,    # Shader Processor Input
            "TA": 2,     # Texture Addresser
            "TD": 2,     # Texture Data
            "TCP": 4,    # L1 Cache (Texture Cache per Pipe)
            "TCC": 4,    # L2 Cache / Memory Controller
            "TCA": 4,    # L2 Cache Arbiter
            "CPC": 2,    # Command Processor - Compute
            "CPF": 2,    # Command Processor - Fetch
            "CPG": 2,    # Command Processor - Graphics
            "GRBM": 2,   # Graphics Register Bus Manager
            "GDS": 4,    # Global Data Share
            "GL1A": 4,   # Global L1 Arbiter
            "GL1C": 4,   # Global L1 Cache
            "GL2A": 4,   # Global L2 Arbiter
            "GL2C": 4,   # Global L2 Cache
            "GCR": 2,    # Global Cache Router
            "GUS": 2,    # Graphics Utility Shader
            "CB": 4,     # Color Buffer
            "DB": 4,     # Depth Buffer
            "RMI": 4,    # Resource Management Interface
            "RPB": 4,    # Read Path Buffer
            "SDMA": 2,   # System DMA
        }
