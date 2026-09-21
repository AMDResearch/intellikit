"""
GFX1150 (RDNA 3.5) Backend

gfx1150 (Strix Point) is GFX11 hardware -- rocminfo reports it as compatible
with the amdgcn-amd-amdhsa--gfx11-generic ISA -- so it shares aqlprofile's
GFX11 counter-block layout with gfx1100, not GFX12's with gfx1201.

It exposes the same counter set as its Strix Halo sibling gfx1151, minus the
ALUStalledByLDS and LdsLatency derived counters (see counter_defs.yaml).
"""

from .device_info import query_device_specs
from .gfx1100 import GFX1100Backend


class GFX1150Backend(GFX1100Backend):
    """AMD RDNA 3.5 (GFX1150) backend - same hardware config as gfx1100."""

    def _get_device_specs(self):
        return query_device_specs("gfx1150")
