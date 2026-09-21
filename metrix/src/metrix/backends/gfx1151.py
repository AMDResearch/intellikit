"""
GFX1151 (RDNA 3.5) Backend

gfx1151 (Strix Halo) is GFX11 hardware -- rocminfo reports it as compatible
with the amdgcn-amd-amdhsa--gfx11-generic ISA -- so it shares aqlprofile's
GFX11 counter-block layout with gfx1100, not GFX12's with gfx1201.
"""

from .device_info import query_device_specs
from .gfx1100 import GFX1100Backend


class GFX1151Backend(GFX1100Backend):
    """AMD RDNA 3.5 (GFX1151) backend - same hardware config as gfx1100."""

    def _get_device_specs(self):
        return query_device_specs("gfx1151")
