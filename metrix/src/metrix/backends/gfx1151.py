"""
GFX1151 (RDNA4) Backend

Shares all hardware configuration with gfx1201 (RDNA4).
"""

from .device_info import query_device_specs
from .gfx1201 import GFX1201Backend


class GFX1151Backend(GFX1201Backend):
    """AMD RDNA4 (GFX1151) backend - same hardware config as gfx1201."""

    def _get_device_specs(self):
        return query_device_specs("gfx1151")
