"""
GFX1151 (RDNA4) Backend

Shares all hardware configuration with gfx1201 (RDNA4).
"""

from .base import DeviceSpecs
from .gfx1201 import GFX1201Backend


class GFX1151Backend(GFX1201Backend):
    """AMD RDNA4 (GFX1151) backend - same hardware config as gfx1201."""

    def _get_device_specs(self) -> DeviceSpecs:
        return DeviceSpecs(
            arch="gfx1151",
            name="AMD Radeon Graphics (RDNA4)",
            wavefront_size=32,
        )
