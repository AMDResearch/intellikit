"""
GFX1100 (RDNA3) backend: GFX11Backend with the gfx1100 device specs.
counter_defs.yaml defines only the gfx1100 metrics validated on a Radeon Pro W7900,
and their counters need the profile_standard perf level.
"""

from .device_info import query_device_specs
from .rdna_common import GFX11Backend


class GFX1100Backend(GFX11Backend):
    """AMD RDNA3 (gfx1100) backend."""

    def _get_device_specs(self):
        return query_device_specs("gfx1100")
