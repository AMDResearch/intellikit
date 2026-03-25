"""
GFX942 (MI300X) Backend

Metrics are loaded from counter_defs.yaml.
This file provides architecture-specific infrastructure only.
Device specs are queried from hipGetDeviceProperties at runtime.
"""

from .base import CounterBackend, DeviceSpecs, ProfileResult
from .device_info import query_device_specs
from ..utils.common import split_counters_into_passes
from ..profiler.rocprof_wrapper import ROCProfV3Wrapper
from typing import List, Optional, Dict


class GFX942Backend(CounterBackend):
    """
    AMD MI300X (gfx942) counter backend

    Metric definitions live in counter_defs.yaml.
    """

    def _get_device_specs(self) -> DeviceSpecs:
        return query_device_specs("gfx942")

    def _get_counter_groups(self, counters: List[str]) -> List[List[str]]:
        """
        Group counters into passes using MI300X-specific block limits.

        This keeps the hardware-specific knowledge (block limits and naming)
        in the gfx942 backend while reusing the generic helper from
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
        Return per-hardware-block counter limits for gfx942 (MI300X).

        These limits define how many performance counters can be simultaneously
        collected from each hardware block in a single profiling pass.
        """
        return {
            "SQ": 8,
            "TA": 2,
            "TD": 2,
            "TCP": 4,
            "TCC": 4,
            "CPC": 2,
            "CPF": 2,
            "SPI": 6,
            "GRBM": 2,
            "GDS": 4,
        }

    def _run_rocprof(
        self,
        command: str,
        counters: List[str],
        kernel_filter: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout_seconds: Optional[int] = 0,
    ) -> List[ProfileResult]:
        """Run rocprofv3 and return results (single pass only - base class handles multi-pass)"""
        wrapper = ROCProfV3Wrapper(timeout_seconds=timeout_seconds)
        return wrapper.profile(command, counters, kernel_filter=kernel_filter, cwd=cwd)
