# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Coalescing Efficiency Microbenchmarks.

Validates memory access pattern metrics using coalesced, strided, and
scattered access patterns.

Metrics covered:
  - memory.coalescing_efficiency
  - memory.global_load_efficiency
  - memory.global_store_efficiency
"""

import pytest

from .conftest import compile_hip, get_arch, profile_longest_kernel

COALESCING_METRICS = [
    "memory.coalescing_efficiency",
    "memory.global_load_efficiency",
    "memory.global_store_efficiency",
]


class TestCoalescingEfficiency:
    """Validate coalescing metrics with different access patterns."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, arch):
        self.binary = compile_hip("coalescing.hip", tmp_path)
        self.arch = arch

    def test_coalesced_high_efficiency(self):
        """Stride-1 (coalesced) access should show high coalescing efficiency."""
        cmd = f"{self.binary} 0"  # mode=coalesced
        m = profile_longest_kernel(cmd, ["memory.coalescing_efficiency"])

        eff = m.get("memory.coalescing_efficiency", 0)
        assert eff >= 70.0, (
            f"Coalesced access efficiency too low: {eff:.1f}%"
        )
        assert eff <= 100.0

    def test_strided_reduced_efficiency(self):
        """Stride-32 access should show reduced coalescing efficiency."""
        cmd = f"{self.binary} 1 32"  # mode=strided, stride=32
        m = profile_longest_kernel(cmd, ["memory.coalescing_efficiency"])

        eff = m.get("memory.coalescing_efficiency", 0)
        assert eff < 50.0, (
            f"Stride-32 coalescing efficiency too high: {eff:.1f}%"
        )

    def test_scattered_lowest_efficiency(self):
        """Scattered (random) access should show very low coalescing efficiency."""
        cmd = f"{self.binary} 2"  # mode=scattered
        m = profile_longest_kernel(cmd, ["memory.coalescing_efficiency"])

        eff = m.get("memory.coalescing_efficiency", 0)
        assert eff < 40.0, (
            f"Scattered coalescing efficiency too high: {eff:.1f}%"
        )

    def test_efficiency_decreases_with_stride(self):
        """Coalescing efficiency should decrease: coalesced > strided > scattered."""
        m_coal = profile_longest_kernel(
            f"{self.binary} 0", ["memory.coalescing_efficiency"]
        )
        m_stride = profile_longest_kernel(
            f"{self.binary} 1 32", ["memory.coalescing_efficiency"]
        )
        m_scatter = profile_longest_kernel(
            f"{self.binary} 2", ["memory.coalescing_efficiency"]
        )

        coal = m_coal.get("memory.coalescing_efficiency", 0)
        stride = m_stride.get("memory.coalescing_efficiency", 0)
        scatter = m_scatter.get("memory.coalescing_efficiency", 0)

        assert coal > stride, (
            f"Coalesced ({coal:.1f}%) should be > strided ({stride:.1f}%)"
        )
        assert stride >= scatter, (
            f"Strided ({stride:.1f}%) should be >= scattered ({scatter:.1f}%)"
        )

    # -- Global load/store efficiency --

    def test_global_load_efficiency_coalesced(self):
        """Coalesced access should show high global load efficiency."""
        cmd = f"{self.binary} 0"
        m = profile_longest_kernel(cmd, ["memory.global_load_efficiency"])

        eff = m.get("memory.global_load_efficiency", 0)
        assert eff >= 50.0, (
            f"Coalesced global load efficiency too low: {eff:.1f}%"
        )

    def test_global_store_efficiency_coalesced(self):
        """Coalesced access should show high global store efficiency."""
        cmd = f"{self.binary} 0"
        m = profile_longest_kernel(cmd, ["memory.global_store_efficiency"])

        eff = m.get("memory.global_store_efficiency", 0)
        assert eff >= 50.0, (
            f"Coalesced global store efficiency too low: {eff:.1f}%"
        )

    @pytest.mark.parametrize("stride", [2, 4, 8, 16])
    def test_load_efficiency_decreases_with_stride(self, stride):
        """Global load efficiency should be lower with larger strides."""
        m_coal = profile_longest_kernel(
            f"{self.binary} 0", ["memory.global_load_efficiency"]
        )
        m_stride = profile_longest_kernel(
            f"{self.binary} 1 {stride}", ["memory.global_load_efficiency"]
        )

        coal_eff = m_coal.get("memory.global_load_efficiency", 0)
        stride_eff = m_stride.get("memory.global_load_efficiency", 0)

        assert coal_eff >= stride_eff * 0.9, (
            f"Coalesced ({coal_eff:.1f}%) should be >= stride-{stride} ({stride_eff:.1f}%)"
        )
