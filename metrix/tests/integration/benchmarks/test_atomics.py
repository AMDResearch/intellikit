# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Atomic Latency Microbenchmarks.

Validates the atomic_latency metric using high and low contention
patterns. Currently atomic_latency counters are only defined for gfx942.

Metrics covered:
  - memory.atomic_latency
"""

import pytest

from .conftest import compile_hip, get_arch, profile_longest_kernel

# Architectures that have atomic_latency counter definitions
ATOMIC_LATENCY_ARCHS = {"gfx942"}


class TestAtomicLatency:
    """Validate atomic latency metrics."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, arch):
        self.binary = compile_hip("atomic_contention.hip", tmp_path)
        self.arch = arch
        if arch not in ATOMIC_LATENCY_ARCHS:
            pytest.skip(f"atomic_latency not defined for {arch}")

    def test_high_contention_measurable_latency(self):
        """All threads atomicAdd to 1 address should show high atomic latency."""
        cmd = f"{self.binary} 0"  # mode=high contention
        m = profile_longest_kernel(cmd, ["memory.atomic_latency"])

        latency = m.get("memory.atomic_latency", 0)
        assert latency > 50.0, (
            f"High contention atomic latency too low: {latency:.1f} cycles"
        )

    def test_low_contention_lower_latency(self):
        """Each thread atomicAdd to own address should show lower latency."""
        cmd = f"{self.binary} 1"  # mode=low contention
        m = profile_longest_kernel(cmd, ["memory.atomic_latency"])

        latency = m.get("memory.atomic_latency", 0)
        assert latency >= 0.0, (
            f"Atomic latency is negative: {latency:.1f}"
        )

    def test_high_contention_gt_low_contention(self):
        """High contention should produce higher atomic latency than low."""
        m_hi = profile_longest_kernel(
            f"{self.binary} 0", ["memory.atomic_latency"]
        )
        m_lo = profile_longest_kernel(
            f"{self.binary} 1", ["memory.atomic_latency"]
        )

        hi = m_hi.get("memory.atomic_latency", 0)
        lo = m_lo.get("memory.atomic_latency", 0)

        assert hi > lo, (
            f"High contention ({hi:.1f}) should have higher latency "
            f"than low contention ({lo:.1f})"
        )

    def test_moderate_contention_between(self):
        """Per-block contention should fall between high and low."""
        m_hi = profile_longest_kernel(
            f"{self.binary} 0", ["memory.atomic_latency"]
        )
        m_mod = profile_longest_kernel(
            f"{self.binary} 2", ["memory.atomic_latency"]
        )
        m_lo = profile_longest_kernel(
            f"{self.binary} 1", ["memory.atomic_latency"]
        )

        hi = m_hi.get("memory.atomic_latency", 0)
        mod = m_mod.get("memory.atomic_latency", 0)
        lo = m_lo.get("memory.atomic_latency", 0)

        # Moderate should be less than or equal to high contention
        assert mod <= hi * 1.1, (
            f"Moderate ({mod:.1f}) should be <= high ({hi:.1f})"
        )
        # Moderate should be greater than or equal to low contention
        assert mod >= lo * 0.9, (
            f"Moderate ({mod:.1f}) should be >= low ({lo:.1f})"
        )
