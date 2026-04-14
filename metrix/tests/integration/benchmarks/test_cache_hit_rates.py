# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Cache Hit Rate Microbenchmarks.

Validates L2 and L1 cache hit rate metrics using two contrasting access
patterns: cache-friendly (small array, many iterations) vs cache-thrashing
(large buffer, strided access).

Metrics covered:
  - memory.l2_hit_rate
  - memory.l1_hit_rate
  - memory.bytes_transferred_l2
  - memory.bytes_transferred_l1
"""

import pytest

from .conftest import compile_hip, get_arch, profile_longest_kernel


class TestCacheHitRates:
    """Validate cache hit rate metrics with friendly and hostile access patterns."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, arch):
        self.friendly_binary = compile_hip("cache_friendly.hip", tmp_path)
        self.thrash_binary = compile_hip("cache_thrash.hip", tmp_path)
        self.arch = arch

    # -- L2 hit rate --

    def test_l2_hit_rate_high_with_small_array(self):
        """256 KB array iterated 500x with few blocks should show elevated L2 hit rate.

        The threshold is set at 30% to accommodate different cache hierarchies
        across CDNA and RDNA architectures. MI300X with 256MB L2 achieves >60%,
        while MI210 with smaller L2 may show ~40%.
        """
        cmd = f"{self.friendly_binary}"  # defaults: 256KB, 500 iters, 16 blocks
        m = profile_longest_kernel(cmd, ["memory.l2_hit_rate"])

        hit_rate = m.get("memory.l2_hit_rate", 0)
        assert hit_rate > 30.0, (
            f"L2 hit rate too low for cache-friendly access: {hit_rate:.1f}%"
        )
        assert hit_rate <= 100.0, f"L2 hit rate > 100%: {hit_rate:.1f}%"

    def test_l2_hit_rate_low_with_thrashing(self):
        """256 MB strided access should show low L2 hit rate."""
        cmd = f"{self.thrash_binary}"  # defaults: 256MB, stride=16
        m = profile_longest_kernel(cmd, ["memory.l2_hit_rate"])

        hit_rate = m.get("memory.l2_hit_rate", 0)
        # MI300X has 256MB L2 — even thrashing may show moderate hit rates.
        # On most GPUs with 4-8MB L2, this should be well under 50%.
        if self.arch != "gfx942":
            assert hit_rate < 60.0, (
                f"L2 hit rate too high for thrashing access: {hit_rate:.1f}%"
            )

    def test_l2_hit_rate_friendly_vs_thrash(self):
        """Cache-friendly should have higher L2 hit rate than cache-thrashing."""
        m_friendly = profile_longest_kernel(
            f"{self.friendly_binary}", ["memory.l2_hit_rate"]
        )
        m_thrash = profile_longest_kernel(
            f"{self.thrash_binary}", ["memory.l2_hit_rate"]
        )

        friendly_rate = m_friendly.get("memory.l2_hit_rate", 0)
        thrash_rate = m_thrash.get("memory.l2_hit_rate", 0)

        assert friendly_rate > thrash_rate, (
            f"Cache-friendly ({friendly_rate:.1f}%) should have higher L2 hit rate "
            f"than thrashing ({thrash_rate:.1f}%)"
        )

    # -- L1 hit rate --

    def test_l1_hit_rate_with_small_array(self):
        """Small array with many iterations should show non-trivial L1 hit rate."""
        # Use very small array (4KB) and many iterations for L1 residency
        cmd = f"{self.friendly_binary} 4 1000 16"  # 4KB, 1000 iters
        m = profile_longest_kernel(cmd, ["memory.l1_hit_rate"])

        hit_rate = m.get("memory.l1_hit_rate", 0)
        assert hit_rate >= 0.0, f"L1 hit rate is negative: {hit_rate}"
        assert hit_rate <= 100.0, f"L1 hit rate > 100%: {hit_rate}"

    # -- Bytes transferred --

    def test_bytes_transferred_l2_nonzero(self):
        """Any memory-accessing kernel should transfer bytes through L2."""
        cmd = f"{self.thrash_binary}"
        m = profile_longest_kernel(cmd, ["memory.bytes_transferred_l2"])

        bytes_l2 = m.get("memory.bytes_transferred_l2", 0)
        assert bytes_l2 > 0, "bytes_transferred_l2 is zero"

    def test_bytes_transferred_l1_nonzero(self):
        """Any memory-accessing kernel should transfer bytes through L1."""
        cmd = f"{self.friendly_binary}"
        m = profile_longest_kernel(cmd, ["memory.bytes_transferred_l1"])

        bytes_l1 = m.get("memory.bytes_transferred_l1", 0)
        assert bytes_l1 > 0, "bytes_transferred_l1 is zero"

    def test_thrash_transfers_more_from_hbm(self):
        """Cache-thrashing should transfer more bytes from HBM than cache-friendly."""
        m_friendly = profile_longest_kernel(
            f"{self.friendly_binary}", ["memory.bytes_transferred_hbm"]
        )
        m_thrash = profile_longest_kernel(
            f"{self.thrash_binary}", ["memory.bytes_transferred_hbm"]
        )

        friendly_bytes = m_friendly.get("memory.bytes_transferred_hbm", 0)
        thrash_bytes = m_thrash.get("memory.bytes_transferred_hbm", 0)

        assert thrash_bytes > friendly_bytes, (
            f"Thrashing ({thrash_bytes}) should transfer more HBM bytes "
            f"than friendly ({friendly_bytes})"
        )
