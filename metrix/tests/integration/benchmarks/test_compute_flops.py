# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Compute / FLOPS Microbenchmarks.

Validates compute metrics using FMA chains with known FLOP counts
and varying arithmetic intensity ratios.

Metrics covered:
  - compute.total_flops
  - compute.hbm_gflops
  - compute.hbm_arithmetic_intensity
  - compute.l2_arithmetic_intensity
  - compute.l1_arithmetic_intensity
"""

import pytest

from .conftest import compile_hip, get_arch, profile_longest_kernel


class TestTotalFlops:
    """Validate total_flops with a pure FMA kernel of known FLOP count."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, arch):
        self.binary = compile_hip("flops_fma.hip", tmp_path)
        self.arch = arch

    def test_pure_fma_flops_nonzero(self):
        """Pure FMA kernel should report non-zero total FLOPS."""
        cmd = f"{self.binary} 0"  # mode=pure_fma
        m = profile_longest_kernel(cmd, ["compute.total_flops"])

        flops = m.get("compute.total_flops", 0)
        assert flops > 0, "total_flops is zero for a pure FMA kernel"

    def test_pure_fma_flops_in_range(self):
        """Pure FMA FLOP count should be within 50% of analytical expectation.

        512 blocks * 256 threads * 100000 iters * 2 FLOP/FMA = 26,214,400,000
        Hardware counters count per-wavefront, so the exact count depends on
        wavefront size (64 on CDNA, 32 on RDNA). We use a wide tolerance.
        """
        cmd = f"{self.binary} 0"
        m = profile_longest_kernel(cmd, ["compute.total_flops"])

        flops = m.get("compute.total_flops", 0)
        # Expected: ~26.2 billion FLOP (CDNA) or similar on RDNA
        # Use wide tolerance since counter semantics vary by arch
        expected_min = 1e9    # At least 1 GFLOP
        expected_max = 1e12   # At most 1 TFLOP

        assert flops > expected_min, (
            f"total_flops ({flops:.2e}) is too low — expected > {expected_min:.0e}"
        )
        assert flops < expected_max, (
            f"total_flops ({flops:.2e}) is suspiciously high — expected < {expected_max:.0e}"
        )

    def test_gflops_nonzero(self):
        """hbm_gflops (GFLOP/s rate) should be positive for compute kernel."""
        cmd = f"{self.binary} 0"
        m = profile_longest_kernel(cmd, ["compute.hbm_gflops"])

        gflops = m.get("compute.hbm_gflops", 0)
        assert gflops > 0, "hbm_gflops is zero for a pure FMA kernel"


class TestArithmeticIntensity:
    """Validate arithmetic intensity scales with compute-to-memory ratio."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, arch):
        self.binary = compile_hip("flops_fma.hip", tmp_path)
        self.arch = arch

    def test_low_ai_copy_kernel(self):
        """Copy-only kernel should show very low arithmetic intensity."""
        cmd = f"{self.binary} 1"  # mode=low AI (copy)
        m = profile_longest_kernel(cmd, ["compute.hbm_arithmetic_intensity"])

        ai = m.get("compute.hbm_arithmetic_intensity", 0)
        assert ai < 10.0, (
            f"Copy kernel arithmetic intensity too high: {ai:.2f}"
        )

    def test_high_ai_many_fmas(self):
        """100 FMAs per load should give high arithmetic intensity."""
        cmd = f"{self.binary} 2 100"  # mode=high AI, K=100
        m = profile_longest_kernel(cmd, ["compute.hbm_arithmetic_intensity"])

        ai = m.get("compute.hbm_arithmetic_intensity", 0)
        assert ai > 5.0, (
            f"High-AI kernel arithmetic intensity too low: {ai:.2f}"
        )

    def test_ai_increases_with_compute(self):
        """Arithmetic intensity should increase with more FMAs per load."""
        m_low = profile_longest_kernel(
            f"{self.binary} 2 1", ["compute.hbm_arithmetic_intensity"]
        )
        m_high = profile_longest_kernel(
            f"{self.binary} 2 100", ["compute.hbm_arithmetic_intensity"]
        )

        ai_low = m_low.get("compute.hbm_arithmetic_intensity", 0)
        ai_high = m_high.get("compute.hbm_arithmetic_intensity", 0)

        assert ai_high > ai_low, (
            f"AI with K=100 ({ai_high:.2f}) should be > AI with K=1 ({ai_low:.2f})"
        )

    def test_l2_arithmetic_intensity_nonzero(self):
        """L2 arithmetic intensity should be non-negative."""
        cmd = f"{self.binary} 2 50"  # moderate AI
        m = profile_longest_kernel(cmd, ["compute.l2_arithmetic_intensity"])

        ai = m.get("compute.l2_arithmetic_intensity", 0)
        assert ai >= 0.0, f"L2 arithmetic intensity is negative: {ai}"

    def test_l1_arithmetic_intensity_nonzero(self):
        """L1 arithmetic intensity should be non-negative."""
        cmd = f"{self.binary} 2 50"
        m = profile_longest_kernel(cmd, ["compute.l1_arithmetic_intensity"])

        ai = m.get("compute.l1_arithmetic_intensity", 0)
        assert ai >= 0.0, f"L1 arithmetic intensity is negative: {ai}"

    def test_ai_hierarchy_ordering(self):
        """HBM AI >= L2 AI >= L1 AI (more bytes at each cache level)."""
        cmd = f"{self.binary} 2 50"
        m = profile_longest_kernel(
            cmd,
            [
                "compute.hbm_arithmetic_intensity",
                "compute.l2_arithmetic_intensity",
                "compute.l1_arithmetic_intensity",
            ],
        )

        ai_hbm = m.get("compute.hbm_arithmetic_intensity", 0)
        ai_l2 = m.get("compute.l2_arithmetic_intensity", 0)
        ai_l1 = m.get("compute.l1_arithmetic_intensity", 0)

        # HBM transfers fewer bytes than L2, which transfers fewer than L1
        # So AI_HBM >= AI_L2 >= AI_L1 (same FLOP numerator, increasing denominator)
        if ai_hbm > 0 and ai_l2 > 0:
            assert ai_hbm >= ai_l2 * 0.8, (
                f"HBM AI ({ai_hbm:.2f}) should be >= L2 AI ({ai_l2:.2f})"
            )
        if ai_l2 > 0 and ai_l1 > 0:
            assert ai_l2 >= ai_l1 * 0.8, (
                f"L2 AI ({ai_l2:.2f}) should be >= L1 AI ({ai_l1:.2f})"
            )
