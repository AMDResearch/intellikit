# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
LDS Bank Conflict Microbenchmarks.

Validates LDS metrics using sequential (conflict-free) and strided
(conflict-heavy) LDS access patterns.

Metrics covered:
  - memory.lds_utilization
  - memory.lds_bank_conflicts
"""

import pytest

from .conftest import compile_hip, get_arch, profile_longest_kernel


class TestLDSBankConflicts:
    """Validate LDS bank conflict metrics."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, arch):
        self.binary = compile_hip("lds_conflicts.hip", tmp_path)
        self.arch = arch

    def test_no_conflicts_sequential(self):
        """Sequential LDS access should show low bank conflict count."""
        cmd = f"{self.binary} 0"  # mode=no conflict
        m = profile_longest_kernel(cmd, ["memory.lds_bank_conflicts"])

        conflicts = m.get("memory.lds_bank_conflicts", 0)
        assert conflicts < 5.0, (
            f"Sequential LDS access shows too many bank conflicts: {conflicts:.1f}"
        )

    def test_high_conflicts_stride32(self):
        """Stride-32 LDS access should show elevated bank conflict count."""
        cmd = f"{self.binary} 1"  # mode=max conflict
        m = profile_longest_kernel(cmd, ["memory.lds_bank_conflicts"])

        conflicts = m.get("memory.lds_bank_conflicts", 0)
        assert conflicts > 2.0, (
            f"Stride-32 LDS should show bank conflicts: {conflicts:.1f}"
        )

    def test_conflicts_increase_with_stride(self):
        """Bank conflicts should increase: sequential < stride-2 < stride-32."""
        m_seq = profile_longest_kernel(
            f"{self.binary} 0", ["memory.lds_bank_conflicts"]
        )
        m_mod = profile_longest_kernel(
            f"{self.binary} 2", ["memory.lds_bank_conflicts"]
        )
        m_max = profile_longest_kernel(
            f"{self.binary} 1", ["memory.lds_bank_conflicts"]
        )

        seq = m_seq.get("memory.lds_bank_conflicts", 0)
        mod = m_mod.get("memory.lds_bank_conflicts", 0)
        mx = m_max.get("memory.lds_bank_conflicts", 0)

        assert mx >= mod, (
            f"Stride-32 ({mx:.1f}) should have >= conflicts than stride-2 ({mod:.1f})"
        )
        assert mod >= seq, (
            f"Stride-2 ({mod:.1f}) should have >= conflicts than sequential ({seq:.1f})"
        )

    def test_lds_utilization_nonzero(self):
        """Any LDS-using kernel should show non-zero LDS utilization."""
        cmd = f"{self.binary} 0"
        m = profile_longest_kernel(cmd, ["memory.lds_utilization"])

        util = m.get("memory.lds_utilization", 0)
        assert util > 0, "LDS utilization is zero for a kernel using shared memory"

    def test_lds_utilization_bounded(self):
        """LDS utilization should be in valid range [0, 100]."""
        cmd = f"{self.binary} 1"  # max conflict → high utilization
        m = profile_longest_kernel(cmd, ["memory.lds_utilization"])

        util = m.get("memory.lds_utilization", 0)
        assert 0.0 <= util <= 100.0, f"LDS utilization out of range: {util:.1f}%"
