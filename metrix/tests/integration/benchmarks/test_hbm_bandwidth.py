# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
HBM Bandwidth Microbenchmarks.

Validates metrix bandwidth metrics using persistent grid-stride kernels
with large (256MB) buffers that bypass L2 cache entirely.

Metrics covered:
  - memory.hbm_read_bandwidth
  - memory.hbm_write_bandwidth
  - memory.hbm_bandwidth_utilization
  - memory.bytes_transferred_hbm
  - memory.l2_bandwidth
"""

import pytest

from .conftest import compile_hip, get_arch, profile_longest_kernel

# ---------------------------------------------------------------------------
# Expected achievable bandwidth ranges by architecture (GB/s)
#
# These are NOT peak theoretical — they are conservative ranges that a
# well-written copy kernel should fall within.  The ranges are wide to
# accommodate different GPU SKUs within the same arch family.
# ---------------------------------------------------------------------------

EXPECTED_BW_RANGES = {
    # CDNA2: MI210 ~1.6 TB/s peak, MI250X ~3.2 TB/s peak
    "gfx90a":  {"min_gb_s": 500,  "max_gb_s": 3200},
    # CDNA3: MI300X/A ~5.2 TB/s peak, MI325X ~6.4 TB/s peak
    "gfx942":  {"min_gb_s": 2000, "max_gb_s": 6400},
    # CDNA4: MI350X/355X ~8 TB/s peak
    "gfx950":  {"min_gb_s": 3000, "max_gb_s": 8000},
    # RDNA2: RX 6000 series
    "gfx1030": {"min_gb_s": 100,  "max_gb_s": 600},
    # RDNA3: RX 7000 series
    "gfx1100": {"min_gb_s": 200,  "max_gb_s": 900},
    # RDNA3.5: Strix/Halo iGPUs (shared memory, low bandwidth)
    "gfx1150": {"min_gb_s": 30,   "max_gb_s": 200},
    "gfx1151": {"min_gb_s": 30,   "max_gb_s": 200},
    # RDNA4: Navi4x discrete
    "gfx1201": {"min_gb_s": 200,  "max_gb_s": 1000},
}

# Minimum absolute bandwidth (GB/s) any GPU should achieve on a copy kernel.
# If we're below this, something is fundamentally wrong.
ABSOLUTE_MIN_BW = 10.0

# Dtype argument mapping for the HIP binary
DTYPES = {
    "float4":  "0",
    "double2": "1",
    "float":   "2",
    "double":  "3",
}


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestHBMBandwidth:
    """Validate HBM bandwidth metrics with persistent grid-stride kernels."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, arch):
        self.binary = compile_hip("hbm_bandwidth.hip", tmp_path)
        self.arch = arch
        self.bw_range = EXPECTED_BW_RANGES.get(
            arch, {"min_gb_s": ABSOLUTE_MIN_BW, "max_gb_s": 10000}
        )

    # -- Copy bandwidth (read + write) --

    @pytest.mark.parametrize("dtype", ["float4", "double2", "float", "double"])
    def test_copy_bandwidth_nonzero(self, dtype):
        """Copy kernel should achieve measurable read AND write bandwidth."""
        cmd = f"{self.binary} 0 {DTYPES[dtype]}"
        m = profile_longest_kernel(
            cmd,
            ["memory.hbm_read_bandwidth", "memory.hbm_write_bandwidth",
             "memory.bytes_transferred_hbm"],
        )

        read_bw = m.get("memory.hbm_read_bandwidth", 0)
        write_bw = m.get("memory.hbm_write_bandwidth", 0)

        assert read_bw > ABSOLUTE_MIN_BW, (
            f"[{dtype}] HBM read bandwidth too low: {read_bw:.2f} GB/s"
        )
        assert write_bw > ABSOLUTE_MIN_BW, (
            f"[{dtype}] HBM write bandwidth too low: {write_bw:.2f} GB/s"
        )

        # Bytes transferred should be at least the buffer size (256MB)
        bytes_xfer = m.get("memory.bytes_transferred_hbm", 0)
        assert bytes_xfer >= 200 * 1024 * 1024, (
            f"[{dtype}] bytes_transferred_hbm suspiciously low: {bytes_xfer}"
        )

    def test_copy_bandwidth_in_expected_range(self):
        """float4 copy should fall within arch-specific bandwidth range."""
        cmd = f"{self.binary} 0 0"  # copy, float4
        m = profile_longest_kernel(
            cmd,
            ["memory.hbm_read_bandwidth", "memory.hbm_write_bandwidth"],
        )

        # Total effective bandwidth = read + write
        total_bw = m.get("memory.hbm_read_bandwidth", 0) + m.get("memory.hbm_write_bandwidth", 0)
        assert total_bw >= self.bw_range["min_gb_s"] * 0.5, (
            f"Total copy BW ({total_bw:.1f} GB/s) below 50% of expected min "
            f"({self.bw_range['min_gb_s']} GB/s) for {self.arch}"
        )

    def test_vectorized_faster_than_scalar(self):
        """float4 (16B/load) should achieve >= scalar float (4B/load) bandwidth."""
        cmd_f4 = f"{self.binary} 0 0"    # float4
        cmd_f1 = f"{self.binary} 0 2"    # float

        m_f4 = profile_longest_kernel(cmd_f4, ["memory.hbm_read_bandwidth"])
        m_f1 = profile_longest_kernel(cmd_f1, ["memory.hbm_read_bandwidth"])

        bw_f4 = m_f4.get("memory.hbm_read_bandwidth", 0)
        bw_f1 = m_f1.get("memory.hbm_read_bandwidth", 0)

        # float4 should be at least as fast (usually faster) than scalar
        assert bw_f4 >= bw_f1 * 0.9, (
            f"float4 ({bw_f4:.1f} GB/s) unexpectedly slower than float ({bw_f1:.1f} GB/s)"
        )

    # -- Read-only bandwidth --

    def test_read_only_bandwidth(self):
        """Read-only kernel should show high read BW, negligible write BW."""
        cmd = f"{self.binary} 1 0"  # read, float4
        m = profile_longest_kernel(
            cmd,
            ["memory.hbm_read_bandwidth", "memory.hbm_write_bandwidth"],
        )

        read_bw = m.get("memory.hbm_read_bandwidth", 0)
        write_bw = m.get("memory.hbm_write_bandwidth", 0)

        assert read_bw > ABSOLUTE_MIN_BW, (
            f"Read-only kernel read BW too low: {read_bw:.2f} GB/s"
        )

        # Write BW should be negligible (only 1 dummy write per launch)
        if write_bw > 0 and read_bw > 0:
            ratio = read_bw / max(write_bw, 0.001)
            assert ratio > 5, (
                f"Read-only kernel has too much write traffic: "
                f"read={read_bw:.1f} GB/s, write={write_bw:.1f} GB/s (ratio={ratio:.1f})"
            )

    # -- Write-only bandwidth --

    def test_write_only_bandwidth(self):
        """Write-only kernel should show measurable write BW."""
        cmd = f"{self.binary} 2 0"  # write, float4
        m = profile_longest_kernel(
            cmd,
            ["memory.hbm_write_bandwidth", "memory.hbm_read_bandwidth"],
        )

        write_bw = m.get("memory.hbm_write_bandwidth", 0)
        assert write_bw > ABSOLUTE_MIN_BW, (
            f"Write-only kernel write BW too low: {write_bw:.2f} GB/s"
        )

    # -- Bandwidth utilization --

    def test_bandwidth_utilization_range(self):
        """Copy kernel utilization should be meaningful (>5%) and <= 100%."""
        cmd = f"{self.binary} 0 0"  # copy, float4
        m = profile_longest_kernel(cmd, ["memory.hbm_bandwidth_utilization"])

        util = m.get("memory.hbm_bandwidth_utilization", 0)
        assert util > 5.0, (
            f"HBM utilization too low for a copy kernel: {util:.2f}%"
        )
        assert util <= 100.0, (
            f"HBM utilization exceeds 100%: {util:.2f}%"
        )

    # -- L2 bandwidth --

    def test_l2_bandwidth_nonzero(self):
        """Copy of 256MB data should show non-zero L2 bandwidth."""
        cmd = f"{self.binary} 0 0"  # copy, float4
        m = profile_longest_kernel(cmd, ["memory.l2_bandwidth"])

        l2_bw = m.get("memory.l2_bandwidth", 0)
        assert l2_bw > 0, f"L2 bandwidth is zero for a 256MB copy kernel"


# ---------------------------------------------------------------------------
# Standalone: run all benchmarks and print results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess, tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="hbm_bench_") as tmp:
        tmp_path = Path(tmp)
        binary = compile_hip("hbm_bandwidth.hip", tmp_path)
        arch = get_arch()
        print(f"Architecture: {arch}")
        print(f"Binary: {binary}")
        print()

        for mode_name, mode_arg in [("copy", "0"), ("read", "1"), ("write", "2")]:
            for dtype_name, dtype_arg in DTYPES.items():
                cmd = f"{binary} {mode_arg} {dtype_arg}"
                metrics = [
                    "memory.hbm_read_bandwidth",
                    "memory.hbm_write_bandwidth",
                    "memory.hbm_bandwidth_utilization",
                    "memory.bytes_transferred_hbm",
                    "memory.l2_bandwidth",
                ]
                try:
                    m = profile_longest_kernel(cmd, metrics)
                    print(f"{mode_name:5s} {dtype_name:8s}: ", end="")
                    for k, v in sorted(m.items()):
                        short = k.split(".")[-1]
                        print(f"  {short}={v:.2f}", end="")
                    print()
                except Exception as e:
                    print(f"{mode_name:5s} {dtype_name:8s}: ERROR - {e}")
