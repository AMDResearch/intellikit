"""
Dynamic GPU device info from rocminfo / rocm-smi + peak spec lookup table.

Queryable values (num_cu, wavefront_size, etc.) are read from the live
system so that one backend class works for every SKU in an architecture
family (e.g. MI210 vs MI250X both use gfx90a).

Theoretical peak values (TFLOPS, HBM bandwidth) that cannot be read from
hardware are stored in a small per-chip-ID table with source links.

When the requested arch does not match the GPU actually installed (e.g.
unit tests creating a gfx942 backend on an MI210 machine), static
fallback specs are used instead.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .base import DeviceSpecs


# ---------------------------------------------------------------------------
# Peak specs that cannot be queried from hardware.
# Keyed by (gfx_arch, chip_id_hex) so different SKUs within the same arch
# get correct values.  chip_id_hex = None acts as the arch-level default.
#
# Sources are listed per-entry so they can be verified / updated.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Fallback specs used when the requested arch differs from the installed GPU
# (e.g. unit tests) or when rocminfo is unavailable.
#
# Sources:
#   HW specs:     https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
#   MI300X peaks: https://www.amd.com/en/products/accelerators/instinct/mi300/platform.html
#   MI210 peaks:  https://www.amd.com/en/products/accelerators/instinct/mi200/mi210.html
# ---------------------------------------------------------------------------
def _fallback_specs() -> Dict[str, "DeviceSpecs"]:
    from .base import DeviceSpecs

    return {
        "gfx942": DeviceSpecs(
            arch="gfx942",
            name="AMD Instinct MI300X",
            num_cu=304,
            max_waves_per_cu=32,
            wavefront_size=64,
            base_clock_mhz=2100.0,
            hbm_bandwidth_gbs=5300.0,
            l2_size_mb=256.0,
            lds_size_per_cu_kb=64.0,
        ),
        "gfx90a": DeviceSpecs(
            arch="gfx90a",
            name="AMD Instinct MI210",
            num_cu=104,
            max_waves_per_cu=32,
            wavefront_size=64,
            base_clock_mhz=1700.0,
            hbm_bandwidth_gbs=1600.0,
            l2_size_mb=8.0,
            lds_size_per_cu_kb=64.0,
        ),
    }


# HBM peak bandwidth per arch (only value not queryable from hardware)
_HBM_PEAK_GBS: Dict[str, float] = {
    "gfx942": 5300.0,
    "gfx90a": 1600.0,
}


# ---------------------------------------------------------------------------
# rocminfo parser — one call, structured results for the first GPU agent
# ---------------------------------------------------------------------------
@dataclass
class RocmInfoGPU:
    """Parsed GPU agent block from rocminfo."""

    arch: str = ""
    marketing_name: str = ""
    chip_id_hex: str = ""
    num_cu: int = 0
    simds_per_cu: int = 0
    max_waves_per_cu: int = 0
    wavefront_size: int = 64
    max_clock_mhz: int = 0
    l1_cache_kb: int = 0
    l2_cache_kb: int = 0
    lds_size_kb: int = 0


def _parse_rocminfo() -> RocmInfoGPU:
    """Run ``rocminfo`` and parse the first GPU agent."""
    try:
        proc = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"rocminfo unavailable: {exc}") from exc

    if proc.returncode != 0:
        raise RuntimeError(f"rocminfo failed (rc={proc.returncode}): {proc.stderr}")

    gpu = RocmInfoGPU()
    in_gpu_agent = False
    in_cache = False
    found_group_segment = False

    pending_agent_name = ""
    pending_marketing_name = ""

    for line in proc.stdout.splitlines():
        stripped = line.strip()

        if stripped.startswith("*******"):
            if in_gpu_agent:
                break
            pending_agent_name = ""
            pending_marketing_name = ""
            continue

        if stripped.startswith("Name:") and not in_gpu_agent:
            pending_agent_name = stripped.split(":", 1)[1].strip()
            continue

        if stripped.startswith("Marketing Name:") and not in_gpu_agent:
            pending_marketing_name = stripped.split(":", 1)[1].strip()
            continue

        if "Device Type:" in stripped and "GPU" in stripped:
            in_gpu_agent = True
            in_cache = False
            m = re.search(r"(gfx\w+)", pending_agent_name)
            if m:
                gpu.arch = m.group(1)
            gpu.marketing_name = pending_marketing_name
            continue

        if not in_gpu_agent:
            continue

        if "Device Type:" in stripped and "GPU" not in stripped:
            break

        if "Cache Info:" in stripped:
            in_cache = True
            continue
        if "Pool Info:" in stripped or "ISA Info:" in stripped:
            in_cache = False

        if in_cache:
            m = re.match(r"L1:\s+(\d+)", stripped)
            if m:
                gpu.l1_cache_kb = int(m.group(1))
            m = re.match(r"L2:\s+(\d+)", stripped)
            if m:
                gpu.l2_cache_kb = int(m.group(1))

        if stripped.startswith("Chip ID:"):
            m = re.search(r"\((0x[0-9a-fA-F]+)\)", stripped)
            if m:
                gpu.chip_id_hex = m.group(1).lower()
        elif stripped.startswith("Compute Unit:"):
            m = re.search(r"(\d+)", stripped.split(":")[1])
            if m:
                gpu.num_cu = int(m.group(1))
        elif stripped.startswith("SIMDs per CU:"):
            m = re.search(r"(\d+)", stripped.split(":")[1])
            if m:
                gpu.simds_per_cu = int(m.group(1))
        elif stripped.startswith("Max Waves Per CU:"):
            m = re.search(r"(\d+)", stripped.split(":")[1])
            if m:
                gpu.max_waves_per_cu = int(m.group(1))
        elif stripped.startswith("Wavefront Size:"):
            m = re.search(r"(\d+)", stripped.split(":")[1])
            if m:
                gpu.wavefront_size = int(m.group(1))
        elif stripped.startswith("Max Clock Freq"):
            m = re.search(r"(\d+)", stripped.split(":")[1])
            if m:
                gpu.max_clock_mhz = int(m.group(1))
        elif "Segment:" in stripped and "GROUP" in stripped:
            found_group_segment = True
        elif found_group_segment and stripped.startswith("Size:"):
            m = re.search(r"(\d+)", stripped.split(":")[1])
            if m:
                gpu.lds_size_kb = int(m.group(1))
            found_group_segment = False

    return gpu



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def query_device_specs(arch: str) -> "DeviceSpecs":
    """
    Build a DeviceSpecs by querying rocminfo/rocm-smi for live values.

    If the requested *arch* does not match the GPU actually installed
    (common in unit tests), a static fallback is returned instead.

    Args:
        arch: GFX architecture string (e.g. "gfx90a", "gfx942")

    Returns:
        Fully populated DeviceSpecs
    """
    from .base import DeviceSpecs

    # Try live query
    hw_arch = None
    try:
        gpu = _parse_rocminfo()
        hw_arch = gpu.arch or None
    except RuntimeError:
        gpu = None

    # If the hardware matches the requested arch, use live values
    if gpu and hw_arch == arch:
        return DeviceSpecs(
            arch=arch,
            name=gpu.marketing_name or f"AMD GPU ({arch})",
            num_cu=gpu.num_cu,
            max_waves_per_cu=gpu.max_waves_per_cu,
            wavefront_size=gpu.wavefront_size,
            base_clock_mhz=float(gpu.max_clock_mhz),
            hbm_bandwidth_gbs=_HBM_PEAK_GBS.get(arch, 0.0),
            l2_size_mb=gpu.l2_cache_kb / 1024.0,
            lds_size_per_cu_kb=float(gpu.lds_size_kb or 64),
        )

    # Arch mismatch or rocminfo unavailable — use static fallback
    fallback = _fallback_specs()
    if arch in fallback:
        return fallback[arch]

    return DeviceSpecs(
        arch=arch,
        name=f"AMD GPU ({arch})",
        hbm_bandwidth_gbs=_HBM_PEAK_GBS.get(arch, 0.0),
    )
