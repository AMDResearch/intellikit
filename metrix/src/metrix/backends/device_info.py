"""
Dynamic GPU device info via ``gpu_query`` HIP binary.

Ships ``gpu_query.hip`` as package data and compiles it on first use with
``hipcc``.  The compiled binary is cached next to the source file so
subsequent calls are instant.

Usage::

    from .device_info import query_device_specs

    specs = query_device_specs("gfx942")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .base import DeviceSpecs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# gpu_query binary: locate source, compile on first use, cache
# ---------------------------------------------------------------------------
_GPU_QUERY_SOURCE = "gpu_query.hip"


def _find_hip_source() -> Optional[Path]:
    """Locate gpu_query.hip shipped as package data."""
    pkg_dir = Path(__file__).resolve().parent
    src = pkg_dir / _GPU_QUERY_SOURCE
    if src.is_file():
        return src

    # Editable installs: also search via package __path__
    try:
        import metrix.backends as _pkg

        for p in _pkg.__path__:
            candidate = Path(p) / _GPU_QUERY_SOURCE
            if candidate.is_file():
                return candidate
    except (ImportError, AttributeError):
        pass

    return None


def _get_cache_dir() -> Path:
    """Return a user-writable cache directory for compiled binaries."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    cache = base / "metrix"
    try:
        cache.mkdir(parents=True, exist_ok=True)
        return cache
    except PermissionError:
        cache = Path("/tmp") / "metrix"
        cache.mkdir(parents=True, exist_ok=True)
        return cache


def _compile_gpu_query(source: Path) -> Path:
    """Compile gpu_query.hip with hipcc if not already cached."""
    cache = _get_cache_dir()
    binary = cache / "gpu_query"

    # Recompile if binary missing or source is newer
    if binary.is_file() and binary.stat().st_mtime >= source.stat().st_mtime:
        return binary

    hipcc = shutil.which("hipcc")
    if hipcc is None:
        raise RuntimeError("hipcc not found on PATH. Install ROCm or add hipcc to PATH.")

    logger.info("Compiling gpu_query.hip (first run only)...")
    try:
        proc = subprocess.run(
            [hipcc, str(source), "-o", str(binary), "-O2"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"hipcc timed out compiling gpu_query: {exc}") from exc

    if proc.returncode != 0:
        raise RuntimeError(f"hipcc failed (rc={proc.returncode}):\n{proc.stderr}")

    return binary


def _run_gpu_query(device_id: Optional[int] = None) -> List[dict]:
    """Run the gpu_query binary and return parsed JSON."""
    source = _find_hip_source()
    if source is None:
        raise RuntimeError(
            f"Cannot find {_GPU_QUERY_SOURCE} in metrix package data. "
            "Reinstall metrix or check your installation."
        )

    binary = _compile_gpu_query(source)

    cmd = [str(binary)]
    if device_id is not None:
        cmd.append(str(device_id))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"gpu_query failed: {exc}") from exc

    if proc.returncode != 0:
        raise RuntimeError(f"gpu_query failed (rc={proc.returncode}): {proc.stderr}")

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"gpu_query returned invalid JSON: {exc}\nOutput: {proc.stdout[:500]}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def query_device_specs(arch: str, device_id: int = 0) -> "DeviceSpecs":
    """
    Build a DeviceSpecs by querying the GPU via a compiled HIP binary.

    Raises RuntimeError if the requested *arch* does not match the
    installed GPU or if hipcc / HIP is unavailable.

    Args:
        arch: GFX architecture string (e.g. "gfx90a", "gfx942")
        device_id: HIP device index (default 0)

    Returns:
        Fully populated DeviceSpecs
    """
    from .base import DeviceSpecs

    results = _run_gpu_query(device_id)
    gpu = results[0]
    hw_arch = gpu["gcn_arch_name"].split(":")[0]

    if hw_arch != arch:
        raise RuntimeError(
            f"Architecture mismatch: requested {arch} but device {device_id} is {hw_arch}"
        )

    # Convert raw HIP values to DeviceSpecs units
    wavefront_size = gpu["wavefront_size"]
    max_waves_per_cu = (
        gpu["max_threads_per_multiprocessor"] // wavefront_size if wavefront_size > 0 else 0
    )
    base_clock_mhz = gpu["clock_rate_khz"] / 1000.0
    l2_size_mb = gpu["l2_cache_size_bytes"] / (1024.0 * 1024.0)
    lds_size_per_cu_kb = gpu["max_shared_memory_per_multiprocessor"] / 1024.0

    # Compute theoretical bandwidth from memory clock + bus width.
    # HIP's memoryClockRate is the MCLK from amdgpu_dpm_get_mclk():
    #   HBM2/HBM2e (gfx90a): MCLK = data strobe clock, DDR → 2x multiplier
    #   HBM3 (gfx94x):       MCLK = CK (command clock) = half the data strobe → 4x
    #   HBM3e (gfx95x):      same as HBM3 → 4x
    #   GDDR6 (RDNA):        MCLK = data clock, DDR → 2x
    # Ref: ROCm/rocm-systems projects/clr/rocclr/device/rocm/rocdevice.cpp
    mem_clock = gpu["memory_clock_rate_khz"]
    bus_width = gpu["memory_bus_width_bits"]
    if mem_clock <= 0 or bus_width <= 0:
        raise RuntimeError(
            f"Device {device_id} ({arch}) reported invalid memory specs: "
            f"memory_clock_rate_khz={mem_clock}, memory_bus_width_bits={bus_width}"
        )
    mem_multiplier = 4.0 if arch.startswith(("gfx94", "gfx95")) else 2.0
    hbm_bw_gbs = mem_multiplier * mem_clock * 1e3 * bus_width / 8.0 / 1e9

    return DeviceSpecs(
        arch=arch,
        name=gpu["name"],
        num_cu=gpu["num_cu"],
        max_waves_per_cu=max_waves_per_cu,
        wavefront_size=wavefront_size,
        base_clock_mhz=base_clock_mhz,
        hbm_bandwidth_gbs=hbm_bw_gbs,
        l2_size_mb=l2_size_mb,
        lds_size_per_cu_kb=lds_size_per_cu_kb,
    )
