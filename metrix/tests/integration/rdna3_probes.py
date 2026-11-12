# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Probe kernels with analytically known counter values on gfx1100 (RDNA3).

Each probe is a small HIP program whose launch shape and instruction stream
fix what the hardware counters must read:

* ``stream_copy``    -- float4 copy of 1 GiB: VRAM traffic and bandwidth.
* ``buffer_copy``    -- raw buffer loads/stores: wave and instruction counts
  only, since the TA buffer metrics are unsupported on gfx1100.
* ``alu_fma_full`` / ``alu_fma_half`` -- FMA loops at full and half occupancy:
  wave and instruction counts, GPU busy, and occupancy measured from per-wave
  timestamps.
* ``cache_warm`` / ``cache_resident`` -- agent-scope loads of a 1 MiB buffer,
  first cold and then 2000 times while L2-resident: wave and instruction
  counts only, since the L2 metrics are unsupported on gfx1100.
* ``lds_stride_<S>`` -- LDS reads at dword stride S: bank-conflict model.

``build_probe`` compiles a probe and keeps the device assembly that went into
the binary; ``IsaCounts`` reads instruction counts from it, so instruction
expectations come from the code that actually ran rather than from the HIP
source. The ``expected_*`` functions turn launch shape, assembly and measured
kernel duration into expected values, and the tolerance constants say how
close a measurement must be.

The gfx1100 integration tests in test_counter_validation.py use this module.
"""

import errno
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ARCH = "gfx1100"
WAVE_SIZE = 32

# The gfx1100 metrics in counter_defs.yaml; their counters read correctly only
# at the profile_standard perf level.
SAFE_METRICS = (
    "compute.gpu_utilization",
    "GPUBusy",
    "Wavefronts",
    "VALUInsts",
    "SALUInsts",
    "OccupancyPercent",
    "memory.hbm_read_bandwidth",
    "memory.hbm_write_bandwidth",
    "memory.hbm_bandwidth_utilization",
    "memory.bytes_transferred_hbm",
    "memory.lds_bank_conflicts",
    "memory.lds_bank_conflict_percent",
)
PROFILE_STANDARD = "profile_standard"
PERF_LEVEL_FILE = "power_dpm_force_performance_level"
PERF_LEVEL_EBUSY_RETRIES = 50
SYSFS_PCI_DEVICES = Path("/sys/bus/pci/devices")
# Max resident waves per CU on gfx11: 16 per SIMD x 2 SIMDs. rocprofiler-sdk's
# OccupancyPercent divides by the same constant.
MAX_WAVES_PER_CU = 32
LDS_BANKS = 32

# --- Tolerances -------------------------------------------------------------
BYTES_REL_TOL = 0.05
BANDWIDTH_REL_TOL = 0.10
UTILIZATION_REL_TOL = 0.10
INST_REL_TOL = 0.02
BUSY_MIN_PERCENT = 95.0
OCCUPANCY_REL_TOL = 0.10
LDS_CONFLICT_ZERO_MAX = 0.05
LDS_CONFLICT_REL_TOL = 0.10

# --- Launch shapes ----------------------------------------------------------
STREAM_BYTES = 1 << 30
STREAM_BLOCK = 256

BUFFER_BLOCKS = 1024
BUFFER_BLOCK = 256
BUFFER_ITERS = 64

ALU_BLOCK = 256
ALU_ITERS = 20000
ALU_CHAINS = 8

CACHE_ELEMS = 1 << 18  # 1 MiB of 32-bit words
CACHE_BLOCK = 256
CACHE_ITERS = 2000

LDS_STRIDES = (1, 2, 3, 4, 8, 16, 32, 64)
LDS_ELEMS = 8192
LDS_BLOCKS = 192
LDS_BLOCK = 256
LDS_ITERS = 4096

# gfx11 raw-buffer descriptor dword 3 (ck_tile CK_TILE_BUFFER_RESOURCE_3RD_DWORD).
GFX11_BUFFER_RSRC_DWORD3 = 0x31004000

HIP_HEADER = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call) do {                                         \
    hipError_t err = (call);                                         \
    if (err != hipSuccess) {                                         \
        fprintf(stderr, "HIP error: %s at %s:%d\n",                 \
                hipGetErrorString(err), __FILE__, __LINE__);         \
        exit(1);                                                     \
    }                                                                \
} while(0)
"""


# =========================================================================
# Sources
# =========================================================================

STREAM_COPY_SRC = (
    HIP_HEADER
    + rf"""
extern "C" __global__ void stream_copy(const float4* __restrict__ in,
                                       float4* __restrict__ out, size_t n) {{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}}
int main() {{
    const size_t bytes = {STREAM_BYTES}ull, n = bytes / sizeof(float4);
    float4 *in, *out;
    HIP_CHECK(hipMalloc(&in, bytes));
    HIP_CHECK(hipMalloc(&out, bytes));
    HIP_CHECK(hipMemset(in, 1, bytes));
    HIP_CHECK(hipMemset(out, 0, bytes));
    HIP_CHECK(hipDeviceSynchronize());
    const unsigned blocks = (unsigned)(n / {STREAM_BLOCK});
    for (int r = 0; r < 2; r++) stream_copy<<<blocks, {STREAM_BLOCK}>>>(in, out, n);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));
    return 0;
}}
"""
)

BUFFER_COPY_SRC = (
    HIP_HEADER
    + rf"""
extern "C" __global__ void buffer_copy(float* in, float* out, int iters) {{
    const int total = gridDim.x * blockDim.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int bytes = total * iters * 4;
    __amdgpu_buffer_rsrc_t src = __builtin_amdgcn_make_buffer_rsrc(in, 0, bytes, {GFX11_BUFFER_RSRC_DWORD3:#x});
    __amdgpu_buffer_rsrc_t dst = __builtin_amdgcn_make_buffer_rsrc(out, 0, bytes, {GFX11_BUFFER_RSRC_DWORD3:#x});
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {{
        int off = (i * total + gid) * 4;
        unsigned v = __builtin_amdgcn_raw_buffer_load_b32(src, off, 0, 0);
        __builtin_amdgcn_raw_buffer_store_b32(v + 1u, dst, off, 0, 0);
    }}
}}
int main() {{
    const int blocks = {BUFFER_BLOCKS}, block = {BUFFER_BLOCK}, iters = {BUFFER_ITERS};
    const size_t bytes = (size_t)blocks * block * iters * 4;
    float *in, *out;
    HIP_CHECK(hipMalloc(&in, bytes));
    HIP_CHECK(hipMalloc(&out, bytes));
    HIP_CHECK(hipMemset(in, 0, bytes));
    HIP_CHECK(hipDeviceSynchronize());
    for (int r = 0; r < 2; r++) buffer_copy<<<blocks, block>>>(in, out, iters);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));
    return 0;
}}
"""
)


def _alu_kernel(name: str) -> str:
    chains = range(ALU_CHAINS)
    init = "\n".join(f"    float c{k} = threadIdx.x * {k + 1}e-6f;" for k in chains)
    body = "\n".join(f"        c{k} = __builtin_fmaf(c{k}, m, a);" for k in chains)
    total = " + ".join(f"c{k}" for k in chains)
    return f"""
extern "C" __global__ void {name}(float* __restrict__ out, long long* __restrict__ times,
                                  int iters, float m, float a) {{
    const long long t0 = wall_clock64();
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
{init}
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {{
{body}
    }}
    out[gid] = {total};
    const long long t1 = wall_clock64();
    if ((threadIdx.x & {WAVE_SIZE - 1}) == 0) {{
        times[2 * (gid / {WAVE_SIZE})] = t0;
        times[2 * (gid / {WAVE_SIZE}) + 1] = t1;
    }}
}}
"""


def alu_blocks(num_cu: int, fraction: float) -> int:
    """Blocks that fill ``fraction`` of the device's resident-wave capacity."""
    waves_per_block = ALU_BLOCK // WAVE_SIZE
    return int(round(num_cu * MAX_WAVES_PER_CU * fraction / waves_per_block))


WAVE_LOG_ENV = "RDNA3_PROBE_WAVE_LOG"
WAVE_LOG_DEFAULT = "alu_fma_waves.txt"


def alu_source(num_cu: int) -> str:
    """FMA probe; half fills half the wave slots once, full oversubscribes 16x.

    Each wave stores its start and end ``wall_clock64()`` reading. After each
    launch the host appends ``<kernel> <waves> <sum of wave lifetimes> <span>``
    (in wall-clock ticks) to the file named by ``$RDNA3_PROBE_WAVE_LOG``, which
    ``expected_occupancy_percent`` turns into an independent occupancy value.
    """
    half = alu_blocks(num_cu, 0.5)
    full = alu_blocks(num_cu, 16.0)
    launch = r"""
static void launch_{k}(int blocks, int block, int iters, float* out, long long* d_times,
                       long long* h_times, FILE* log) {{
    const int waves = blocks * block / {wave};
    HIP_CHECK(hipMemset(d_times, 0, (size_t)waves * 2 * sizeof(long long)));
    {k}<<<blocks, block>>>(out, d_times, iters, 0.999f, 1e-3f);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_times, d_times, (size_t)waves * 2 * sizeof(long long),
                        hipMemcpyDeviceToHost));
    long long lo = h_times[0], hi = h_times[1], sum = 0;
    for (int w = 0; w < waves; w++) {{
        const long long s = h_times[2 * w], e = h_times[2 * w + 1];
        sum += e - s;
        if (s < lo) lo = s;
        if (e > hi) hi = e;
    }}
    fprintf(log, "{k} %d %lld %lld\n", waves, sum, hi - lo);
    fflush(log);
}}
"""
    return (
        HIP_HEADER
        + _alu_kernel("alu_fma_half")
        + _alu_kernel("alu_fma_full")
        + launch.format(k="alu_fma_half", wave=WAVE_SIZE)
        + launch.format(k="alu_fma_full", wave=WAVE_SIZE)
        + f"""
int main() {{
    const int half = {half}, full = {full}, block = {ALU_BLOCK}, iters = {ALU_ITERS};
    const size_t max_waves = (size_t)full * block / {WAVE_SIZE};
    const char* log_path = getenv("{WAVE_LOG_ENV}");
    FILE* log = fopen(log_path ? log_path : "{WAVE_LOG_DEFAULT}", "a");
    if (!log) {{ perror("wave log"); return 1; }}
    float* out;
    long long *d_times, *h_times = (long long*)malloc(max_waves * 2 * sizeof(long long));
    HIP_CHECK(hipMalloc(&out, (size_t)full * block * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_times, max_waves * 2 * sizeof(long long)));
    HIP_CHECK(hipDeviceSynchronize());
    launch_alu_fma_half(half, block, iters, out, d_times, h_times, log);
    launch_alu_fma_full(full, block, iters, out, d_times, h_times, log);
    fclose(log);
    free(h_times);
    HIP_CHECK(hipFree(d_times));
    HIP_CHECK(hipFree(out));
    return 0;
}}
"""
    )


def _cache_kernel(name: str) -> str:
    return f"""
extern "C" __global__ void {name}(const unsigned* __restrict__ in, unsigned* __restrict__ out, int iters) {{
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {{
        unsigned idx = (gid + (unsigned)i * {CACHE_BLOCK}u) & {CACHE_ELEMS - 1}u;
        acc += __hip_atomic_load(&in[idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }}
    out[gid] = acc;
}}
"""


CACHE_RESIDENT_SRC = (
    HIP_HEADER
    + _cache_kernel("cache_warm")
    + _cache_kernel("cache_resident")
    + rf"""
int main() {{
    const int elems = {CACHE_ELEMS}, block = {CACHE_BLOCK}, blocks = elems / block;
    unsigned *in, *out;
    HIP_CHECK(hipMalloc(&in, elems * sizeof(unsigned)));
    HIP_CHECK(hipMalloc(&out, elems * sizeof(unsigned)));
    HIP_CHECK(hipMemset(in, 1, elems * sizeof(unsigned)));
    HIP_CHECK(hipDeviceSynchronize());
    cache_warm<<<blocks, block>>>(in, out, 1);
    cache_resident<<<blocks, block>>>(in, out, {CACHE_ITERS});
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));
    return 0;
}}
"""
)


def _lds_kernel(stride: int) -> str:
    return f"""
extern "C" __global__ void lds_stride_{stride}(float* __restrict__ out, int iters) {{
    __shared__ float lds[{LDS_ELEMS}];
    const int tid = threadIdx.x;
    #pragma unroll
    for (int j = 0; j < {LDS_ELEMS // LDS_BLOCK}; j++) lds[tid + j * {LDS_BLOCK}] = (float)(tid + j);
    __syncthreads();
    float acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) acc += lds[((tid + {WAVE_SIZE} * i) * {stride}) & {LDS_ELEMS - 1}];
    out[blockIdx.x * blockDim.x + tid] = acc;
}}
"""


LDS_STRIDE_SRC = (
    HIP_HEADER
    + "".join(_lds_kernel(s) for s in LDS_STRIDES)
    + f"""
int main() {{
    const int blocks = {LDS_BLOCKS}, block = {LDS_BLOCK}, iters = {LDS_ITERS};
    float* out;
    HIP_CHECK(hipMalloc(&out, (size_t)blocks * block * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());
"""
    + "".join(
        f"    lds_stride_{s}<<<blocks, block>>>(out, iters);\n    HIP_CHECK(hipDeviceSynchronize());\n"
        for s in LDS_STRIDES
    )
    + """    HIP_CHECK(hipFree(out));
    return 0;
}
"""
)


# =========================================================================
# Build and ISA
# =========================================================================


@dataclass
class IsaCounts:
    """Instruction counts of one kernel, split into its loop body and the rest.

    Every probe kernel has at most one loop, run ``iters`` times, so the
    dynamic count per wave is ``outside + iters * body`` for each class.
    """

    kernel: str
    body: Dict[str, int]
    outside: Dict[str, int]
    body_lines: List[str] = field(default_factory=list)

    def per_wave(self, klass: str, iters: int) -> int:
        return self.outside.get(klass, 0) + iters * self.body.get(klass, 0)


_SMEM = (
    "s_load",
    "s_buffer_load",
    "s_store",
    "s_buffer_store",
    "s_dcache",
    "s_gl1_inv",
    "s_memtime",
    "s_memrealtime",
    "s_atc_probe",
    "s_scratch",
)
_SOPP = (
    "s_nop",
    "s_endpgm",
    "s_branch",
    "s_cbranch",
    "s_barrier",
    "s_waitcnt",
    "s_sleep",
    "s_setprio",
    "s_sendmsg",
    "s_trap",
    "s_icache_inv",
    "s_incperflevel",
    "s_decperflevel",
    "s_ttracedata",
    "s_delay_alu",
    "s_clause",
    "s_set_inst_prefetch_distance",
    "s_inst_prefetch",
    "s_code_end",
    "s_wakeup",
    "s_sethalt",
    "s_setkill",
    "s_round_mode",
    "s_denorm_mode",
)


def classify(mnemonic: str) -> str:
    """Instruction class: valu, salu, sopp, smem, ds, vmem_load, vmem_store or other.

    ``salu`` is SOP1/SOP2/SOPC/SOPK arithmetic; waits, branches and other
    SOPP-style sequencer instructions are ``sopp``.
    """
    if mnemonic.startswith("v_"):
        return "valu"
    if mnemonic.startswith("ds_"):
        return "ds"
    if mnemonic.startswith(("global_load", "buffer_load", "flat_load", "scratch_load")):
        return "vmem_load"
    if mnemonic.startswith(("global_store", "buffer_store", "flat_store", "scratch_store")):
        return "vmem_store"
    if mnemonic.startswith(_SMEM):
        return "smem"
    if mnemonic.startswith(_SOPP):
        return "sopp"
    if mnemonic.startswith("s_"):
        return "salu"
    return "other"


def parse_kernel_isa(asm: str, kernel: str) -> IsaCounts:
    """Split one kernel's instructions in compiler assembly into loop body and rest."""
    start = re.search(rf"^{re.escape(kernel)}:\s*(?:;.*)?$", asm, re.M)
    if start is None:
        raise ValueError(f"kernel {kernel} not found in assembly")
    end = re.compile(r"^\.Lfunc_end\d+:", re.M).search(asm, start.end())
    lines = asm[start.end() : end.start() if end else len(asm)].splitlines()

    entries = []  # ("label", name) or ("inst", mnemonic, operands, text)
    for raw in lines:
        text = raw.split(";", 1)[0].rstrip()
        if not text.strip():
            continue
        label = re.match(r"^(\.?[\w.$]+):", text)
        if label:
            entries.append(("label", label.group(1)))
            continue
        stripped = text.strip()
        if stripped.startswith("."):
            continue
        parts = stripped.split(None, 1)
        entries.append(("inst", parts[0], parts[1] if len(parts) > 1 else "", stripped))

    label_pos = {e[1]: i for i, e in enumerate(entries) if e[0] == "label"}
    loops = []
    for i, e in enumerate(entries):
        if e[0] == "inst" and e[1].startswith(("s_cbranch", "s_branch")):
            target = e[2].split(",")[0].strip()
            if target in label_pos and label_pos[target] < i:
                loops.append((label_pos[target], i))
    if len(loops) > 1:
        raise ValueError(f"{kernel}: expected at most one loop, found {len(loops)}")
    lo, hi = loops[0] if loops else (-1, -1)

    body, outside, body_lines = {}, {}, []
    for i, e in enumerate(entries):
        if e[0] != "inst":
            continue
        bucket = body if lo <= i <= hi else outside
        klass = classify(e[1])
        bucket[klass] = bucket.get(klass, 0) + 1
        if bucket is body:
            body_lines.append(e[3])
    return IsaCounts(kernel=kernel, body=body, outside=outside, body_lines=body_lines)


@dataclass
class BuiltProbe:
    name: str
    binary: Path
    asm: Path
    isa: Dict[str, IsaCounts]


def build_probe(name: str, source: str, kernels: List[str], out_dir: Path) -> BuiltProbe:
    """Compile a probe for gfx1100, keep its device assembly and parse each kernel."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / f"{name}.hip"
    binary = out_dir / name
    src.write_text(source)
    r = subprocess.run(
        ["hipcc", f"--offload-arch={ARCH}", "-O2", "-save-temps", str(src), "-o", str(binary)],
        capture_output=True,
        text=True,
        cwd=out_dir,
        timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError(f"hipcc failed for {name}:\n{r.stderr}")
    asm = out_dir / f"{name}-hip-amdgcn-amd-amdhsa-{ARCH}.s"
    if not asm.is_file():
        raise RuntimeError(f"hipcc -save-temps did not leave {asm.name} in {out_dir}")
    text = asm.read_text()
    return BuiltProbe(name, binary, asm, {k: parse_kernel_isa(text, k) for k in kernels})


def rocminfo_compute_units(arch: str = ARCH) -> int:
    """CU count of the first visible ``arch`` agent, as rocminfo reports it.

    HIP's multiProcessorCount on gfx11 counts WGPs (48 on a W7900), while
    rocprofiler's CU_NUM and the occupancy capacity count CUs (96).
    """
    out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=120).stdout
    for agent in re.split(r"\n\*+\s*\nAgent \d+\s*\n\*+", out):
        if re.search(rf"^\s*Name:\s+{re.escape(arch)}\s*$", agent, re.M):
            m = re.search(r"^\s*Compute Unit:\s+(\d+)", agent, re.M)
            if m:
                return int(m.group(1))
    raise RuntimeError(f"rocminfo lists no {arch} agent with a CU count")


def perf_level(device_id: int = 0) -> str:
    """The GPU's amdgpu perf level from sysfs, such as ``auto`` or ``profile_standard``.

    Reads of a runtime-suspended card fail with EBUSY, so the render node is held
    open during the read and EBUSY is retried.
    """
    from metrix.backends.device_info import _run_gpu_query

    gpu = _run_gpu_query(device_id)[0]
    bdf = f"{gpu['pci_domain_id']:04x}:{gpu['pci_bus_id']:02x}:{gpu['pci_device_id']:02x}.0"
    device = SYSFS_PCI_DEVICES / bdf
    renders = sorted((device / "drm").glob("renderD*"))
    fd = os.open(f"/dev/dri/{renders[0].name}", os.O_RDWR | os.O_CLOEXEC) if renders else None
    try:
        for attempt in range(PERF_LEVEL_EBUSY_RETRIES):
            try:
                return (device / PERF_LEVEL_FILE).read_text().strip()
            except OSError as exc:
                if exc.errno != errno.EBUSY or attempt == PERF_LEVEL_EBUSY_RETRIES - 1:
                    raise
                time.sleep(0.1)
        raise AssertionError("unreachable")
    finally:
        if fd is not None:
            os.close(fd)


def build_all(out_dir: Path, num_cu: int) -> Dict[str, BuiltProbe]:
    """Compile every probe into ``out_dir``. ``num_cu`` is rocprofiler's CU_NUM."""
    lds_kernels = [f"lds_stride_{s}" for s in LDS_STRIDES]
    return {
        "stream_copy": build_probe("stream_copy", STREAM_COPY_SRC, ["stream_copy"], out_dir),
        "buffer_copy": build_probe("buffer_copy", BUFFER_COPY_SRC, ["buffer_copy"], out_dir),
        "alu_fma": build_probe(
            "alu_fma", alu_source(num_cu), ["alu_fma_half", "alu_fma_full"], out_dir
        ),
        "cache_resident": build_probe(
            "cache_resident", CACHE_RESIDENT_SRC, ["cache_warm", "cache_resident"], out_dir
        ),
        "lds_stride": build_probe("lds_stride", LDS_STRIDE_SRC, lds_kernels, out_dir),
    }


# =========================================================================
# Expected values
# =========================================================================


@dataclass(frozen=True)
class Expectation:
    """What one metric should read, and how close counts as a pass.

    ``kind`` is ``rel`` (|measured - value| <= tol * value), ``min``
    (measured >= value), ``max`` (measured <= value) or ``exact``.
    """

    value: float
    kind: str
    tol: float = 0.0

    def check(self, measured: Optional[float]) -> bool:
        if measured is None or (isinstance(measured, float) and math.isnan(measured)):
            return False
        if self.kind == "exact":
            return measured == self.value
        if self.kind == "min":
            return measured >= self.value
        if self.kind == "max":
            return measured <= self.value
        if self.value == 0:
            return measured == 0
        return abs(measured - self.value) <= self.tol * abs(self.value)

    def describe(self) -> str:
        if self.kind == "exact":
            return f"= {self.value:,.0f}"
        if self.kind == "min":
            return f">= {self.value:g}"
        if self.kind == "max":
            return f"<= {self.value:g}"
        return f"{self.value:,.4g} +/- {self.tol:.0%}"


def _gbs(nbytes: float, duration_us: float) -> float:
    return nbytes / duration_us / 1e3


def expected_wave_instructions(isa: IsaCounts, iters: int, waves: int) -> Dict[str, Expectation]:
    """Launched waves, and per-wave VALU/SALU counts from the kernel's own assembly."""
    return {
        "Wavefronts": Expectation(waves, "exact"),
        "VALUInsts": Expectation(isa.per_wave("valu", iters), "rel", INST_REL_TOL),
        "SALUInsts": Expectation(isa.per_wave("salu", iters), "rel", INST_REL_TOL),
    }


def expected_stream_copy(
    duration_us: float, peak_gbs: float, dispatches: int = 1
) -> Dict[str, Expectation]:
    """1 GiB read and 1 GiB written once per dispatch, all over the L2-to-fabric interface.

    ``memory.bytes_transferred_hbm`` is a correlated metric: aggregated by kernel
    it sums the kernel's ``dispatches`` in one run.
    """
    n = float(STREAM_BYTES)
    return {
        "memory.bytes_transferred_hbm": Expectation(2 * n * dispatches, "rel", BYTES_REL_TOL),
        "memory.hbm_read_bandwidth": Expectation(_gbs(n, duration_us), "rel", BANDWIDTH_REL_TOL),
        "memory.hbm_write_bandwidth": Expectation(_gbs(n, duration_us), "rel", BANDWIDTH_REL_TOL),
        "memory.hbm_bandwidth_utilization": Expectation(
            100.0 * _gbs(2 * n, duration_us) / peak_gbs, "rel", UTILIZATION_REL_TOL
        ),
    }


@dataclass(frozen=True)
class WaveLog:
    """One ALU launch as the probe saw it: wave count, summed lifetimes, span (ticks)."""

    kernel: str
    waves: int
    lifetime_ticks: int
    span_ticks: int


def read_wave_log(path: Path) -> List[WaveLog]:
    entries = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 4:
            entries.append(WaveLog(parts[0], int(parts[1]), int(parts[2]), int(parts[3])))
    return entries


def wave_occupancy_percent(logs: List[WaveLog], kernel: str, num_cu: int) -> float:
    """Mean resident waves over the kernel span as a percent of wave slots.

    Averaged over every launch of ``kernel`` in the log (one per profiling
    pass). This measures the quantity OccupancyPercent reports, but from
    in-kernel timestamps instead of SQ_WAVE_CYCLES / GRBM_GUI_ACTIVE.
    """
    values = [
        100.0 * w.lifetime_ticks / (w.span_ticks * num_cu * MAX_WAVES_PER_CU)
        for w in logs
        if w.kernel == kernel and w.span_ticks > 0
    ]
    if not values:
        raise ValueError(f"no wave-log entries for {kernel}")
    return sum(values) / len(values)


def expected_alu(
    isa: IsaCounts, blocks: int, occupancy_percent: Optional[float] = None
) -> Dict[str, Expectation]:
    """FMA loop: exact wave count, ISA instruction counts, busy GPU, occupancy.

    ``VALUInsts`` and ``SALUInsts`` are per-wave averages (SQ_INSTS_x / SQ_WAVES).
    ``occupancy_percent`` comes from ``wave_occupancy_percent``; the scheduler
    decides how waves spread over CUs, so the launch shape alone does not fix it.
    """
    expected = expected_wave_instructions(isa, ALU_ITERS, blocks * ALU_BLOCK // WAVE_SIZE)
    expected["compute.gpu_utilization"] = Expectation(BUSY_MIN_PERCENT, "min")
    expected["GPUBusy"] = Expectation(BUSY_MIN_PERCENT, "min")
    if occupancy_percent is not None:
        expected["OccupancyPercent"] = Expectation(occupancy_percent, "rel", OCCUPANCY_REL_TOL)
    return expected


def lds_conflict_degree(stride: int, banks: int = LDS_BANKS, lanes: int = WAVE_SIZE) -> int:
    """Lanes of one wave32 LDS access that land in the same bank at dword stride."""
    distinct_banks = banks // math.gcd(stride, banks)
    return max(1, lanes // distinct_banks)


def lds_conflict_percent_model(stride: int) -> float:
    """Percent of LDS-active cycles stalled: a degree-d conflict costs d cycles, d-1 stalled."""
    d = lds_conflict_degree(stride)
    return 100.0 * (d - 1) / d


def expected_lds_conflict(stride: int) -> Dict[str, Expectation]:
    metric = "memory.lds_bank_conflict_percent"
    model = lds_conflict_percent_model(stride)
    if model == 0:
        return {metric: Expectation(LDS_CONFLICT_ZERO_MAX, "max")}
    return {metric: Expectation(model, "rel", LDS_CONFLICT_REL_TOL)}


def expected_lds_conflicts_per_access(stride: int, isa: IsaCounts) -> Dict[str, Expectation]:
    """Bank-conflict stall cycles per LDS instruction.

    Each loop read stalls d - 1 cycles; the stride-1 initialisation stores
    outside the loop stall none but still count as LDS instructions.
    """
    loop_reads = isa.body.get("ds", 0) * LDS_ITERS
    value = (lds_conflict_degree(stride) - 1) * loop_reads / isa.per_wave("ds", LDS_ITERS)
    if value == 0:
        return {"memory.lds_bank_conflicts": Expectation(LDS_CONFLICT_ZERO_MAX, "max")}
    return {"memory.lds_bank_conflicts": Expectation(value, "rel", LDS_CONFLICT_REL_TOL)}


def expected_for_kernel(
    kernel: str,
    isa: IsaCounts,
    duration_us: float,
    peak_gbs: float,
    num_cu: int,
    wave_logs: Optional[List[WaveLog]] = None,
    dispatches: int = 1,
) -> Dict[str, Expectation]:
    """Every expectation that applies to one probe kernel, keyed by metric name.

    ``dispatches`` is the kernel's launch count per run (``KernelResults.dispatch_count``).
    """
    if kernel == "stream_copy":
        waves = STREAM_BYTES // 16 // WAVE_SIZE
        return {
            **expected_wave_instructions(isa, 0, waves),
            **expected_stream_copy(duration_us, peak_gbs, dispatches),
        }
    if kernel == "buffer_copy":
        return expected_wave_instructions(
            isa, BUFFER_ITERS, BUFFER_BLOCKS * BUFFER_BLOCK // WAVE_SIZE
        )
    if kernel in ("alu_fma_half", "alu_fma_full"):
        blocks = alu_blocks(num_cu, 0.5 if kernel.endswith("half") else 16.0)
        occupancy = wave_occupancy_percent(wave_logs, kernel, num_cu) if wave_logs else None
        return expected_alu(isa, blocks, occupancy)
    if kernel == "cache_warm":
        return expected_wave_instructions(isa, 1, CACHE_ELEMS // WAVE_SIZE)
    if kernel == "cache_resident":
        return expected_wave_instructions(isa, CACHE_ITERS, CACHE_ELEMS // WAVE_SIZE)
    if kernel.startswith("lds_stride_"):
        stride = int(kernel.rsplit("_", 1)[1])
        return {
            **expected_wave_instructions(isa, LDS_ITERS, LDS_BLOCKS * LDS_BLOCK // WAVE_SIZE),
            **expected_lds_conflict(stride),
            **expected_lds_conflicts_per_access(stride, isa),
        }
    raise KeyError(kernel)


PROBE_KERNEL_REGEX = {
    "stream_copy": "^stream_copy$",
    "buffer_copy": "^buffer_copy$",
    "alu_fma": "^alu_fma_(half|full)$",
    "cache_resident": "^cache_(warm|resident)$",
    "lds_stride": "^lds_stride_",
}
