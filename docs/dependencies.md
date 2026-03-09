# IntelliKit Dependencies

This document describes all dependencies required by each tool in the IntelliKit monorepo. It covers only dependencies that must be **manually installed** — Python packages installed automatically by `pip`/`uv` and C++ libraries fetched automatically via CMake FetchContent/CPM are not listed here.

---

## Shared Requirements

All GPU tools require:

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.10 (root metapackage) | Individual tools may support older versions (see below) |
| AMD ROCm | >= 6.0 | Provides GPU runtime, compilers, and profiling tools |
| AMD GPU | MI300+ | Required for GPU profiling tools |

---

## Tool-by-Tool Dependencies

### Accordo

**Purpose:** Automated correctness validation for GPU kernel optimizations.

#### System Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.8 | |
| ROCm | >= 6.0 | Provides `hipcc` (C++ compiler) and HSA runtime (`libhsa-runtime64`) |
| CMake | >= 3.22 | Required to build the C++ library at runtime |
| `libdwarf` | system package | Required by kernelDB (see below) |
| `libelf` | system package | Required by kernelDB (see below) |
| `libzstd` | system package | Required by kernelDB (see below) |

The C++ library is built automatically during `pip install`. kernelDB (fetched automatically from GitHub) requires `libdwarf`, `libelf`, and `libzstd` to be present on the host system — see [kernelDB dependencies](#kerneldb-system-dependencies) below.

---

### Linex

**Purpose:** Source-level GPU performance profiling – maps GPU cycles to source code lines.

#### System Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.8 | |
| ROCm | >= 7.0 | Must include `rocprofv3` |
| `rocprofv3` | from ROCm 7.0+ | Used to capture SQTT traces |

Linex invokes `rocprofv3` via subprocess to collect hardware SQTT (Shader Queue Thread Trace) data. The binary being profiled must be compiled with `-g` for source-line mapping.

---

### Metrix

**Purpose:** Hardware counter profiling with human-readable metrics.

#### System Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.9 | |
| ROCm | >= 6.x | Must include `rocprofv3` and `rocminfo` |
| `rocprofv3` | from ROCm 6.x | Used for hardware counter collection |
| `rocminfo` | from ROCm | Used to auto-detect GPU architecture |

Metrix invokes `rocprofv3` to collect hardware counters and `rocminfo` to auto-detect the GPU architecture (e.g., `gfx942` for MI300X). Supported architectures: `gfx942` (MI300X), `gfx90a` (MI200 series), `gfx1201`.

---

### Nexus

**Purpose:** HSA packet source code extractor – intercepts GPU kernel launches to extract source and assembly.

#### System Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.8 | |
| ROCm | >= 6.0 | Provides HSA runtime (`libhsa-runtime64`) and LLVM |
| CMake | >= 3.22 | Required to build the C++ shared library |
| LLVM | from ROCm (`/opt/rocm/llvm`) | Set via `LLVM_INSTALL_DIR` CMake variable |
| `libdwarf` | system package | Required by kernelDB (see below) |
| `libelf` | system package | Required by kernelDB (see below) |
| `libzstd` | system package | Required by kernelDB (see below) |

The C++ shared library (`libnexus.so`) is built during `pip install` via scikit-build-core and CMake. kernelDB (fetched automatically from GitHub via CPM) requires `libdwarf`, `libelf`, and `libzstd` to be present on the host system — see [kernelDB dependencies](#kerneldb-system-dependencies) below.

---

### ROCm-MCP (`rocm_mcp`)

**Purpose:** Collection of MCP servers exposing ROCm tools to LLMs (HIP compiler, HIP docs, system info).

#### System Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.10 | |
| `hipcc` | from ROCm | Required by `hip-compiler-mcp`; defaults to `hipcc` on PATH |
| `rocminfo` | from ROCm | Required by `rocminfo-mcp`; defaults to `rocminfo` on PATH |

Individual servers have different runtime requirements:

| Server | Required System Tool |
|---|---|
| `hip-compiler-mcp` | `hipcc` (from ROCm, or set `INTELLIKIT_HIPCC`) |
| `hip-docs-mcp` | Internet access (fetches HIP docs from AMD website) |
| `rocminfo-mcp` | `rocminfo` (from ROCm, or set `INTELLIKIT_ROCMINFO`) |

---

### uprof-MCP (`uprof_mcp`)

**Purpose:** MCP server for profiling CPU code via AMD uProf.

#### System Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.10 | |
| AMD uProf | >= 5.1 (tested up to 6.0) | Must be installed separately; provides `AMDuProfCLI` |
| x86 CPU | — | uProf only supports x86 architecture |

The profiler executable defaults to `/opt/AMDuProf_<version>/bin/AMDuProfCLI` (e.g., `/opt/AMDuProf_5.1-701/bin/AMDuProfCLI`). A custom path can be provided via the `uprof` constructor argument or the `INTELLIKIT_UPROF_CLI` environment variable.

AMD uProf is available for download from [AMD's developer portal](https://www.amd.com/en/developer/uprof.html).

---

## External Dependency Details

### kernelDB System Dependencies

kernelDB ([`AMDResearch/kerneldb`](https://github.com/AMDResearch/kerneldb), pinned to commit [`6e00939`](https://github.com/AMDResearch/kerneldb/commit/6e0093972be276a51ed2bea963c756caa9324325)) is pulled automatically from GitHub when building Accordo (Python `pip` dependency) or Nexus (CMake CPM dependency). However, kernelDB's own build requires several **system libraries** that must be present on the host:

| Library | Ubuntu/Debian package | RHEL/CentOS package | Notes |
|---|---|---|---|
| `libdwarf` | `libdwarf-dev` | `libdwarf-devel` | DWARF debug info parsing; **required** (fatal CMake error if missing) |
| `libelf` | `libelf-dev` | `elfutils-libelf-devel` | ELF binary parsing; linked as `elf` |
| `libzstd` | `libzstd-dev` | `libzstd-devel` | Zstandard compression; linked as `zstd` |
| `amd_comgr` | from ROCm | from ROCm | AMD Code Object Manager; part of ROCm installation |
| `libhsa-runtime64` | from ROCm (`hsa-rocr-dev`) | from ROCm (`hsa-rocr-dev`) | HSA runtime; part of ROCm installation |

Install the system packages before building Accordo or Nexus:

```bash
# Ubuntu/Debian
sudo apt install libdwarf-dev libelf-dev libzstd-dev

# RHEL/CentOS
sudo dnf install libdwarf-devel elfutils-libelf-devel libzstd-devel
```

`amd_comgr` and `libhsa-runtime64` are provided by the ROCm installation and do not require separate installation.

### ROCm / HSA Runtime

- **Source:** https://rocm.docs.amd.com/
- **Used by:** Accordo, Nexus (link against `libhsa-runtime64.so`), Metrix, Linex (invoke `rocprofv3`), ROCm-MCP (`hipcc`, `rocminfo`)
- **Installation:** Follow [ROCm installation guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html). Default path: `/opt/rocm`.

After installation, ensure the ROCm binaries are in your `PATH`:

```bash
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### AMD uProf

- **Source:** https://www.amd.com/en/developer/uprof.html
- **Used by:** uprof-MCP
- **Installation:** Download and install the `.deb` or `.rpm` package from the AMD developer portal:

```bash
# Debian/Ubuntu
sudo dpkg -i AMDuProf_<version>_amd64.deb

# RHEL/CentOS
sudo rpm -ivh AMDuProf_<version>_x86_64.rpm
```

Default install path: `/opt/AMDuProf_<version>/`.

---

## Dependency Graph

```
intellikit (root metapackage)
├── accordo
│   ├── Python: numpy, mcp[cli], kerneldb (auto-fetched from GitHub)
│   └── System: ROCm >= 6.0, CMake >= 3.22, libdwarf, libelf, libzstd
├── linex
│   ├── Python: mcp[cli]
│   └── System: rocprofv3 (ROCm 7.0+)
├── metrix
│   ├── Python: pandas>=1.5.0, mcp[cli]
│   └── System: rocprofv3 (ROCm 6.x+), rocminfo
├── nexus
│   ├── Python: mcp[cli]
│   └── System: ROCm >= 6.0, LLVM (from ROCm), CMake >= 3.22, libdwarf, libelf, libzstd
├── rocm_mcp
│   ├── Python: mcp[cli]>=1.21.0, beautifulsoup4>=4.12.0, httpx>=0.28.1
│   └── System: hipcc (ROCm), rocminfo (ROCm)
└── uprof_mcp
    ├── Python: mcp[cli]>=1.21.0
    └── System: AMD uProf (AMDuProfCLI)
```
