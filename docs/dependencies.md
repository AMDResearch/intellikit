# IntelliKit Dependencies

This document describes all dependencies required by each tool in the IntelliKit monorepo.

---

## Overview

IntelliKit is a collection of standalone tools. Each tool can be installed and used independently. Dependencies fall into three categories:

- **Python packages** – installed automatically via `pip`/`uv`
- **System tools** – ROCm utilities, compilers, or external programs that must be present on the host
- **C++ build dependencies** – libraries fetched and compiled during the CMake build step (transparent to users who install via `pip`)

---

## Shared Requirements

All tools require:

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.10 (root metapackage) | Individual tools may support older versions (see below) |
| AMD ROCm | >= 6.0 | Required for GPU tools (Metrix, Linex, Nexus, Accordo) |
| AMD GPU | MI300+ | Required for GPU profiling tools |

---

## Tool-by-Tool Dependencies

### Accordo

**Purpose:** Automated correctness validation for GPU kernel optimizations.

#### Python Dependencies

| Package | Version | Source |
|---|---|---|
| `numpy` | any | [PyPI](https://pypi.org/project/numpy/) |
| `mcp[cli]` | any | [PyPI](https://pypi.org/project/mcp/) |
| `kerneldb` | commit `6e0093972be276a51ed2bea963c756caa9324325` | [GitHub – AMDResearch/kerneldb](https://github.com/AMDResearch/kerneldb) |

`kerneldb` is fetched automatically from GitHub during `pip install`. It is a private AMD research dependency that provides kernel signature extraction utilities.

#### System Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.8 | |
| ROCm toolchain | >= 6.0 | Provides `hipcc` and HSA runtime |
| CMake | >= 3.22 | Required to build the C++ library at runtime |
| C++23-capable compiler | (via ROCm LLVM) | Typically `clang++` from `/opt/rocm/llvm` |

#### C++ Build Dependencies (fetched automatically via CMake FetchContent / CPM)

These are downloaded and compiled automatically when Accordo's C++ library is built:

| Library | Version | Source |
|---|---|---|
| `spdlog` | v1.11.0 | [GitHub – gabime/spdlog](https://github.com/gabime/spdlog) |
| `fmt` | 7.1.3 | [GitHub – fmtlib/fmt](https://github.com/fmtlib/fmt) |
| `nlohmann/json` | 3.11.3 | [GitHub – nlohmann/json](https://github.com/nlohmann/json) |
| HSA Runtime (`hsa-runtime64`) | from ROCm | `/opt/rocm/lib/libhsa-runtime64.so` |

#### Runtime System Library

| Library | Notes |
|---|---|
| `libhsa-runtime64.so` | From ROCm installation (`/opt/rocm/lib/`) |

---

### Linex

**Purpose:** Source-level GPU performance profiling – maps GPU cycles to source code lines.

#### Python Dependencies

| Package | Version | Source |
|---|---|---|
| `mcp[cli]` | any | [PyPI](https://pypi.org/project/mcp/) |

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

#### Python Dependencies

| Package | Version | Source |
|---|---|---|
| `pandas` | >= 1.5.0 | [PyPI](https://pypi.org/project/pandas/) |
| `mcp[cli]` | any | [PyPI](https://pypi.org/project/mcp/) |

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

#### Python Dependencies

| Package | Version | Source |
|---|---|---|
| `mcp[cli]` | any | [PyPI](https://pypi.org/project/mcp/) |

#### System Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.8 | |
| ROCm | >= 6.0 | Provides HSA runtime and LLVM |
| CMake | >= 3.22 | Required to build the C++ shared library |
| LLVM | from ROCm (`/opt/rocm/llvm`) | Set via `LLVM_INSTALL_DIR` CMake variable |
| C++23-capable compiler | (via ROCm LLVM) | Typically `clang++` from `/opt/rocm/llvm` |

#### C++ Build Dependencies (fetched automatically via CMake FetchContent / CPM)

| Library | Version | Source |
|---|---|---|
| `spdlog` | v1.11.0 | [GitHub – gabime/spdlog](https://github.com/gabime/spdlog) |
| `fmt` | 7.1.3 | [GitHub – fmtlib/fmt](https://github.com/fmtlib/fmt) |
| `nlohmann/json` | 3.11.3 | [GitHub – nlohmann/json](https://github.com/nlohmann/json) |
| `kerneldb` | commit `6e0093972be276a51ed2bea963c756caa9324325` | [GitHub – AMDResearch/kerneldb](https://github.com/AMDResearch/kerneldb) |
| HSA Runtime (`hsa-runtime64`) | from ROCm | `/opt/rocm/lib/libhsa-runtime64.so` |

The C++ shared library (`libnexus.so`) is built during `pip install` via scikit-build-core and CMake. The `kerneldb` shared library (`libkernelDB64.so`) is bundled into the Python package at install time.

#### Runtime System Library

| Library | Notes |
|---|---|
| `libhsa-runtime64.so` | From ROCm installation (`/opt/rocm/lib/`) |

---

### ROCm-MCP (`rocm_mcp`)

**Purpose:** Collection of MCP servers exposing ROCm tools to LLMs (HIP compiler, HIP docs, system info).

#### Python Dependencies

| Package | Version | Source |
|---|---|---|
| `mcp[cli]` | >= 1.21.0 | [PyPI](https://pypi.org/project/mcp/) |
| `beautifulsoup4` | >= 4.12.0 | [PyPI](https://pypi.org/project/beautifulsoup4/) |
| `httpx` | >= 0.28.1 | [PyPI](https://pypi.org/project/httpx/) |

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

#### Optional Dev/Example Dependencies

| Package | Notes |
|---|---|
| `langchain`, `langchain-mcp-adapters`, `langchain-openai` | For running LangChain-based examples |
| `python-dotenv` | For example scripts using `.env` files |

---

### uprof-MCP (`uprof_mcp`)

**Purpose:** MCP server for profiling CPU code via AMD uProf.

#### Python Dependencies

| Package | Version | Source |
|---|---|---|
| `mcp[cli]` | >= 1.21.0 | [PyPI](https://pypi.org/project/mcp/) |

#### System Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.10 | |
| AMD uProf | >= 5.1 (tested up to 6.0) | Must be installed separately; provides `AMDuProfCLI` |
| x86 CPU | — | uProf only supports x86 architecture |

The profiler executable defaults to `/opt/AMDuProf_<version>/bin/AMDuProfCLI` (e.g., `/opt/AMDuProf_5.1-701/bin/AMDuProfCLI`). A custom path can be provided via the `uprof` constructor argument or the `INTELLIKIT_UPROF_CLI` environment variable.

AMD uProf is available for download from [AMD's developer portal](https://www.amd.com/en/developer/uprof.html).

#### Optional Dev/Example Dependencies

| Package | Notes |
|---|---|
| `langchain`, `langchain-mcp-adapters`, `langchain-openai` | For running LangChain-based examples |
| `python-dotenv` | For example scripts using `.env` files |

---

## External / Third-Party Dependency Details

### kernelDB

- **Repository:** https://github.com/AMDResearch/kerneldb
- **Pinned commit:** `6e0093972be276a51ed2bea963c756caa9324325`
- **Used by:** Accordo (Python package, installed via `pip`), Nexus (C++ build, via CPM)
- **Purpose:** Provides kernel binary parsing and signature extraction utilities for AMD GPU kernels.
- **Installation:** Automatically installed as a Python dependency when installing Accordo. For Nexus, the C++ library is fetched and compiled by CMake during `pip install nexus`.

### ROCm / HSA Runtime

- **Source:** https://rocm.docs.amd.com/
- **Used by:** Accordo, Nexus (link against `libhsa-runtime64.so`), Metrix, Linex (invoke `rocprofv3`), ROCm-MCP (`hipcc`, `rocminfo`)
- **Installation:** Follow [ROCm installation guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html). Default path: `/opt/rocm`.

### AMD uProf

- **Source:** https://www.amd.com/en/developer/uprof.html
- **Used by:** uprof-MCP
- **Installation:** Download and install the `.deb` or `.rpm` package from the AMD developer portal. Default install path: `/opt/AMDuProf_<version>/`.

---

## Dependency Graph

```
intellikit (root metapackage)
├── accordo
│   ├── Python: numpy, mcp[cli], kerneldb (GitHub)
│   └── C++ (runtime build): spdlog, fmt, nlohmann/json, HSA runtime, CMake, HIP compiler
├── linex
│   ├── Python: mcp[cli]
│   └── System: rocprofv3 (ROCm 7.0+)
├── metrix
│   ├── Python: pandas>=1.5.0, mcp[cli]
│   └── System: rocprofv3 (ROCm 6.x+), rocminfo
├── nexus
│   ├── Python: mcp[cli]
│   └── C++ (build-time): spdlog, fmt, nlohmann/json, kerneldb (GitHub), HSA runtime, LLVM (from ROCm), CMake
├── rocm_mcp
│   ├── Python: mcp[cli]>=1.21.0, beautifulsoup4>=4.12.0, httpx>=0.28.1
│   └── System: hipcc (ROCm), rocminfo (ROCm)
└── uprof_mcp
    ├── Python: mcp[cli]>=1.21.0
    └── System: AMD uProf (AMDuProfCLI)
```

---

## Installation Notes

### ROCm Installation

Most tools require ROCm. Install following the [official ROCm installation guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html).

After installation, ensure the ROCm binaries are in your `PATH`:

```bash
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### AMD uProf Installation

Download from [AMD uProf](https://www.amd.com/en/developer/uprof.html) and install the package:

```bash
# Debian/Ubuntu
sudo dpkg -i AMDuProf_<version>_amd64.deb

# RHEL/CentOS
sudo rpm -ivh AMDuProf_<version>_x86_64.rpm
```

### kernelDB

`kerneldb` is installed automatically when you install `accordo`:

```bash
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=accordo"
```

It is fetched from a pinned commit on GitHub:
```
git+https://github.com/AMDResearch/kerneldb.git@6e0093972be276a51ed2bea963c756caa9324325
```

For Nexus, `kerneldb` is fetched and built from source by CMake during `pip install nexus` (via CPM).

### Building C++ Components

Accordo and Nexus contain C++ code that is compiled automatically during `pip install`. You need:

- CMake >= 3.22
- A C++23-compatible compiler (provided by ROCm's LLVM: `/opt/rocm/llvm/bin/clang++`)
- ROCm installed at `/opt/rocm` (or set `ROCM_PATH`)
- For Nexus: LLVM from ROCm (set `LLVM_INSTALL_DIR=/opt/rocm/llvm`)

```bash
# Manual build for Nexus (optional):
cd nexus
cmake -B build \
    -DCMAKE_PREFIX_PATH=${ROCM_PATH:-/opt/rocm} \
    -DLLVM_INSTALL_DIR=/opt/rocm/llvm \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 16
```
