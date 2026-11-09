---
myst:
    html_meta:
        "description": "Build and install Omniprobe, the tool IntelliKit's omniprobe package drives, and check that IntelliKit can use it."
        "keywords": "IntelliKit, Omniprobe, install, build, ROCm, LLVM pass plugin, PATH"
---

# Install Omniprobe

IntelliKit's `omniprobe` package drives [Omniprobe](https://github.com/AMDResearch/omniprobe), a
separate tool that is built from source. Installing IntelliKit does not install Omniprobe. This page
builds it at the commit IntelliKit is tested with.

## Check whether you already have it

```bash
omniprobe-ik check
```

It reports the `omniprobe` executable, its instrumentation plugins, its Python API, the GPU
architecture, and the ROCm Omniprobe was built for compared with the `hipcc` on `PATH`. It exits 0
when Omniprobe is usable. The only setting IntelliKit needs is `<prefix>/omniprobe/bin` of an
Omniprobe install on `PATH`; the plugins and the Python API are found from there.

## Prerequisites

- ROCm 7.0 or later, including the LLVM development package (`rocm-llvm-dev` on Ubuntu). Omniprobe
  is tested with ROCm 7.0, 7.1 and 7.2; IntelliKit's `omniprobe` package is tested with ROCm 7.2.0
  on gfx90a.
- CMake 3.15 or later, Python 3.10 or later, git.
- System packages (Ubuntu 24.04 names, from Omniprobe's build documentation):

  ```bash
  sudo apt-get install -y git build-essential cmake ninja-build clang lld \
      libzstd-dev libomp-dev libdwarf-dev python3-dev python3-venv rocm-llvm-dev
  ```

  The configure step stops if the libdwarf headers are missing.
- The `omniprobe` launcher is a script run by the first `python3` on `PATH`, and it imports
  `pyfiglet`. Install `pyfiglet` for that `python3` (the command below does, when run with the same
  `PATH`; outside a virtualenv add `--user`, or on Ubuntu use `sudo apt-get install python3-pyfiglet`).
  In an MCP setup this is the `python3` found through the server's `PATH`.

## Get the source

IntelliKit is tested with Omniprobe commit `dbb53ed`:

```bash
git clone --recurse-submodules https://github.com/AMDResearch/omniprobe.git
cd omniprobe
git checkout dbb53ed && git submodule update --init --recursive
python3 -m pip install -r omniprobe/requirements.txt
```

## Build and install

Set `ROCM_PATH` in the environment to the ROCm you will compile your applications with, and the
GPU architecture to build for (`rocminfo | grep -m1 -o 'gfx[0-9a-z]*'` prints it). Omniprobe's
CMake reads `ROCM_PATH` from the environment: passed only as `-DROCM_PATH=...` it is ignored and
`/opt/rocm` is used.

```bash
export ROCM_PATH=/opt/rocm-7.2.0
cmake -B build -DROCM_PATH=$ROCM_PATH -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_INSTALL_PREFIX=$HOME/.local/omniprobe
cmake --build build -j16
cmake --install build
```

Measured on a 128-core node with `-j16`, one architecture, prerequisites installed: configure 7 s,
build 20 s, install under 1 s.

## Use it

```bash
export PATH=$HOME/.local/omniprobe/omniprobe/bin:$PATH
omniprobe-ik check
```

For an MCP client, put the same directory in the `PATH` of the `omniprobe-mcp` server's `env`
configuration (see [Set up MCP](../how-to/mcp-setup.md)); MCP clients usually start servers
with a minimal environment.

## One build per ROCm

Omniprobe's instrumentation plugins are LLVM pass plugins built with `$ROCM_PATH/llvm`, and they
are loaded by the `hipcc` that compiles your application. LLVM gives pass plugins no stable
interface across versions, so build Omniprobe with the ROCm you compile with; `omniprobe-ik check`
warns when the two differ. Measured on gfx90a: a trivial kernel compiled by ROCm 7.1.0's and
6.4.1's `hipcc` with plugins built for 7.2.0 gave the same Omniprobe report as with 7.2.0's —
a mismatch is not certain to fail, and not certain to work.

At run time, Omniprobe (22c4d95 and later) picks a ROCm installation by itself (`ROCM_PATH` if it has
`llvm/bin/llvm-objdump`, otherwise the ROCm its libraries link against). Omniprobe builds older than
`149cf2c` need `ROCM_PATH` set to that ROCm, or they report nothing.

## Triton

Instrumenting Triton kernels needs a second plugin set built against Triton's LLVM, which takes
15-90 minutes to build. See Omniprobe's
[Triton instrumentation guide](https://github.com/AMDResearch/omniprobe/blob/main/docs/triton-instrumentation.md).
IntelliKit's `omniprobe` package is not tested with Triton.
