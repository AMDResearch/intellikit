<!--
MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# Nexus: HSA Packet Source Code Extractor

> [!IMPORTANT]  
> This project is intended for research purposes only and is provided by AMD Research and Advanced Development team. 
This is not a product. Use it at your own risk and discretion.
> 
Nexus is a custom tool that intercepts Heterogeneous System Architecture (HSA) packets, extracts the source code from them, and outputs it to a JSON file containing the assembly and the HIP code.


## Build

To build Nexus, use CMake:

```bash
cmake -B build\
    -DCMAKE_PREFIX_PATH=${ROCM_PATH}\
    -DLLVM_INSTALL_DIR=/opt/rocm/llvm\
    -DCMAKE_BUILD_TYPE=Debug

cmake --build build --parallel 16
```

This will build the Nexus library and its dependencies. You can adjust the `CMAKE_BUILD_TYPE` variable to change the build configuration (e.g., debug or release).

## Usage

### Options

* `NEXUS_LOG_LEVEL`: Verbosity level (0 = none, 1 = info, 2 = warning, 3 = error, 4 = detail)
* `NEXUS_OUTPUT_FILE`: Path to the JSON output file
* `NEXUS_EXTRA_SEARCH_PREFIX`: Additional search directories for HIP files with relative paths. Supports wildcards and is a colon-separated list.


### Example

To use Nexus, simply export the following environment variables and run your application:

```terminal
export HSA_TOOLS_LIB=build/lib/libnexus.so
export NEXUS_LOG_LEVEL=3
export NEXUS_OUTPUT_FILE=result.json

cd test/
hipcc vector_add.hip -g  -o vector_add
cd ..
./test/vector_add
```


Outputs:

<details><summary>Log</summary>
<p>

```terminal
./test/vector_add 
[DETAIL]: [src/nexus.cpp:775] Creating maestro singleton
[DETAIL]: [src/nexus.cpp:110] Saving current APIs.
[DETAIL]: [src/nexus.cpp:112] Hooking new APIs.
[DETAIL]: [src/nexus.cpp:114] Discovering agents.
[DETAIL]: [src/nexus.cpp:118] Agent Handle: 0xdafbe0 , Name: AMD EPYC 7763 64-Core Processor
[DETAIL]: [src/nexus.cpp:118] Agent Handle: 0xdb3e00 , Name: AMD EPYC 7763 64-Core Processor
[DETAIL]: [src/nexus.cpp:118] Agent Handle: 0xdb75e0 , Name: gfx90a
[DETAIL]: [src/nexus.cpp:118] Agent Handle: 0xdc10f0 , Name: gfx90a
[DETAIL]: [src/nexus.cpp:118] Agent Handle: 0xdc5af0 , Name: gfx90a
[DETAIL]: [src/nexus.cpp:118] Agent Handle: 0xdca4f0 , Name: gfx90a
[DETAIL]: [src/nexus.cpp:118] Agent Handle: 0xdcef20 , Name: gfx90a
[DETAIL]: [src/nexus.cpp:118] Agent Handle: 0xdd3970 , Name: gfx90a
[DETAIL]: [src/nexus.cpp:118] Agent Handle: 0xdd8370 , Name: gfx90a
[DETAIL]: [src/nexus.cpp:118] Agent Handle: 0xddcd90 , Name: gfx90a
[INFO]: [src/nexus.cpp:130] NEXUS_PIPE_NAME is not set. Set it to communicate with driver script.
[DETAIL]: [src/nexus.hpp:102] Agent: 0xdafbe0, Name: AMD EPYC 7763 64-Core Processor, Type: CPU
[DETAIL]: [src/nexus.hpp:104] Memory Pools:
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 540392173568, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 540392173568, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 540392173568, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 540392173568, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:112] Memory Regions:
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:102] Agent: 0xdb3e00, Name: AMD EPYC 7763 64-Core Processor, Type: CPU
[DETAIL]: [src/nexus.hpp:104] Memory Pools:
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 1082260680704, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 1082260680704, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 1082260680704, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 1082260680704, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:112] Memory Regions:
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:102] Agent: 0xdb75e0, Name: gfx90a, Type: GPU
[DETAIL]: [src/nexus.hpp:104] Memory Pools:
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 65536, Fine-Grained: false, Coarse-Grained: true
[DETAIL]: [src/nexus.hpp:112] Memory Regions:
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 68702699520, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 68702699520, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 68702699520, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 65536, Global: false, Kernarg: false, Local: true
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:102] Agent: 0xdc10f0, Name: gfx90a, Type: GPU
[DETAIL]: [src/nexus.hpp:104] Memory Pools:
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 65536, Fine-Grained: false, Coarse-Grained: true
[DETAIL]: [src/nexus.hpp:112] Memory Regions:
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:102] Agent: 0xdc5af0, Name: gfx90a, Type: GPU
[DETAIL]: [src/nexus.hpp:104] Memory Pools:
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 65536, Fine-Grained: false, Coarse-Grained: true
[DETAIL]: [src/nexus.hpp:112] Memory Regions:
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:102] Agent: 0xdca4f0, Name: gfx90a, Type: GPU
[DETAIL]: [src/nexus.hpp:104] Memory Pools:
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 65536, Fine-Grained: false, Coarse-Grained: true
[DETAIL]: [src/nexus.hpp:112] Memory Regions:
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:102] Agent: 0xdcef20, Name: gfx90a, Type: GPU
[DETAIL]: [src/nexus.hpp:104] Memory Pools:
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 65536, Fine-Grained: false, Coarse-Grained: true
[DETAIL]: [src/nexus.hpp:112] Memory Regions:
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:102] Agent: 0xdd3970, Name: gfx90a, Type: GPU
[DETAIL]: [src/nexus.hpp:104] Memory Pools:
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 65536, Fine-Grained: false, Coarse-Grained: true
[DETAIL]: [src/nexus.hpp:112] Memory Regions:
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:102] Agent: 0xdd8370, Name: gfx90a, Type: GPU
[DETAIL]: [src/nexus.hpp:104] Memory Pools:
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 65536, Fine-Grained: false, Coarse-Grained: true
[DETAIL]: [src/nexus.hpp:112] Memory Regions:
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:102] Agent: 0xddcd90, Name: gfx90a, Type: GPU
[DETAIL]: [src/nexus.hpp:104] Memory Pools:
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 68702699520, Fine-Grained: true, Coarse-Grained: false
[DETAIL]: [src/nexus.hpp:109]   - Pool Size: 65536, Fine-Grained: false, Coarse-Grained: true
[DETAIL]: [src/nexus.hpp:112] Memory Regions:
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 540392173568, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[DETAIL]: [src/nexus.hpp:118]   - Region Size: 1082260680704, Global: true, Kernarg: false, Local: false
[INFO]: [src/nexus.cpp:151] Found 0 kernels
[DETAIL]: [src/nexus.cpp:780] Creating maestro singleton completed
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe96cd00000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe96cb00000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe96c900000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe96c700000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe96c500000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe96c300000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe96c100000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe166e00000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe166c00000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe166a00000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe166800000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe166600000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe166400000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe166200000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe166000000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe165e00000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 448 bytes at 0x7fe96ec6e000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 4096 bytes at 0x7fe165a00000
[DETAIL]: [src/nexus.cpp:727] Creating nexus queue
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1048576 bytes at 0x7fe164c00000
[DETAIL]: [src/nexus.cpp:481] Creating a code object reader from memory 0x1489c10 (37568 bytes) (filename: [heap])
Adding [heap]
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_streamOpsWait.kd (demangled: __amd_rocclr_streamOpsWait.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x15434f0
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_copyBufferAligned.kd (demangled: __amd_rocclr_copyBufferAligned.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x14e3400
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_copyBufferRect.kd (demangled: __amd_rocclr_copyBufferRect.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x14a1770
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_copyBufferRectAligned.kd (demangled: __amd_rocclr_copyBufferRectAligned.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x14a18d0
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_copyImage.kd (demangled: __amd_rocclr_copyImage.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x155db70
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_fillBufferAligned.kd (demangled: __amd_rocclr_fillBufferAligned.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x1414ff0
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_copyImage1DA.kd (demangled: __amd_rocclr_copyImage1DA.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x155dd20
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_copyImageToBuffer.kd (demangled: __amd_rocclr_copyImageToBuffer.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x1504310
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_initHeap.kd (demangled: __amd_rocclr_initHeap.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x15436a0
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_streamOpsWrite.kd (demangled: __amd_rocclr_streamOpsWrite.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x15044c0
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_copyBuffer.kd (demangled: __amd_rocclr_copyBuffer.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x12ed420
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_fillBufferAligned2D.kd (demangled: __amd_rocclr_fillBufferAligned2D.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x1603130
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_copyBufferToImage.kd (demangled: __amd_rocclr_copyBufferToImage.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x1504160
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_fillImage.kd (demangled: __amd_rocclr_fillImage.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x155d9c0
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel __amd_rocclr_gwsInit.kd (demangled: __amd_rocclr_gwsInit.kd)
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x1543900
[DETAIL]: [src/nexus.cpp:610] Executing packet: HSA_PACKET_TYPE_BARRIER_AND
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 4096 bytes at 0x7fe165a01000
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 1052672 bytes at 0x7fe164800000
[DETAIL]: [src/nexus.cpp:610] Executing packet: HSA_PACKET_TYPE_BARRIER_AND
[DETAIL]: [src/nexus.cpp:702] HSA Allocated 4096 bytes at 0x7fe165a02000
[DETAIL]: [src/nexus.cpp:481] Creating a code object reader from memory 0x20b000 (6896 bytes) (filename: /work1/amd/muhaawad/git/amd/audacious/maestro/external/nexus/test/vector_add)
Adding /work1/amd/muhaawad/git/amd/audacious/maestro/external/nexus/test/vector_add
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel _ZN6thrust11hip_rocprim14__parallel_for6kernelILj256ENS0_20__uninitialized_fill7functorINS_10device_ptrIfEEfEEmLj1EEEvT0_T1_S9_.kd (demangled: void thrust::hip_rocprim::__parallel_for::kernel<256u, thrust::hip_rocprim::__uninitialized_fill::functor<thrust::device_ptr<float>, float>, unsigned long, 1u>(thrust::hip_rocprim::__uninitialized_fill::functor<thrust::device_ptr<float>, float>, unsigned long, unsigned long) [clone .kd])
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0xe7e470
[DETAIL]: [src/nexus.cpp:507] Looking up the kernel _Z10vector_addPKfS0_Pfi.kd (demangled: vector_add(float const*, float const*, float*, int) [clone .kd])
[DETAIL]: [src/nexus.cpp:537] Looking up the symbol 0x1529840
[DETAIL]: [src/nexus.cpp:610] Executing packet: Dispatch Packet
        Kernel name: void thrust::hip_rocprim::__parallel_for::kernel<256u, thrust::hip_rocprim::__uninitialized_fill::functor<thrust::device_ptr<float>, float>, unsigned long, 1u>(thrust::hip_rocprim::__uninitialized_fill::functor<thrust::device_ptr<float>, float>, unsigned long, unsigned long)
        Release Scope: HSA_FENCE_SCOPE_AGENT
        Acquire Scope: HSA_FENCE_SCOPE_AGENT
        setup: 0x3
        workgroup_size_x: 0x100
        workgroup_size_y: 0x1
        workgroup_size_z: 0x1
        grid_size_x: 0x400
        grid_size_y: 0x1
        grid_size_z: 0x1
        private_segment_size: 0x0
        group_segment_size: 0x0
        kernel_object: 0x7fe96ec28d00
        kernarg_address: 0x7fe164c00000
        completion_signal: 0x0
[DETAIL]: [src/nexus.cpp:631] Dumping the kernels at: result.json
[DETAIL]: [src/nexus.cpp:678] Dumped kernel data to: result.json
[DETAIL]: [src/nexus.cpp:683] Processed kernel: void thrust::hip_rocprim::__parallel_for::kernel<256u, thrust::hip_rocprim::__uninitialized_fill::functor<thrust::device_ptr<float>, float>, unsigned long, 1u>(thrust::hip_rocprim::__uninitialized_fill::functor<thrust::device_ptr<float>, float>, unsigned long, unsigned long)
[DETAIL]: [src/nexus.cpp:610] Executing packet: HSA_PACKET_TYPE_BARRIER_AND
[DETAIL]: [src/nexus.cpp:610] Executing packet: Dispatch Packet
        Kernel name: vector_add(float const*, float const*, float*, int)
        Release Scope: HSA_FENCE_SCOPE_AGENT
        Acquire Scope: HSA_FENCE_SCOPE_AGENT
        setup: 0x3
        workgroup_size_x: 0x100
        workgroup_size_y: 0x1
        workgroup_size_z: 0x1
        grid_size_x: 0x400
        grid_size_y: 0x1
        grid_size_z: 0x1
        private_segment_size: 0x0
        group_segment_size: 0x0
        kernel_object: 0x7fe96ec28cc0
        kernarg_address: 0x7fe164c00080
        completion_signal: 0x0
[DETAIL]: [src/nexus.cpp:631] Dumping the kernels at: result.json
[DETAIL]: [src/nexus.cpp:678] Dumped kernel data to: result.json
[DETAIL]: [src/nexus.cpp:683] Processed kernel: vector_add(float const*, float const*, float*, int)
[DETAIL]: [src/nexus.cpp:610] Executing packet: HSA_PACKET_TYPE_BARRIER_AND
[DETAIL]: [src/nexus.cpp:610] Executing packet: HSA_PACKET_TYPE_BARRIER_AND
Success
[DETAIL]: [src/nexus.cpp:764] Destroying nexus queue
```

</p>
</details> 


<details><summary>JSON Output</summary>
<p>


```console
cat result.json 
{
    "kernels": {
        "vector_add(float const*, float const*, float*, int)": {
            "assembly": [
                "global_load_dword v6, v[4:5], off                          // 000000001E68: DC508000 067F0004 ",
                "global_load_dword v7, v[2:3], off                          // 000000001E70: DC508000 077F0002 ",
                "global_store_dword v[0:1], v2, off                         // 000000001E8C: DC708000 007F0200 ",
                "s_load_dword s1, s[4:5], 0x18                              // 000000001E08: C0020042 00000018 ",
                "s_load_dwordx4 s[0:3], s[4:5], 0x0                         // 000000001E30: C00A0002 00000000 ",
                "s_load_dwordx2 s[6:7], s[4:5], 0x10                        // 000000001E38: C0060182 00000010 ",
                "s_load_dword s0, s[4:5], 0x2c                              // 000000001E00: C0020002 0000002C "
            ],
            "files": [
                "/work1/amd/muhaawad/git/amd/audacious/maestro/external/nexus/test/vector_add.hip",
                "/work1/amd/muhaawad/git/amd/audacious/maestro/external/nexus/test/vector_add.hip",
                "/work1/amd/muhaawad/git/amd/audacious/maestro/external/nexus/test/vector_add.hip",
                "/opt/rocm-6.3.2/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h",
                "/opt/rocm-6.3.2/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h",
                "/opt/rocm-6.3.2/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h",
                "/opt/rocm-6.3.2/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h"
            ],
            "hip": [
                "    c[idx] = a[idx] + b[idx];",
                "    c[idx] = a[idx] + b[idx];",
                "    c[idx] = a[idx] + b[idx];",
                "__DEVICE__ unsigned int __hip_get_thread_idx_x() { return __ockl_get_local_id(0); }",
                "__DEVICE__ unsigned int __hip_get_thread_idx_x() { return __ockl_get_local_id(0); }",
                "__DEVICE__ unsigned int __hip_get_thread_idx_x() { return __ockl_get_local_id(0); }",
                "__DEVICE__ unsigned int __hip_get_block_dim_x() { return __ockl_get_local_size(0); }"
            ],
            "lines": [
                11,
                11,
                11,
                264,
                264,
                264,
                274
            ],
            "signature": "vector_add(float const*, float const*, float*, int)"
        },
        "void thrust::hip_rocprim::__parallel_for::kernel<256u, thrust::hip_rocprim::__uninitialized_fill::functor<thrust::device_ptr<float>, float>, unsigned long, 1u>(thrust::hip_rocprim::__uninitialized_fill::functor<thrust::device_ptr<float>, float>, unsigned long, unsigned long)": {
            "assembly": [
                "s_load_dwordx2 s[2:3], s[4:5], 0x0                         // 000000002354: C0060082 00000000 ",
                "s_load_dword s6, s[4:5], 0x8                               // 00000000235C: C0020182 00000008 ",
                "flat_store_dword v[0:1], v2                                // 000000002388: DC700000 00000200 ",
                "s_load_dwordx4 s[8:11], s[4:5], 0x10                       // 000000002300: C00A0202 00000010 "
            ],
            "files": [
                "/opt/rocm-6.3.2/lib/llvm/bin/../../../include/thrust/system/hip/detail/parallel_for.h",
                "/opt/rocm-6.3.2/lib/llvm/bin/../../../include/thrust/system/hip/detail/parallel_for.h",
                "/opt/rocm-6.3.2/lib/llvm/bin/../../../include/thrust/system/hip/detail/uninitialized_fill.h",
                "/opt/rocm-6.3.2/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h"
            ],
            "hip": [
                "",
                "",
                "            ::new(static_cast<void*>(&out)) value_type(value);",
                "__DEVICE__ unsigned int __hip_get_block_idx_x() { return __ockl_get_group_id(0); }"
            ],
            "lines": [
                4294967295,
                4294967295,
                64,
                269
            ],
            "signature": "void thrust::hip_rocprim::__parallel_for::kernel<256u, thrust::hip_rocprim::__uninitialized_fill::functor<thrust::device_ptr<float>, float>, unsigned long, 1u>(thrust::hip_rocprim::__uninitialized_fill::functor<thrust::device_ptr<float>, float>, unsigned long, unsigned long)"
        }
    }
}
```

</p>
</details> 
