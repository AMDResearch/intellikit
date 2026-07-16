# IntelliKit compatibility matrix

This topic lists the known version requirements for IntelliKit.

## Requirements

IntelliKit requires the following software and hardware.

| Requirement | Required by | Notes |
|-------------|-------------|-------|
| Python | All tools | 3.10 or later. |
| ROCm 7.0+ | Metrix, Linex, Nexus, Accordo, Kerncap, ROCm MCP | Required for GPU profiling and kernel analysis. Not needed for host-only tools. |
| GPU | Metrix, Linex, Nexus, Accordo, Kerncap, ROCm MCP | Both Instinct and RDNA GPUs are supported. Instinct MI300+ recommended for full GPU functionality. |
| uProf | `uprof_mcp` only | AMD uProf on x86. |
| cmake, libdwarf-dev, libzstd-dev | Accordo, Nexus | Required for C++ build via KernelDB. See the following section for details. |
