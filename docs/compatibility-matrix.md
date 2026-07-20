# IntelliKit compatibility matrix

Use the following matrix to view the compatibility and system requirements for IntelliKit:

| Requirement | Required by | Notes |
|-------------|-------------|-------|
| Python| Metrix, Linex, Nexus, Accordo, Kerncap, ROCm MCP, uProf MCP | 3.10 or later. |
| OS |  Metrix, Linex, Nexus, Accordo, Kerncap, ROCm MCP, uProf MCP | Ubuntu 22.04 and 24.04. |
| ROCm  | Metrix, Linex, Nexus, Accordo, Kerncap, ROCm MCP | 7.2.x. Required for GPU profiling and kernel analysis. Not needed for host-only tools. |
| GPU | Metrix, Linex, Nexus, Accordo, Kerncap, ROCm MCP | Both Instinct and Radeon GPUs are supported. Instinct MI300, 325, and 355 are recommended for full GPU functionality. |
| uProf | uProf MCP only | AMD uProf on x86. |
| cmake, libdwarf-dev, libzstd-dev | Accordo, Nexus | Required for C++ build via KernelDB. |
