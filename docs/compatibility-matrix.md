# IntelliKit compatibility matrix

Use the following matrix to view the compatibility and system requirements for IntelliKit:

| Requirement | Required by | Notes |
|-------------|-------------|-------|
| Python| Metrix, Linex, Nexus, Accordo, Kerncap, ROCm MCP, uProf MCP | 3.10 or later. |
| ROCm  | Metrix, Linex, Nexus, Accordo, Kerncap, ROCm MCP | 7.0 or later. Required for GPU profiling and kernel analysis. Not needed for host-only tools. |
| GPU | Metrix, Linex, Nexus, Accordo, Kerncap, ROCm MCP | Both Instinct and Radeon GPUs are supported. Instinct MI300, 325, 355, and 350 are recommended for full GPU functionality. |
| uProf | uProf MCP only | AMD uProf on x86. |
| cmake, libdwarf-dev, libzstd-dev | Accordo, Nexus | Required for C++ build via KernelDB. |
