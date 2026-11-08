# IntelliKit compatibility matrix

Use the following matrix to view the compatibility and system requirements for IntelliKit:

| Requirement | Required by | Notes |
|-------------|-------------|-------|
| Python| Metrix, Linex, Omniprobe, Nexus, Accordo, Kerncap, ROCm MCP, uProf MCP | 3.10 or later. |
| OS |  Metrix, Linex, Omniprobe, Nexus, Accordo, Kerncap, ROCm MCP, uProf MCP | Ubuntu 22.04 and 24.04. |
| ROCm  | Metrix, Linex, Nexus, Accordo, Kerncap, ROCm MCP | 7.2.x or later. Required for GPU profiling and kernel analysis. Not needed for host-only tools. |
| GPU | Metrix, Linex, Omniprobe, Nexus, Accordo, Kerncap, ROCm MCP | Both Instinct and Radeon GPUs are supported. Instinct MI300X, MI325X, and MI355X are recommended for full GPU functionality. |
| uProf | uProf MCP only | AMD uProf on x86. |
| Omniprobe | Omniprobe only | The [Omniprobe](https://github.com/AMDResearch/omniprobe) tool, `<prefix>/omniprobe/bin` on `PATH`; see [Install Omniprobe](getting-started/installing-omniprobe.md). Omniprobe needs ROCm 7.0 or later (tested 7.0-7.2); the IntelliKit package is tested with ROCm 7.2.0. It drives an existing installation and does not bundle or build it. |
| cmake, libdwarf-dev, libzstd-dev | Accordo, Nexus | Required for C++ build via KernelDB. |
