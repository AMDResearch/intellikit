# IntelliKit compatibility matrix

Use the following matrix to view the compatibility and system requirements for IntelliKit:

| Requirement | Required by | Notes |
|-------------|-------------|-------|
| Python| Metrix, Linex, Nexus, Accordo, Kerncap, ROCm MCP, uProf MCP | 3.10 or later. |
| OS |  Metrix, Linex, Nexus, Accordo, Kerncap, ROCm MCP, uProf MCP | Ubuntu 22.04 and 24.04. |
| ROCm  | Metrix, Linex, Nexus, Accordo, Kerncap, ROCm MCP | 7.2.x. Required for GPU profiling and kernel analysis. Not needed for host-only tools. See [ROCm 10](#rocm-10) below. |
| GPU | Metrix, Linex, Nexus, Accordo, Kerncap, ROCm MCP | Both Instinct and Radeon GPUs are supported. Instinct MI300X, MI325X, and MI355X are recommended for full GPU functionality. |
| uProf | uProf MCP only | AMD uProf on x86. |
| cmake, libdwarf-dev, libzstd-dev | Accordo, Nexus | Required for C++ build via KernelDB. |

## ROCm 10

ROCm 10 is not yet supported. Use ROCm 7.2.x.

ROCm 10 adds `hsa_amd_queue_create` and HIP creates its compute queue through
it rather than through `hsa_queue_create`. Nexus and Accordo hook only the
latter, so on ROCm 10 they attach successfully and then observe no dispatch
packets at all: profiling returns no kernels, and reports no error while doing
so. A fix is in progress.

Metrix, Linex, Kerncap and ROCm MCP have not been tested on ROCm 10 and are
neither known to work nor known to be broken there.
