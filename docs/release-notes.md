# Release notes

This topic summarizes the features included in each IntelliKit release.

## IntelliKit 0.1.0

Initial release of IntelliKit, a modular toolkit for GPU kernel development, profiling, and validation.

- **Metrix** — human-readable GPU counter profiling with bottleneck classification. Supports CDNA2/3/4 and RDNA2/3.
- **Linex** — source-line stall analysis, maps cycle counts and ISA to individual source lines.
- **Nexus** — kernel disassembly, register pressure analysis, and dispatch inspection via KernelDB.
- **Accordo** — side-by-side correctness validation across dtypes and tolerances.
- **Kerncap** — GPU kernel dispatch capture and standalone reproducer generation.
- **ROCm MCP** — MCP servers for HIP compilation, amd-smi, and rocminfo.
- **uProf MCP** — host-side CPU hotspot analysis via AMD uProf.
- **MCP Servers** — every tool ships with its own MCP server for structured LLM agent integration.
- **Tool Skills** — every tool ships with an installable SKILL.md playbook.
