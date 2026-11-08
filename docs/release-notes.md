# IntelliKit release notes

This topic summarizes the features included in each IntelliKit release.

## Unreleased

- **Omniprobe** — new tool that attributes uncoalesced global accesses and LDS bank conflicts to source lines, and reports basic-block execution counts. It reports excess cache lines and LDS bank conflicts per access, ranked worst first, and names any dispatched kernel that ran without instrumentation. Available through the Python API, the `omniprobe-ik` CLI, the `omniprobe-mcp` server, and an agent skill.

## IntelliKit 0.1.1

- **ROCm 10 support** — Nexus, Accordo and Kerncap now hook `hsa_amd_queue_create`, the entry point HIP creates its compute queue through on ROCm 10. Previously all three attached to the runtime successfully and then captured no kernels, reporting no error while doing so. ROCm 7.2.x is unaffected and continues to work.
- **Side-by-side ROCm installs** — `ROCM_PATH` is now honoured when locating the HSA runtime. Previously the build always took headers from `/opt/rocm`, so building against another ROCm tree produced a mixed build that ran but could not see newer APIs.
- **Container runtime** — CI builds and runs its container with Docker as well as Apptainer, and no longer requires Apptainer on the runner.
- **Metrix and Linex** — verified on ROCm 10. Neither uses the queue interception path, so no changes were needed.

Validated on MI355X (gfx950).

## IntelliKit 0.1.0

First release of IntelliKit — agent-first tooling that makes GPU kernel profiling, inspection, and validation programmatically accessible to both developers and LLM agents.

- **Metrix** — human-readable GPU counter profiling with bottleneck classification. Supports CDNA2/3/4 and RDNA2/3.
- **Linex** — source-line stall analysis, maps cycle counts and ISA to individual source lines.
- **Nexus** — kernel disassembly, register pressure analysis, and dispatch inspection via KernelDB.
- **Accordo** — side-by-side correctness validation across dtypes and tolerances.
- **Kerncap** — GPU kernel dispatch capture and standalone reproducer generation.
- **ROCm MCP** — MCP servers for HIP compilation, amd-smi, and rocminfo.
- **uProf MCP** — host-side CPU hotspot analysis via AMD uProf.
- **MCP Servers** — every tool ships with its own MCP server for structured LLM agent integration.
- **Tool Skills** — every tool ships with an installable SKILL.md playbook.
