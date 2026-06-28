# What is IntelliKit?

IntelliKit is a collection of Python tools to help you analyze and improve the performance of AMD hardware. It works with ROCm to turn data like hardware counters and traces into easy-to-use Python APIs. For host-side CPU hotspot analysis, the toolkit includes `uprof_mcp`, which integrates AMD uProf. Additionally, IntelliKit supports LLM-style workflows by providing MCP servers and agent skills, enabling you to use installable `SKILL.md` playbooks with platforms such as Cursor, Claude, Codex, and GitHub Copilot.

| Tool | Role | Description |
|------|------|-------------|
| **[Kerncap](tools/kerncap.md)** | Isolate | Captures kernel dispatches and builds standalone reproducers for HIP and Triton code.|
| **[Metrix](tools/metrix.md)** | Profile | Translates hardware counter data into human-readable metrics like bandwidth, cache, and compute performance.|
| **[Linex](tools/linex.md)** | Profile | Maps GPU performance to your source code, and analyzes source-line timing and stall.|
| **[Nexus](tools/nexus.md)** | Inspect | Intercepts HSA packets to reveal what ran on the GPU, including assembly code and HIP source details.|
| **[Accordo](tools/accordo.md)** | Validate | Confirms that optimized kernels produce the same results as their reference implementations.|
| **[ROCm MCP](tools/rocm-mcp.md)** | MCP | Provides HIP compiler, documentation, and rocminfo servers for LLM agents.|
| **[uProf MCP](tools/uprof-mcp.md)** | CPU | Bridges AMD uProf for host-side CPU hotspot analysis.|
