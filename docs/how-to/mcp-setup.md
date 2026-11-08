---
myst:
    html_meta:
        "description": "Configure IntelliKit MCP servers for LLM agents to compile HIP code, profile applications, access documentation, and query GPU hardware with ROCm."
        "keywords": "IntelliKit, MCP, Model Context Protocol, LLM, AI agent, HIP, ROCm, GPU, uProf, metrix-mcp"
---

# Set up MCP in IntelliKit

IntelliKit provides several MCP servers, which allow LLM agents to compile HIP code, profile applications, access documentation, and query GPU hardware. This topic explains how to configure these servers.

## Prerequisites

Before configuring MCP servers, ensure you have the following installed.

- [IntelliKit installation](../getting-started/installation.md)
- `uv` (recommended) or `pip`
- [ROCm installation](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/) for GPU-related MCP servers
- AMD uProf for `uprof-profiler-mcp`
- The [Omniprobe](https://github.com/AMDResearch/omniprobe) tool on `PATH`, with its Python API importable, for `omniprobe-mcp`

## Full configuration

To configure MCP servers using `uv` and a cloned IntelliKit repository, set up your MCP client to point to each package directory. The following is an example configuration:

```json
{
  "mcpServers": {
    "metrix-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/intellikit/metrix", "metrix-mcp"]
    },
    "kerncap-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/intellikit/kerncap", "kerncap-mcp"]
    },
    "hip-compiler-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/intellikit/rocm_mcp", "hip-compiler-mcp"]
    },
    "hip-docs-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/intellikit/rocm_mcp", "hip-docs-mcp"]
    },
    "rocminfo-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/intellikit/rocm_mcp", "rocminfo-mcp"]
    },
    "uprof-profiler-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/intellikit/uprof_mcp", "uprof-profiler-mcp"]
    },
    "omniprobe-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/intellikit/omniprobe", "omniprobe-mcp"],
      "env": {
        "PYTHONPATH": "/path/to/omniprobe-source",
        "ROCR_VISIBLE_DEVICES": "0"
      }
    }
  }
}
```

Replace `/path/to/intellikit` with the actual path to your cloned IntelliKit repository. For `omniprobe-mcp`, also set `PYTHONPATH` to an Omniprobe **source** checkout (its install tree does not include the Python API) and pick the GPU with `ROCR_VISIBLE_DEVICES`: MCP clients usually start servers with a minimal environment, so neither is inherited from your shell.

## Using pip-installed packages

If you installed IntelliKit using `pip` or the `install.sh` script, the MCP console script names are already added to your system's `PATH`. You can configure the MCP client with these simple references:

```json
{
  "mcpServers": {
    "metrix-mcp": {
      "command": "metrix-mcp"
    },
    "kerncap-mcp": {
      "command": "kerncap-mcp"
    },
    "hip-compiler-mcp": {
      "command": "hip-compiler-mcp"
    }
  }
}
```

## Available servers

IntelliKit provides the following MCP servers.

| Server | Package | What it does |
|--------|---------|--------------|
| `metrix-mcp` | `metrix` | Profiles GPU applications and generates human-readable performance metrics|
| `kerncap-mcp` | `kerncap` | Extracts and isolates GPU kernels|
| `hip-compiler-mcp` | `rocm_mcp` | Compiles HIP C/C++ code|
| `hip-docs-mcp` | `rocm_mcp` | Accesses HIP documentation|
| `rocminfo-mcp` | `rocm_mcp` | Queries GPU hardware topology|
| `uprof-profiler-mcp` | `uprof_mcp` | Profiles CPU hotspots using AMD uProf|
| `omniprobe-mcp` | `omniprobe` | Finds uncoalesced accesses and LDS bank conflicts at source level, and basic-block execution counts; exposes `get_instrumentation_flags`, `analyze_memory`, `analyze_basic_blocks`, and `list_analyzers`|

## Agent skills

IntelliKit provides installable `SKILL.md` playbooks for Kerncap, Metrix, Linex, Nexus, Omniprobe, and Accordo:

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash
```

Target options: `--target cursor` | `claude` | `codex` | `agents` | `github`

See [Install IntelliKit](../getting-started/installation.md) for more details about the skills script and usage.
