# Set up MCP

IntelliKit provides several MCP servers, which allow LLM agents to compile HIP code, profile applications, access documentation, and query GPU hardware. This topic explains how to configure these servers.

## Prerequisites

- [IntelliKit](..\getting-started\installation.md)
- `uv` (recommended) or `pip`
- [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) for GPU-related MCP servers
- AMD uProf for `uprof-profiler-mcp`

## Full configuration

To configure MCP servers using `uv` and a cloned IntelliKit repository, set up your MCP client to point to each package directory. Here's an example configuration:

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
    }
  }
}
```

Replace `/path/to/intellikit` with the actual path to your cloned IntelliKit repository.

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

| Server | Package | What it does |
|--------|---------|--------------|
| `metrix-mcp` | `metrix` | Profiles GPU applications and generates human-readable performance metrics|
| `kerncap-mcp` | `kerncap` | Extracts and isolates GPU kernels|
| `hip-compiler-mcp` | `rocm_mcp` | Compiles HIP C/C++ code|
| `hip-docs-mcp` | `rocm_mcp` | Accesses HIP documentation|
| `rocminfo-mcp` | `rocm_mcp` | Queries GPU hardware topology|
| `uprof-profiler-mcp` | `uprof_mcp` | Profiles CPU hotspots using AMD uProf|

## Agent skills

IntelliKit provides installable `SKILL.md` playbooks for Kerncap, Metrix, Linex, Nexus, and Accordo:

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash
```

Target options: `--target cursor` | `claude` | `codex` | `agents` | `github`

See [Install IntelliKit](../getting-started/installation.md) for more details about the skills script and usage.
