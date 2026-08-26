---
myst:
    html_meta:
        "description": "Install IntelliKit tools and agent skills from Git or using pip. Covers system requirements, quick install, individual packages, and install script options."
        "keywords": "IntelliKit, install, ROCm, AMD Instinct, GPU, pip, uv, Python, Kerncap, Metrix, Linex, Nexus, Accordo"
---

# Install IntelliKit

This topic describes how to install IntelliKit.

## Requirements

See [Compatibility matrix](../compatibility-matrix.md) for software and hardware requirements.

### System dependencies for Accordo and Nexus

Accordo and Nexus use C++ components from KernelDB, which are compiled during `pip install`.

To install the system dependencies for Accordo and Nexus:

```bash
# Debian or Ubuntu
sudo apt-get update && sudo apt-get install -y cmake libdwarf-dev libzstd-dev

# Fedora or RHEL
sudo dnf install -y cmake libdwarf-devel libzstd-devel
```

If these packages are missing, the installation script (`install/tools/install.sh`) will exit with an error listing the missing packages. For convenience, the
[IntelliKit Docker image](https://github.com/AMDResearch/intellikit/blob/main/docker/Dockerfile) already includes these dependencies.

## Quick install

You can quickly install all IntelliKit tools directly from Git:

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash
```

To set up agent skills (compatible with Cursor, Claude, and Codex):

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash
```

## Install individual packages

To install a single IntelliKit package from Git:

```bash
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=metrix"
# Also: accordo, kerncap, linex, nexus, rocm_mcp, uprof_mcp
```

## Editable install (development)

For development purposes, you can install packages in "editable" mode:

```bash
git clone https://github.com/AMDResearch/intellikit.git
cd intellikit
pip install -e ./metrix
pip install -e ./linex
# ...any subset
```

## Install script options

The install scripts provide options for customizing which tools are installed and how.

### Tools script (`install/tools/install.sh`)

The tools installation script provides flexible options to customize the process:

- **Default** `pip3`: the script checks that `pip`'s Python is 3.10+ before installing
- **Install subset only**: `--tools metrix,linex,nexus`
- **Custom** `pip`: `--pip-cmd pip3.12` or `--pip-cmd "python3.12 -m pip"`
- **Branch/tag**: `--ref my-branch`
- **Custom Git repository**: `--repo-url https://github.com/you/fork.git`
- **Preview changes**: `--dry-run`

For example, to install a subset of tools, you can pipe flags after `bash -s --`:

```bash
curl -sSL .../install/tools/install.sh | bash -s -- --tools metrix,linex
```

### Skills script (`install/skills/install.sh`)

The skills installation script offers options for customizing skill deployment:

- **Target location**: `--target cursor` | `claude` | `codex` | `agents` | `github` specifies where skills should be deployed
- **Global installation**: `--global`. For example, `~/.cursor/skills/` for Cursor
- **Preview changes**: `--dry-run`
