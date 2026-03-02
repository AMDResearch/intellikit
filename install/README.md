# IntelliKit Install Scripts

## Tools (`install/tools/install.sh`)

Installs all IntelliKit Python tools via pip.

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash
```

**Options:**

| Option | Description |
|--------|-------------|
| `--pip-cmd <cmd>` | Pip command to use (default: `pip`). Example: `--pip-cmd 'python3.12 -m pip'` |
| `--ref <ref>` | Git branch, tag, or commit to install from (default: `main`) |
| `--repo-url <url>` | Git repo URL (default: `https://github.com/AMDResearch/intellikit.git`) |
| `--dry-run` | Print pip commands without running them |
| `--help` | Show help and exit |

**Environment variables** (CLI options take precedence): `PIP_CMD`, `INTELLIKIT_REPO_URL`, `INTELLIKIT_REF`

**Examples:**

```bash
# Custom pip command (for a specific Python version)
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash -s -- --pip-cmd 'python3.12 -m pip'

# Install from a specific branch or tag
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash -s -- --ref v1.2.0

# Dry-run (preview commands without executing)
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash -s -- --dry-run

# From a clone
./install/tools/install.sh --pip-cmd pip3.12 --ref main --dry-run
```

## Agent Skills (`install/skills/install.sh`)

Downloads each tool's `SKILL.md` into a skills directory so AI agents can discover and use IntelliKit tools.

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash
```

**Options:**

| Option | Description |
|--------|-------------|
| `--target <name>` | Where to install: `agents` (default), `codex`, `cursor`, `claude` |
| `--global` | Use user-level dir (e.g. `~/.cursor/skills`) instead of project-level |
| `--base-url <url>` | Base URL for raw files (default: main branch) |
| `--dry-run` | Show what would be downloaded without making changes |
| `--help` | Show help and exit |

**Environment variables**: `INTELLIKIT_RAW_URL`

**Resulting layout:**

| Target | Project-level | Global |
|--------|--------------|--------|
| `agents` | `.agents/skills/` | `~/.agents/skills/` |
| `codex` | `.codex/skills/` | `~/.codex/skills/` |
| `cursor` | `.cursor/skills/` | `~/.cursor/skills/` |
| `claude` | `.claude/skills/` | `~/.claude/skills/` |

**Examples:**

```bash
# Install for Cursor (project-level)
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash -s -- --target cursor

# Install for Claude globally
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash -s -- --target claude --global

# Dry-run
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash -s -- --dry-run

# From a clone
./install/skills/install.sh --target cursor --global
```
