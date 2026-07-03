# Set up skills

IntelliKit ships `SKILL.md` playbooks for five tools: Metrix, Accordo, Nexus, Linex, and Kerncap. These playbooks provide AI coding assistants with step-by-step instructions for profiling, inspecting, and validating GPU kernels.

## What are skills?

A skill is a markdown file (`SKILL.md`) that teaches an AI agent how to use a tool. When installed into a supported agent's skills directory, the agent can invoke the skill by name. Each skill includes:

- Conditions specifying when the skill should be triggered
- Step-by-step instructions on how to use the tool
- Expected inputs and outputs
- Error handling guidance

## Available skills

| Skill | Tool | Triggers on |
|-------|------|-------------|
| `metrix-profiling` | Metrix | Performance analysis, GPU metrics, bandwidth, cache hit rates |
| `test-kerncap` | Kerncap | Kernel extraction, reproducer validation, HIP/Triton isolation |
| `linex-profiling` | Linex | Source-line profiling, stall analysis, cycle mapping |
| `nexus-tracing` | Nexus | GPU execution inspection, HSA packet tracing, assembly extraction |
| `accordo-validation` | Accordo | Kernel correctness checking, output comparison |

## Quick install

Run the following command to install all five skills into `.agents/skills/` in the current directory (the default target):

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash
```

## Choosing a target

Use the `--target` flag to specify where skills are installed. The table below outlines available targets and their respective paths:

| Target | Project-level path | Global path (`--global`) |
|--------|--------------------|--------------------------|
| `agents` (default) | `.agents/skills/` | `~/.agents/skills/` |
| `cursor` | `.cursor/skills/` | `~/.cursor/skills/` |
| `claude` | `.claude/skills/` | `~/.claude/skills/` |
| `codex` | `.codex/skills/` | `~/.codex/skills/` |
| `github` | `.github/agents/skills/` | `~/.github/agents/skills/` |

### Cursor

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh \
  | bash -s -- --target cursor
```

Skills are installed in `.cursor/skills/{tool}/SKILL.md`. The Cursor agent automatically detects and uses installed skills.
### Claude Code

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh \
  | bash -s -- --target claude
```

Skills land in `.claude/skills/{tool}/SKILL.md`.

### Codex

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh \
  | bash -s -- --target codex
```

### GitHub Copilot

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh \
  | bash -s -- --target github
```

Skills are installed in `.github/agents/skills/{tool}/SKILL.md`.

## Global vs project-level

By default, skills are installed into the current directory (project-level). To make skills available across all projects, use the `--global` flag to install them into your home directory:

```bash
curl -sSL .../install/skills/install.sh | bash -s -- --target cursor --global
# Installs to ~/.cursor/skills/
```

## Other options

```bash
# Preview what would be installed
curl -sSL .../install/skills/install.sh | bash -s -- --dry-run

# Use a different branch or fork
curl -sSL .../install/skills/install.sh | bash -s -- --base-url https://raw.githubusercontent.com/you/fork/my-branch
```

## Installed file structure

After installation, each tool gets its own `SKILL.md` file:

```
.cursor/skills/          # or .claude/skills/, .agents/skills/, etc.
├── metrix/SKILL.md
├── kerncap/SKILL.md
├── linex/SKILL.md
├── nexus/SKILL.md
└── accordo/SKILL.md
```

Each file has a YAML frontmatter with `name` and `description`, followed by the full skill playbook.
