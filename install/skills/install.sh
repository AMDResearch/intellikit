#!/usr/bin/env bash
# IntelliKit Agent Skills Installer
# Downloads each tool's SKILL.md into a skills dir. Use --target to pick agent (agents/codex/cursor/claude).
# Usage: curl -sSL .../install/skills/install.sh | bash -s -- [OPTIONS]
#    or: ./install.sh [OPTIONS]

set -e

BASE_URL="https://raw.githubusercontent.com/AMDResearch/intellikit/main"
TOOLS=(metrix accordo nexus)
DRY_RUN=false
GLOBAL=false
TARGET="agents"

print_usage() {
  echo "IntelliKit Agent Skills Installer"
  echo ""
  echo "Usage:"
  echo "  curl -sSL .../install/skills/install.sh | bash -s -- [OPTIONS]"
  echo "  ./install.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --target <name>   Where to install: agents (default), codex, cursor, claude"
  echo "                    agents -> .agents/skills or ~/.agents/skills"
  echo "                    codex  -> .codex/skills or ~/.codex/skills"
  echo "                    cursor -> .cursor/skills or ~/.cursor/skills"
  echo "                    claude -> .claude/skills or ~/.claude/skills"
  echo "  --global          Use user-level dir (e.g. ~/.cursor/skills) instead of project-level"
  echo "  --base-url <url>  Base URL for raw files"
  echo "  --dry-run         Show what would be downloaded without making changes"
  echo "  --help, -h        Show this help message and exit"
  echo ""
  echo "Examples:"
  echo "  ... | bash -s -- --target cursor --dry-run"
  echo "  ... | bash -s -- --target claude --global"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --global)  GLOBAL=true; shift ;;
    --help|-h) print_usage; exit 0 ;;
    --base-url)
      [[ -z "${2:-}" ]] && { echo "Missing value for $1" >&2; exit 1; }
      BASE_URL="$2"; shift 2
      ;;
    --target)
      [[ -z "${2:-}" ]] && { echo "Missing value for $1" >&2; exit 1; }
      TARGET="$2"; shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

# Resolve SKILLS_ROOT from target and global
case "$TARGET" in
  agents|codex|cursor|claude)
    if [[ "$GLOBAL" == true ]]; then
      SKILLS_ROOT="${HOME}/.${TARGET}/skills"
    else
      SKILLS_ROOT="${PWD}/.${TARGET}/skills"
    fi
    ;;
  *)
    echo "Unknown target: $TARGET (use: agents, codex, cursor, claude)" >&2
    exit 1
    ;;
esac

mkdir -p "$SKILLS_ROOT"

for tool in "${TOOLS[@]}"; do
  url="${BASE_URL}/${tool}/skill/SKILL.md"
  dest_dir="${SKILLS_ROOT}/${tool}"
  dest_file="${dest_dir}/SKILL.md"

  if [[ "$DRY_RUN" == true ]]; then
    echo "Would download: $url -> $dest_file"
    continue
  fi

  mkdir -p "$dest_dir"
  if curl -sSLf -o "$dest_file" "$url"; then
    echo "Installed: $dest_file"
  else
    echo "Failed to download: $url" >&2
    exit 1
  fi
done

if [[ "$DRY_RUN" != true ]]; then
  echo ""
  echo "IntelliKit skills are in ${SKILLS_ROOT}:"
  for tool in "${TOOLS[@]}"; do
    echo "  ${SKILLS_ROOT}/${tool}/SKILL.md"
  done
fi
