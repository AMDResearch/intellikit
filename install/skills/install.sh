#!/usr/bin/env bash
# IntelliKit Agent Skills Installer
# Downloads each tool's SKILL.md into .agents/skills/<tool>/ (local) or ~/.agents/skills/ (--global).
# Usage: curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/skills/install.sh | bash
#    or: ./install.sh [--global] [--dry-run]

set -e

BASE_URL="${INTELLIKIT_RAW_URL:-https://raw.githubusercontent.com/AMDResearch/intellikit/main}"
TOOLS=(metrix accordo nexus)
DRY_RUN=false
GLOBAL=false

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --global)  GLOBAL=true ;;
  esac
done

if [[ "$GLOBAL" == true ]]; then
  SKILLS_ROOT="${HOME}/.agents/skills"
else
  SKILLS_ROOT="${PWD}/.agents/skills"
fi

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
