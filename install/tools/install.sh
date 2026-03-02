#!/usr/bin/env bash
# IntelliKit Tools Installer
# Installs all tools from Git via pip (git+https...#subdirectory=<tool>).
# Usage: curl -sSL .../install/tools/install.sh | bash -s -- [OPTIONS]
#    or: ./install.sh [OPTIONS]
# Use --pip-cmd 'python3.12 -m pip' etc.; args work correctly when piped to bash.

set -e

TOOLS=(accordo linex metrix nexus rocm_mcp uprof_mcp)
REPO_URL="https://github.com/AMDResearch/intellikit.git"
REF="main"
PIP_CMD="pip"
DRY_RUN=false

print_usage() {
  echo "IntelliKit Tools Installer"
  echo ""
  echo "Installs all tools from Git: ${TOOLS[*]}"
  echo ""
  echo "Usage:"
  echo "  curl -sSL .../install/tools/install.sh | bash -s -- [OPTIONS]"
  echo "  ./install.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --pip-cmd <cmd>   Pip command (default: pip). Example: --pip-cmd 'python3.12 -m pip'"
  echo "  -p <cmd>          Short for --pip-cmd"
  echo "  --repo-url <url>  Git repo URL (default: https://github.com/AMDResearch/intellikit.git)"
  echo "  --ref <ref>       Git branch/tag/commit (default: main)"
  echo "  --dry-run         Print pip commands without running them"
  echo "  --help, -h        Show this help message and exit"
  echo ""
  echo "Example (works with pipe):"
  echo "  curl -sSL .../install/tools/install.sh | bash -s -- --pip-cmd 'python3.12 -m pip' --dry-run"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --help|-h) print_usage; exit 0 ;;
    --pip-cmd|-p)
      [[ -z "${2:-}" ]] && { echo "Missing value for $1" >&2; exit 1; }
      PIP_CMD="$2"; shift 2
      ;;
    --repo-url)
      [[ -z "${2:-}" ]] && { echo "Missing value for $1" >&2; exit 1; }
      REPO_URL="$2"; shift 2
      ;;
    --ref)
      [[ -z "${2:-}" ]] && { echo "Missing value for $1" >&2; exit 1; }
      REF="$2"; shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

# Pip requires git+ prefix for VCS installs
[[ "$REPO_URL" != git+* ]] && REPO_URL="git+${REPO_URL}"

for tool in "${TOOLS[@]}"; do
  url="${REPO_URL}@${REF}#subdirectory=${tool}"
  if [[ "$DRY_RUN" == true ]]; then
    echo "Would run: ${PIP_CMD} install \"${url}\""
  else
    echo "Installing $tool..."
    eval "${PIP_CMD} install \"${url}\""
  fi
done

if [[ "$DRY_RUN" != true ]]; then
  echo ""
  echo "Done. Installed: ${TOOLS[*]}"
fi
