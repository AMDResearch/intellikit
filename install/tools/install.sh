#!/usr/bin/env bash
# IntelliKit Tools Installer
# Installs all tools from Git via pip (git+https...#subdirectory=<tool>).
# Usage: curl -sSL .../install/tools/install.sh | bash
#    or: ./install.sh [--dry-run]

set -e

TOOLS=(accordo linex metrix nexus rocm_mcp uprof_mcp)
REPO_URL="${INTELLIKIT_REPO_URL:-https://github.com/AMDResearch/intellikit.git}"
REF="${INTELLIKIT_REF:-main}"
PIP_CMD="${PIP_CMD:-pip}"
DRY_RUN=false

print_usage() {
  echo "IntelliKit Tools Installer"
  echo ""
  echo "Installs all tools from Git: ${TOOLS[*]}"
  echo ""
  echo "Usage:"
  echo "  curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash [--dry-run]"
  echo "  ./install.sh [--dry-run]"
  echo ""
  echo "Options:"
  echo "  --dry-run  Print pip commands without running them"
  echo "  --help, -h Show this help message and exit"
  echo ""
  echo "Override: PIP_CMD (default: pip), INTELLIKIT_REPO_URL, INTELLIKIT_REF (default: main)"
  echo "  Example: PIP_CMD=\"python3.12 -m pip\" ./install.sh"
}

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
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
