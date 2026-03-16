#!/usr/bin/env bash
# IntelliKit Tools Installer
# Installs all tools from Git via pip (git+https...#subdirectory=<tool>).
# Usage: curl -sSL <install script URL> | bash -s -- [OPTIONS]
#    or: ./install.sh [OPTIONS]
# CLI options override env (PIP_CMD, INTELLIKIT_REPO_URL, INTELLIKIT_REF). Use args when piping so overrides reach bash.

set -e

TOOLS=(accordo kerncap linex metrix nexus rocm_mcp uprof_mcp)
INSTALL_SCRIPT_URL="https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh"
# Env defaults (CLI options override these); use args when piping to bash so overrides apply
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
  echo "  curl -sSL ${INSTALL_SCRIPT_URL} | bash -s -- [OPTIONS]"
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
  echo "Example (works with pipe; use args so overrides reach bash):"
  echo "  curl -sSL ${INSTALL_SCRIPT_URL} | bash -s -- --pip-cmd 'python3.12 -m pip' --dry-run"
}

require_arg() {
  local opt="$1"
  local val="$2"
  if [[ -z "${val}" || "${val}" == -* ]]; then
    echo "Missing or invalid value for ${opt}" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --help|-h) print_usage; exit 0 ;;
    --pip-cmd|-p)
      require_arg "$1" "${2:-}"
      PIP_CMD="$2"; shift 2
      ;;
    --repo-url)
      require_arg "$1" "${2:-}"
      REPO_URL="$2"; shift 2
      ;;
    --ref)
      require_arg "$1" "${2:-}"
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
