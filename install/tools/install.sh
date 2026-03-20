#!/usr/bin/env bash
# IntelliKit Tools Installer
# Installs tools from Git via pip (git+https...#subdirectory=<tool>).
# Usage: curl -sSL <install script URL> | bash -s -- [OPTIONS]
#    or: ./install.sh [OPTIONS]
# Pass options after bash -s -- when piping from curl so they reach this script.

set -e

ALL_TOOLS=(accordo kerncap linex metrix nexus rocm_mcp uprof_mcp)
INSTALL_SCRIPT_URL="https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh"
REPO_URL="https://github.com/AMDResearch/intellikit.git"
REF="main"
PIP_CMD="pip"
DRY_RUN=false
# Set only via --tools; empty = install all
TOOL_SELECTION=""

print_usage() {
  echo "IntelliKit Tools Installer"
  echo ""
  echo "Default: install all tools from Git: ${ALL_TOOLS[*]}"
  echo ""
  echo "Usage:"
  echo "  curl -sSL ${INSTALL_SCRIPT_URL} | bash -s -- [OPTIONS]"
  echo "  ./install.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --tools <list>    Comma-separated tools to install only (default: all)."
  echo "                    Example: --tools metrix,linex"
  echo "  --pip-cmd <cmd>   Pip command (default: pip). Example: --pip-cmd 'python3.12 -m pip'"
  echo "  -p <cmd>          Short for --pip-cmd"
  echo "  --repo-url <url>  Git repo URL (default: https://github.com/AMDResearch/intellikit.git)"
  echo "  --ref <ref>       Git branch/tag/commit (default: main)"
  echo "  --dry-run         Print pip commands without running them"
  echo "  --help, -h        Show this help message and exit"
  echo ""
  echo "Valid tool names: ${ALL_TOOLS[*]}"
  echo ""
  echo "Example (works with pipe; use args so overrides reach bash):"
  echo "  curl -sSL ${INSTALL_SCRIPT_URL} | bash -s -- --tools metrix,nexus --pip-cmd 'python3.12 -m pip' --dry-run"
}

require_arg() {
  local opt="$1"
  local val="$2"
  if [[ -z "${val}" || "${val}" == -* ]]; then
    echo "Missing or invalid value for ${opt}" >&2
    exit 1
  fi
}

tool_is_known() {
  local name="$1"
  local t
  for t in "${ALL_TOOLS[@]}"; do
    [[ "$t" == "$name" ]] && return 0
  done
  return 1
}

# Trim POSIX whitespace from string (parameter expansion).
trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --help|-h) print_usage; exit 0 ;;
    --tools)
      require_arg "$1" "${2:-}"
      TOOL_SELECTION="$2"
      shift 2
      ;;
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

INSTALL_TOOLS=()
if [[ -z "${TOOL_SELECTION}" ]]; then
  INSTALL_TOOLS=("${ALL_TOOLS[@]}")
else
  IFS=',' read -r -a _raw <<< "${TOOL_SELECTION}"
  already_in_install_list() {
    local needle="$1"
    local e
    for e in "${INSTALL_TOOLS[@]}"; do
      [[ "$e" == "$needle" ]] && return 0
    done
    return 1
  }
  for _part in "${_raw[@]}"; do
    _t="$(trim "${_part}")"
    [[ -z "${_t}" ]] && continue
    if ! tool_is_known "${_t}"; then
      echo "Unknown tool: ${_t}" >&2
      echo "Valid tools: ${ALL_TOOLS[*]}" >&2
      exit 1
    fi
    if already_in_install_list "${_t}"; then
      continue
    fi
    INSTALL_TOOLS+=("${_t}")
  done
  unset -f already_in_install_list
  if [[ ${#INSTALL_TOOLS[@]} -eq 0 ]]; then
    echo "No tools to install after parsing --tools." >&2
    exit 1
  fi
fi

# Pip requires git+ prefix for VCS installs
[[ "$REPO_URL" != git+* ]] && REPO_URL="git+${REPO_URL}"

for tool in "${INSTALL_TOOLS[@]}"; do
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
  echo "Done. Installed: ${INSTALL_TOOLS[*]}"
fi
