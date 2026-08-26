#!/usr/bin/env bash
# IntelliKit Tools Installer
# Installs tools from Git via pip3 (git+https...#subdirectory=<tool>).
# Usage: curl -sSL <install script URL> | bash -s -- [OPTIONS]
#    or: ./install/tools/install.sh [OPTIONS]  (from repo root)
# Pass options after bash -s -- when piping from curl so they reach this script.

set -e

ALL_TOOLS=(accordo kerncap linex metrix nexus rocm_mcp uprof_mcp)
INSTALL_SCRIPT_URL="https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh"
REPO_URL="https://github.com/AMDResearch/intellikit.git"
REF="main"
PIP_CMD="pip3"
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
  echo "  ./install/tools/install.sh [OPTIONS]   # from repo root"
  echo ""
  echo "Options:"
  echo "  --tools <list>    Comma-separated tools to install only (default: all)."
  echo "                    Example: --tools metrix,linex"
  echo "  --pip-cmd <cmd>   Pip command (default: pip3). Example: --pip-cmd 'python3.12 -m pip'"
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

# Require Python >= 3.10 for the interpreter used by PIP_CMD.
require_python_ge_310() {
  local ver_line major minor py_exe

  if ! ver_line=$(eval "${PIP_CMD} --version 2>&1"); then
    echo "Error: cannot run: ${PIP_CMD} --version" >&2
    exit 1
  fi

  if [[ "$ver_line" =~ \(python\ ([0-9]+)\.([0-9]+) ]]; then
    major="${BASH_REMATCH[1]}"
    minor="${BASH_REMATCH[2]}"
    if (( 10#$major < 3 || (10#$major == 3 && 10#$minor < 10) )); then
      echo "Error: Python ${major}.${minor} is too old. IntelliKit requires Python 3.10 or newer." >&2
      echo "(${ver_line})" >&2
      exit 1
    fi
    return 0
  fi

  if [[ "$PIP_CMD" == *" -m pip"* ]]; then
    py_exe="${PIP_CMD%% -m pip*}"
    py_exe="$(trim "$py_exe")"
    if [[ -z "$py_exe" ]]; then
      echo "Error: could not parse interpreter from --pip-cmd=${PIP_CMD}" >&2
      exit 1
    fi
    if ! "$py_exe" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      echo "Error: ${py_exe} must be Python 3.10 or newer (IntelliKit requirement)." >&2
      exit 1
    fi
    return 0
  fi

  if [[ "$PIP_CMD" == "pip3" || "$PIP_CMD" == "pip" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        return 0
      fi
      echo "Error: python3 must be 3.10 or newer. (${ver_line})" >&2
      exit 1
    fi
  fi

  echo "Error: could not verify Python >= 3.10 for PIP_CMD=${PIP_CMD}" >&2
  echo "${PIP_CMD} --version reported: ${ver_line}" >&2
  echo "Use a Python 3.10+ pip, or pass --pip-cmd 'python3.12 -m pip'." >&2
  exit 1
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

require_python_ge_310

# --- System dependency check for tools with C++ builds ---
# accordo and nexus depend on KernelDB which requires cmake, libdwarf-dev,
# and libzstd-dev to compile. Without these, pip install will fail during
# the C++ build step.
needs_native_deps() {
  local t
  for t in "$@"; do
    [[ "$t" == "accordo" || "$t" == "nexus" ]] && return 0
  done
  return 1
}

check_system_deps() {
  local missing=()
  if ! command -v cmake >/dev/null 2>&1; then
    missing+=(cmake)
  fi
  # Check for libdwarf and libzstd headers (dpkg on Debian/Ubuntu, rpm on Fedora/RHEL)
  if command -v dpkg >/dev/null 2>&1; then
    if ! dpkg -s libdwarf-dev >/dev/null 2>&1; then
      missing+=(libdwarf-dev)
    fi
    if ! dpkg -s libzstd-dev >/dev/null 2>&1; then
      missing+=(libzstd-dev)
    fi
  elif command -v rpm >/dev/null 2>&1; then
    if ! rpm -q libdwarf-devel >/dev/null 2>&1; then
      missing+=(libdwarf-devel)
    fi
    if ! rpm -q libzstd-devel >/dev/null 2>&1; then
      missing+=(libzstd-devel)
    fi
  fi
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "" >&2
    echo "Error: Missing system packages required by accordo/nexus: ${missing[*]}" >&2
    echo "Install them first:" >&2
    if command -v apt-get >/dev/null 2>&1; then
      echo "  sudo apt-get update && sudo apt-get install -y ${missing[*]}" >&2
    elif command -v dnf >/dev/null 2>&1; then
      echo "  sudo dnf install -y ${missing[*]}" >&2
    elif command -v yum >/dev/null 2>&1; then
      echo "  sudo yum install -y ${missing[*]}" >&2
    else
      echo "  (use your system package manager to install: ${missing[*]})" >&2
    fi
    echo "" >&2
    echo "Without these, the C++ build step for accordo/nexus (via KernelDB) will fail." >&2
    echo "" >&2
    if [[ "$DRY_RUN" == true ]]; then
      echo "Dry run: continuing despite missing system dependencies." >&2
      return 0
    fi
    exit 1
  fi
}

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

# Warn about missing system deps if installing tools that need C++ builds
if needs_native_deps "${INSTALL_TOOLS[@]}"; then
  check_system_deps
fi

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
