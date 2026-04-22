#!/usr/bin/env bash
# Run commands inside an Apptainer image with the IntelliKit repo bind-mounted at the same path.
#
# Install tests:
#   apptainer_wrap.sh test_tools_install.sh
#   apptainer_wrap.sh test_skills_install.sh
#   apptainer_wrap.sh test_editable_install.sh
#   apptainer_wrap.sh test_pip_subdir_install.sh
#
# Generic:
#   apptainer_wrap.sh [-B|--bind HOST:CONTAINER]... [--env VAR=VAL]... [--] <command> [args...]
#
# Environment:
#   INTELLIKIT_APPTAINER_IMAGE  Path to .sif (optional; see resolve_image below)
#   WORK                        Search path for intelliperf-dev.sif
#   INTELLIKIT_INSTALL_REF      Git ref for default raw URLs / pip-subdir (default: main)
#   INTELLIKIT_GIT_URL          Git repo URL for pip-subdir test
#   INTELLIKIT_TOOLS_INSTALL_URL / INTELLIKIT_SKILLS_INSTALL_URL  Overrides

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Error: apptainer command not found" >&2
  exit 1
fi

resolve_image() {
  if [[ -n "${INTELLIKIT_APPTAINER_IMAGE:-}" ]]; then
    printf '%s\n' "${INTELLIKIT_APPTAINER_IMAGE}"
    return
  fi
  local repo_sif="${REPO_ROOT}/apptainer/images/intellikit.sif"
  if [[ -f "${repo_sif}" ]]; then
    printf '%s\n' "${repo_sif}"
    return
  fi
  if [[ -n "${WORK:-}" ]]; then
    local candidate="${WORK}/intelliperf/apptainer/intelliperf-dev.sif"
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  fi
  local default="/work1/amd/${USER}/intelliperf/apptainer/intelliperf-dev.sif"
  if [[ -f "${default}" ]]; then
    printf '%s\n' "${default}"
    return
  fi
  echo "Error: set INTELLIKIT_APPTAINER_IMAGE to the Apptainer .sif, or build ${repo_sif}." >&2
  exit 1
}

IMAGE="$(resolve_image)"
BIND_ARGS=(-B "${REPO_ROOT}:${REPO_ROOT}")
ENV_ARGS=(--env "PYTHONUNBUFFERED=1")

REF="${INTELLIKIT_INSTALL_REF:-main}"
RAW_BASE="https://raw.githubusercontent.com/AMDResearch/intellikit/${REF}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -B | --bind)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value" >&2
        exit 1
      fi
      BIND_ARGS+=(-B "$2")
      shift 2
      ;;
    --env)
      if [[ $# -lt 2 ]]; then
        echo "Error: --env requires a value" >&2
        exit 1
      fi
      ENV_ARGS+=(--env "$2")
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

resolve_install_test_script() {
  local a="$1"
  local path
  if [[ "${a}" == */* ]]; then
    if [[ ! -f "${a}" ]]; then
      return 1
    fi
    path="$(cd "$(dirname "${a}")" && pwd)/$(basename "${a}")"
  else
    path="${SCRIPT_DIR}/${a}"
  fi
  [[ -f "${path}" ]] || return 1
  printf '%s\n' "${path}"
}

if [[ $# -eq 1 ]]; then
  if test_script="$(resolve_install_test_script "$1")"; then
    case "$(basename "${test_script}")" in
      test_tools_install.sh)
        TOOLS_INSTALL_URL="${INTELLIKIT_TOOLS_INSTALL_URL:-${RAW_BASE}/install/tools/install.sh}"
        ENV_ARGS+=(--env "INTELLIKIT_TOOLS_INSTALL_URL=${TOOLS_INSTALL_URL}")
        ENV_ARGS+=(--env "PIP_NO_INPUT=1")
        echo "[apptainer_wrap] install test: ${test_script}"
        echo "[apptainer_wrap] INTELLIKIT_TOOLS_INSTALL_URL=${TOOLS_INSTALL_URL}"
        echo "[apptainer_wrap] image=${IMAGE}"
        echo "[apptainer_wrap] bind ${REPO_ROOT} -> ${REPO_ROOT}"
        exec apptainer exec "${BIND_ARGS[@]}" "${ENV_ARGS[@]}" "$IMAGE" bash "${test_script}"
        ;;
      test_skills_install.sh)
        SKILLS_INSTALL_URL="${INTELLIKIT_SKILLS_INSTALL_URL:-${RAW_BASE}/install/skills/install.sh}"
        ENV_ARGS+=(--env "INTELLIKIT_SKILLS_INSTALL_URL=${SKILLS_INSTALL_URL}")
        echo "[apptainer_wrap] install test: ${test_script}"
        echo "[apptainer_wrap] INTELLIKIT_SKILLS_INSTALL_URL=${SKILLS_INSTALL_URL}"
        echo "[apptainer_wrap] image=${IMAGE}"
        echo "[apptainer_wrap] bind ${REPO_ROOT} -> ${REPO_ROOT}"
        exec apptainer exec "${BIND_ARGS[@]}" "${ENV_ARGS[@]}" "$IMAGE" bash "${test_script}"
        ;;
      test_editable_install.sh)
        ENV_ARGS+=(--env "INTELLIKIT_REPO_ROOT=${REPO_ROOT}")
        echo "[apptainer_wrap] install test: ${test_script}"
        echo "[apptainer_wrap] image=${IMAGE}"
        echo "[apptainer_wrap] bind ${REPO_ROOT} -> ${REPO_ROOT}"
        exec apptainer exec "${BIND_ARGS[@]}" "${ENV_ARGS[@]}" "$IMAGE" bash "${test_script}"
        ;;
      test_pip_subdir_install.sh)
        GIT_URL="${INTELLIKIT_GIT_URL:-https://github.com/AMDResearch/intellikit.git}"
        ENV_ARGS+=(--env "INTELLIKIT_INSTALL_REF=${REF}")
        ENV_ARGS+=(--env "INTELLIKIT_GIT_URL=${GIT_URL}")
        echo "[apptainer_wrap] install test: ${test_script}"
        echo "[apptainer_wrap] INTELLIKIT_INSTALL_REF=${REF} INTELLIKIT_GIT_URL=${GIT_URL}"
        echo "[apptainer_wrap] image=${IMAGE}"
        echo "[apptainer_wrap] bind ${REPO_ROOT} -> ${REPO_ROOT}"
        exec apptainer exec "${BIND_ARGS[@]}" "${ENV_ARGS[@]}" "$IMAGE" bash "${test_script}"
        ;;
    esac
  fi
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 [-B|--bind HOST:CONTAINER]... [--env VAR=VAL]... [--] <command> [args...]" >&2
  echo "   or: $0 test_tools_install.sh | test_skills_install.sh | test_editable_install.sh | test_pip_subdir_install.sh" >&2
  exit 1
fi

echo "[apptainer_wrap] image=${IMAGE}"
echo "[apptainer_wrap] bind ${REPO_ROOT} -> ${REPO_ROOT}"
exec apptainer exec "${BIND_ARGS[@]}" "${ENV_ARGS[@]}" "$IMAGE" "$@"
