#!/usr/bin/env bash
# Run via: .github/scripts/apptainer_wrap.sh test_editable_install.sh

set -euo pipefail

_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_THIS_DIR}/install_test_pip_diff.inc.sh"

SCRIPT_DIR="${_THIS_DIR}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_REPO="${INTELLIKIT_REPO_ROOT:-${REPO_ROOT}}"

_testname="$(basename "$0" .sh)"
_intellikit_pip_log_setup "${_testname}" || true

TMP_DIR="$(mktemp -d -t intellikit-editable-XXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

REPO_CLONE="${TMP_DIR}/clone"
mkdir -p "${REPO_CLONE}"
cp -R "${SOURCE_REPO}" "${REPO_CLONE}/intellikit"

CLONE_INTELLIKIT="${REPO_CLONE}/intellikit"

# Drop any copied CMake build dirs so scikit-build does not reuse absolute paths
# from another workspace (e.g. /intellikit_workspace vs this temp clone).
for pkg in accordo nexus kerncap; do
  if [[ -d "${CLONE_INTELLIKIT}/${pkg}/build" ]]; then
    rm -rf "${CLONE_INTELLIKIT}/${pkg}/build"
  fi
done

declare -a PKGS=(accordo kerncap linex metrix nexus rocm_mcp uprof_mcp)

for pkg in "${PKGS[@]}"; do
  local_vdir="${TMP_DIR}/venv-${pkg}"
  python3 -m venv "${local_vdir}"
  "${local_vdir}/bin/pip" install --upgrade pip >/dev/null
done

if [[ -n "${_INTELLIKIT_PIP_BEFORE:-}" ]]; then
  _intellikit_pip_aggregate_venvs "${TMP_DIR}" "${_INTELLIKIT_PIP_BEFORE}"
fi

_editable_install_pkg() {
  local pkg="$1"
  local vdir="${TMP_DIR}/venv-${pkg}"
  echo ""
  echo "--- [editable] ${pkg} ---"
  # shellcheck disable=SC1091
  source "${vdir}/bin/activate"
  pip install -e "${CLONE_INTELLIKIT}/${pkg}"
  python - <<PY
import importlib

name = "${pkg}"
m = importlib.import_module(name)
ver = getattr(m, "__version__", "unknown")
print(f"[editable] import ok: {name} ({ver})")
PY
  deactivate
}

for pkg in "${PKGS[@]}"; do
  _editable_install_pkg "${pkg}"
done

if [[ -n "${_INTELLIKIT_PIP_AFTER:-}" ]]; then
  _intellikit_pip_aggregate_venvs "${TMP_DIR}" "${_INTELLIKIT_PIP_AFTER}"
fi
_intellikit_pip_log_finish "${_testname}"

echo ""
echo "[editable] Success — all subdirectory editable installs verified."
