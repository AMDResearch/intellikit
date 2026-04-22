#!/usr/bin/env bash
# Run via: .github/scripts/apptainer_wrap.sh test_pip_subdir_install.sh

set -euo pipefail

_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_THIS_DIR}/install_test_pip_diff.inc.sh"

REF="${INTELLIKIT_INSTALL_REF:-main}"
GIT_URL="${INTELLIKIT_GIT_URL:-https://github.com/AMDResearch/intellikit.git}"

_testname="$(basename "$0" .sh)"
_intellikit_pip_log_setup "${_testname}" || true

TMP_DIR="$(mktemp -d -t intellikit-pip-XXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

declare -a PKGS=(accordo kerncap linex metrix nexus rocm_mcp uprof_mcp)

for pkg in "${PKGS[@]}"; do
  vdir="${TMP_DIR}/venv-${pkg}"
  python3 -m venv "${vdir}"
  "${vdir}/bin/pip" install --upgrade pip >/dev/null
done

if [[ -n "${_INTELLIKIT_PIP_BEFORE:-}" ]]; then
  _intellikit_pip_aggregate_venvs "${TMP_DIR}" "${_INTELLIKIT_PIP_BEFORE}"
fi

_pip_subdir_install_pkg() {
  local pkg="$1"
  local vdir="${TMP_DIR}/venv-${pkg}"
  echo ""
  echo "--- [pip-subdir] ${pkg} ---"
  # shellcheck disable=SC1091
  source "${vdir}/bin/activate"
  echo "[pip-subdir] Installing ${pkg} from Git @${REF}#subdirectory=${pkg} ..."
  pip install "git+${GIT_URL}@${REF}#subdirectory=${pkg}"
  python - <<PY
import importlib

name = "${pkg}"
m = importlib.import_module(name)
ver = getattr(m, "__version__", "unknown")
print(f"[pip-subdir] import ok: {name} ({ver})")
PY
  deactivate
}

for pkg in "${PKGS[@]}"; do
  _pip_subdir_install_pkg "${pkg}"
done

if [[ -n "${_INTELLIKIT_PIP_AFTER:-}" ]]; then
  _intellikit_pip_aggregate_venvs "${TMP_DIR}" "${_INTELLIKIT_PIP_AFTER}"
fi
_intellikit_pip_log_finish "${_testname}"

echo ""
echo "[pip-subdir] Success — each subdirectory install in its own venv, verified after pip."
