#!/usr/bin/env bash
set -euo pipefail

_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REF="${INTELLIKIT_INSTALL_REF:-main}"
GIT_URL="${INTELLIKIT_GIT_URL:-https://github.com/AMDResearch/intellikit.git}"

TMP_DIR="$(mktemp -d -t intellikit-pip-XXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

declare -a PKGS=(accordo kerncap linex metrix nexus rocm_mcp uprof_mcp)

for pkg in "${PKGS[@]}"; do
  vdir="${TMP_DIR}/venv-${pkg}"
  python3 -m venv "${vdir}"
  "${vdir}/bin/pip" install --upgrade pip >/dev/null
done

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

echo ""
echo "[pip-subdir] Success — each subdirectory install in its own venv, verified after pip."
