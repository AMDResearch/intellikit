#!/usr/bin/env bash
set -euo pipefail

_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INTELLIKIT_TOOLS_INSTALL_URL="${INTELLIKIT_TOOLS_INSTALL_URL:-https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh}"

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl not found" >&2
  exit 1
fi

echo "[tools] Installer URL: ${INTELLIKIT_TOOLS_INSTALL_URL}"

TMP_DIR="$(mktemp -d -t intellikit-tools-XXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

python3 -m venv "${TMP_DIR}/venv"
# shellcheck disable=SC1091
source "${TMP_DIR}/venv/bin/activate"

pip install --upgrade pip >/dev/null

curl -sSL "${INTELLIKIT_TOOLS_INSTALL_URL}" | bash -s -- --pip-cmd "python -m pip"

python <<'PY'
import importlib

modules = ["accordo", "kerncap", "linex", "metrix", "nexus", "rocm_mcp", "uprof_mcp"]
failed = []

for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append(f"{name}: {exc}")

if failed:
    raise SystemExit("Module import check failed: " + ", ".join(failed))

print("All IntelliKit tools imported successfully.")
PY

echo "[tools] IntelliKit tools install test completed."
