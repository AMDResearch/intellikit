#!/usr/bin/env bash
set -euo pipefail

_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${INTELLIKIT_SKILLS_INSTALL_URL:-}" ]]; then
  REF="${INTELLIKIT_INSTALL_REF:-main}"
  RAW_BASE="https://raw.githubusercontent.com/AMDResearch/intellikit/${REF}"
  INTELLIKIT_SKILLS_INSTALL_URL="${RAW_BASE}/install/skills/install.sh"
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl not found" >&2
  exit 1
fi

echo "[skills] Installer URL: ${INTELLIKIT_SKILLS_INSTALL_URL}"

TMP_DIR="$(mktemp -d -t intellikit-skills-XXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

cd "${TMP_DIR}"

curl -sSL "${INTELLIKIT_SKILLS_INSTALL_URL}" | bash -s -- --target agents

declare -a REQUIRED=(metrix accordo nexus linex kerncap)
for name in "${REQUIRED[@]}"; do
  path=".agents/skills/${name}/SKILL.md"
  if [[ ! -f "${path}" ]]; then
    echo "Missing skill file: ${path}" >&2
    exit 1
  fi
  if [[ ! -s "${path}" ]]; then
    echo "Empty skill file: ${path}" >&2
    exit 1
  fi
  echo "[skills] verified ${path}"
done

echo "[skills] IntelliKit skills install test completed."
