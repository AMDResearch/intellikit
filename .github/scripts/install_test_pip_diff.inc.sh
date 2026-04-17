# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Optional pip reporting when INTELLIKIT_INSTALL_TEST_LOG_DIR is set.
# Source from a test_*.sh after set -euo pipefail.
#
# shellcheck shell=bash

_intellikit_pip_logging() {
  [[ -n "${INTELLIKIT_INSTALL_TEST_LOG_DIR:-}" ]]
}

# Args: test_name (e.g. test_tools_install, no .sh)
_intellikit_pip_log_setup() {
  local name="$1"
  if ! _intellikit_pip_logging; then
    export _INTELLIKIT_PIP_BEFORE=""
    export _INTELLIKIT_PIP_AFTER=""
    return 1
  fi
  mkdir -p "${INTELLIKIT_INSTALL_TEST_LOG_DIR}"
  export _INTELLIKIT_PIP_BEFORE="${INTELLIKIT_INSTALL_TEST_LOG_DIR}/${name}.pip.before.txt"
  export _INTELLIKIT_PIP_AFTER="${INTELLIKIT_INSTALL_TEST_LOG_DIR}/${name}.pip.after.txt"
  export _INTELLIKIT_PIP_DIFF="${INTELLIKIT_INSTALL_TEST_LOG_DIR}/${name}.pip.diff.txt"
  return 0
}

_intellikit_pip_freeze_to() {
  local py="$1"
  local out="$2"
  if [[ -x "${py}" ]] || command -v "${py}" >/dev/null 2>&1; then
    "${py}" -m pip list --format=freeze 2>/dev/null | LC_ALL=C sort -u >"${out}"
  else
    : >"${out}"
  fi
}

# Args: tmp_root (directory containing venv-*), output file
_intellikit_pip_aggregate_venvs() {
  local tmp="${1:?}"
  local out="${2:?}"
  {
    echo "=== system python ==="
    python3 -m pip list --format=freeze 2>/dev/null | LC_ALL=C sort -u || true
    shopt -s nullglob
    for d in "${tmp}"/venv-*; do
      [[ -d "${d}" ]] || continue
      [[ -x "${d}/bin/python" ]] || continue
      echo "=== ${d##*/} ==="
      "${d}/bin/python" -m pip list --format=freeze 2>/dev/null | LC_ALL=C sort -u || true
    done
  } >"${out}"
}

# Print full lists, unified diff, and write .diff file.
_intellikit_pip_log_finish() {
  local title="$1"
  if ! _intellikit_pip_logging; then
    return 0
  fi
  [[ -n "${_INTELLIKIT_PIP_BEFORE:-}" && -n "${_INTELLIKIT_PIP_AFTER:-}" ]] || return 0
  [[ -f "${_INTELLIKIT_PIP_BEFORE}" && -f "${_INTELLIKIT_PIP_AFTER}" ]] || return 0

  diff -u "${_INTELLIKIT_PIP_BEFORE}" "${_INTELLIKIT_PIP_AFTER}" >"${_INTELLIKIT_PIP_DIFF}" || true

  echo ""
  echo "========== ${title}: pip list BEFORE =========="
  cat "${_INTELLIKIT_PIP_BEFORE}"
  echo ""
  echo "========== ${title}: pip list AFTER =========="
  cat "${_INTELLIKIT_PIP_AFTER}"
  echo ""
  echo "========== ${title}: pip list unified diff (after vs before) =========="
  cat "${_INTELLIKIT_PIP_DIFF}"
  echo ""
}
