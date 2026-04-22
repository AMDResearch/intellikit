#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Simple Apptainer container exec script
# Usage: container_exec.sh <command>
#
# Image: INTELLIKIT_SIF if set; else if INTELLIKIT_SIF_HOME=1 use $HOME/apptainer/intellikit.sif
#       (matches container_build.sh for CI); else <repo>/apptainer/images/intellikit.sif.
#
# Overlay: ephemeral .img files under <repo>/apptainer/overlays/ (created per run, removed after).
#          Override directory with INTELLIKIT_OVERLAY_DIR=/path/to/dir

set -e

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"

# Command is all arguments
COMMAND="$@"
if [ -z "$COMMAND" ]; then
    echo "[ERROR] No command provided" >&2
    echo "Usage: $0 <command>" >&2
    exit 1
fi

# Check if Apptainer is available
if ! command -v apptainer &> /dev/null; then
    echo "[ERROR] Apptainer not found" >&2
    exit 1
fi

if [ -n "${INTELLIKIT_SIF:-}" ]; then
    IMAGE="${INTELLIKIT_SIF}"
elif [ "${INTELLIKIT_SIF_HOME:-}" = "1" ]; then
    IMAGE="${HOME}/apptainer/intellikit.sif"
else
    IMAGE="${_REPO_ROOT}/apptainer/images/intellikit.sif"
fi
if [ ! -f "$IMAGE" ]; then
    echo "[ERROR] Apptainer image not found at $IMAGE" >&2
    echo "[ERROR] Build with: bash apptainer/build.sh  (from repo root) or set INTELLIKIT_SIF" >&2
    exit 1
fi

# Writable overlay image (under repo apptainer/overlays/ by default)
OVERLAY_DIR="${INTELLIKIT_OVERLAY_DIR:-${_REPO_ROOT}/apptainer/overlays}"
mkdir -p "${OVERLAY_DIR}"
OVERLAY="${OVERLAY_DIR}/intellikit_overlay_$$_$(date +%s%N).img"
if ! apptainer overlay create --size 16384 --create-dir /var/cache/intellikit "${OVERLAY}" > /dev/null 2>&1; then
    echo "[ERROR] Failed to create Apptainer overlay"
    exit 1
fi

# Build exec command
EXEC_CMD="apptainer exec --overlay ${OVERLAY} --no-home --cleanenv"
EXEC_CMD="$EXEC_CMD --bind ${PWD}:/intellikit_workspace --cwd /intellikit_workspace"

# Execute with cleanup of overlay file
EXIT_CODE=0
$EXEC_CMD "$IMAGE" bash -c "set -e; $COMMAND" || EXIT_CODE=$?

# Clean up overlay file (always cleanup, even on failure)
rm -f "${OVERLAY}" 2>/dev/null || true

exit $EXIT_CODE
