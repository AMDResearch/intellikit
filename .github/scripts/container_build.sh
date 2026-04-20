#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

set -e

# Check if Apptainer is available
if ! command -v apptainer &> /dev/null; then
    echo "[ERROR] Apptainer not found. Please install Apptainer to continue"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Image output: INTELLIKIT_SIF if set; else if INTELLIKIT_SIF_HOME=1 use $HOME/apptainer/intellikit.sif
# (survives git clean and separate Actions jobs on the same runner home); else repo path.
if [ -n "${INTELLIKIT_SIF:-}" ]; then
    IMAGE_FILE="${INTELLIKIT_SIF}"
elif [ "${INTELLIKIT_SIF_HOME:-}" = "1" ]; then
    IMAGE_FILE="${HOME}/apptainer/intellikit.sif"
else
    IMAGE_FILE="${REPO_ROOT}/apptainer/images/intellikit.sif"
fi
mkdir -p "$(dirname "${IMAGE_FILE}")"
HASH_FILE="$(dirname "${IMAGE_FILE}")/intellikit.def.sha256"

# Large Triton/vLLM builds exhaust small host /tmp; default to workspace dirs if unset.
if [ -z "${APPTAINER_TMPDIR:-}" ]; then
    export APPTAINER_TMPDIR="${REPO_ROOT}/apptainer/.apptainer-tmp"
fi
if [ -z "${APPTAINER_CACHEDIR:-}" ]; then
    export APPTAINER_CACHEDIR="${REPO_ROOT}/apptainer/.apptainer-cache"
fi
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

DEF_FILE="${REPO_ROOT}/apptainer/intellikit.def"

# Hash the definition only. When adding a vLLM (or other) layer that uses extra
# tracked inputs, extend this to include those files so cache invalidates correctly.
CURRENT_HASH=$(sha256sum "$DEF_FILE" | awk '{print $1}')

# Check if rebuild is needed
if [ -f "$IMAGE_FILE" ] && [ -f "$HASH_FILE" ] && [ "$CURRENT_HASH" = "$(cat "$HASH_FILE")" ]; then
    echo "[INFO] Definition unchanged (hash: $CURRENT_HASH), using cached image"
    exit 0
fi

# Rebuild (cwd = repo root so %files paths in the def resolve)
echo "[INFO] Building Apptainer image..."
ulimit -n 65535 2>/dev/null || ulimit -n 16384 2>/dev/null || true
cd "${REPO_ROOT}"
apptainer build --force "$IMAGE_FILE" "$DEF_FILE"
echo "$CURRENT_HASH" > "$HASH_FILE"
echo "[INFO] Build completed (hash: $CURRENT_HASH)"
