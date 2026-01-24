#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Simple Apptainer container exec script
# Usage: container_exec.sh <command>

set -e

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

# Use fixed image path
IMAGE=~/apptainer/intellikit-dev.sif
if [ ! -f "$IMAGE" ]; then
    echo "[ERROR] Apptainer image not found at $IMAGE" >&2
    exit 1
fi

# Create temporary overlay in workspace (auto-cleaned when runner is removed)
OVERLAY="./intellikit_overlay_$$_$(date +%s%N).img"
if ! apptainer overlay create --size 16384 --create-dir /var/cache/intellikit "${OVERLAY}" > /dev/null 2>&1; then
    echo "[ERROR] Failed to create Apptainer overlay"
    exit 1
fi

# Build exec command
EXEC_CMD="apptainer exec --overlay ${OVERLAY} --no-home --cleanenv"
EXEC_CMD="$EXEC_CMD --bind ${PWD}:/intellikit_workspace --cwd /intellikit_workspace"

# Execute with cleanup of overlay file
EXIT_CODE=0
$EXEC_CMD "$IMAGE" bash -c "$COMMAND" || EXIT_CODE=$?

# Clean up overlay file (always cleanup, even on failure)
rm -f "${OVERLAY}" 2>/dev/null || true

exit $EXIT_CODE
