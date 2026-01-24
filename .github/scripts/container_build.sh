#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

set -e

# Check if Apptainer is available
if ! command -v apptainer &> /dev/null; then
    echo "[ERROR] Apptainer not found. Please install Apptainer to continue"
    exit 1
fi

# Paths
DEF_FILE="apptainer/intellikit.def"
IMAGE_FILE=~/apptainer/intellikit-dev.sif
HASH_FILE=~/apptainer/intellikit.def.sha256

# Create directory
mkdir -p ~/apptainer

# Calculate current hash
CURRENT_HASH=$(sha256sum "$DEF_FILE" | awk '{print $1}')

# Check if rebuild is needed
if [ -f "$IMAGE_FILE" ] && [ -f "$HASH_FILE" ] && [ "$CURRENT_HASH" = "$(cat "$HASH_FILE")" ]; then
    echo "[INFO] Definition unchanged (hash: $CURRENT_HASH), using cached image"
    exit 0
fi

# Rebuild
echo "[INFO] Building Apptainer image..."
apptainer build --force "$IMAGE_FILE" "$DEF_FILE"
echo "$CURRENT_HASH" > "$HASH_FILE"
echo "[INFO] Build completed (hash: $CURRENT_HASH)"
