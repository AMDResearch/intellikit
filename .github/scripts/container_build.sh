#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Build Apptainer container image

set -e

# Check if Apptainer is available
if ! command -v apptainer &> /dev/null; then
    echo "[ERROR] Apptainer not found"
    echo "[ERROR] Please install Apptainer to continue"
    exit 1
fi

echo "[INFO] Building Apptainer image..."

# Create persistent Apptainer directory
mkdir -p ~/apptainer

# Build Apptainer image from definition file (only if it doesn't exist)
if [ ! -f ~/apptainer/intellikit-dev.sif ]; then
    echo "[INFO] Building new Apptainer image..."
    apptainer build ~/apptainer/intellikit-dev.sif apptainer/intellikit.def
else
    echo "[INFO] Using existing Apptainer image at ~/apptainer/intellikit-dev.sif"
fi

echo "[INFO] Container build completed successfully"
