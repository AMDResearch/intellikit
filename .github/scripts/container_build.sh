#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Build the IntelliKit dev container with whichever runtime is available.
# Mirrors the dual-runtime selection iris uses in the same script.

set -e

# Apptainer being installed does not mean it can build. Building runs %post as
# root inside a user namespace: a setuid starter does that directly, otherwise
# it needs a uid mapping, which AppArmor can refuse even when the userns
# sysctls read permissive. Probe the operation rather than the binary.
apptainer_can_build() {
    [ -u /usr/libexec/apptainer/bin/starter-suid ] && return 0
    unshare --user --map-root-user true 2>/dev/null
}

# CONTAINER_RUNTIME forces a runtime. This matters when apptainer is installed
# but cannot build: on a host with kernel.apparmor_restrict_unprivileged_userns=1,
# no setuid starter and no /etc/subuid entry, `apptainer build` fails in %post
# while docker works. Autodetection alone would pick apptainer and fail there.
if [ -n "$CONTAINER_RUNTIME" ]; then
    if ! command -v "$CONTAINER_RUNTIME" &> /dev/null; then
        echo "[ERROR] CONTAINER_RUNTIME=$CONTAINER_RUNTIME but it is not installed"
        exit 1
    fi
    echo "[INFO] Using $CONTAINER_RUNTIME (forced via CONTAINER_RUNTIME)"
elif command -v apptainer &> /dev/null && apptainer_can_build; then
    CONTAINER_RUNTIME="apptainer"
    echo "[INFO] Using Apptainer"
elif command -v docker &> /dev/null; then
    CONTAINER_RUNTIME="docker"
    if command -v apptainer &> /dev/null; then
        echo "[INFO] Using Docker (Apptainer is installed but cannot build:" \
             "no setuid starter and no usable user namespace)"
    else
        echo "[INFO] Using Docker"
    fi
elif command -v apptainer &> /dev/null; then
    CONTAINER_RUNTIME="apptainer"
    echo "[WARN] Using Apptainer, which cannot build on this host, and Docker is absent"
else
    echo "[ERROR] Neither Apptainer nor Docker is available"
    exit 1
fi

if [ "$CONTAINER_RUNTIME" = "apptainer" ]; then
    DEF_FILE="apptainer/intellikit.def"
    IMAGE_FILE=~/apptainer/intellikit-dev.sif
    HASH_FILE=~/apptainer/intellikit.def.sha256

    mkdir -p ~/apptainer
    CURRENT_HASH=$(sha256sum "$DEF_FILE" | awk '{print $1}')

    if [ -f "$IMAGE_FILE" ] && [ -f "$HASH_FILE" ] && [ "$CURRENT_HASH" = "$(cat "$HASH_FILE")" ]; then
        echo "[INFO] Definition unchanged (hash: $CURRENT_HASH), using cached image"
        exit 0
    fi

    echo "[INFO] Building Apptainer image..."
    apptainer build --force "$IMAGE_FILE" "$DEF_FILE"
    echo "$CURRENT_HASH" > "$HASH_FILE"
    echo "[INFO] Build completed (hash: $CURRENT_HASH)"

elif [ "$CONTAINER_RUNTIME" = "docker" ]; then
    IMAGE_NAME=${DOCKER_IMAGE_NAME:-"intellikit-dev"}
    DOCKER_DIR="$(dirname "$(realpath "$0")")/../../docker"

    # Rebuild when the Dockerfile changes, mirroring the def-hash check above.
    CURRENT_HASH=$(sha256sum "$DOCKER_DIR/Dockerfile" | awk '{print $1}')
    HASH_FILE=~/.intellikit-docker-image.sha256

    if docker image inspect "$IMAGE_NAME" &> /dev/null \
       && [ -f "$HASH_FILE" ] && [ "$CURRENT_HASH" = "$(cat "$HASH_FILE")" ]; then
        echo "[INFO] Dockerfile unchanged (hash: $CURRENT_HASH), using existing image: $IMAGE_NAME"
        exit 0
    fi

    echo "[INFO] Building Docker image: $IMAGE_NAME"
    docker build -t "$IMAGE_NAME" "$DOCKER_DIR"
    echo "$CURRENT_HASH" > "$HASH_FILE"
    echo "[INFO] Build completed (hash: $CURRENT_HASH)"
fi

echo "[INFO] Container build completed successfully with $CONTAINER_RUNTIME"
