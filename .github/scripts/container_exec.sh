#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Execute a command inside the IntelliKit dev container, using whichever
# runtime is available. Mirrors the dual-runtime selection iris uses.
#
# Usage: container_exec.sh <command>

set -e

COMMAND="$@"
if [ -z "$COMMAND" ]; then
    echo "[ERROR] No command provided" >&2
    echo "Usage: $0 <command>" >&2
    exit 1
fi

# Apptainer being installed does not mean it can build. Building runs %post as
# root inside a user namespace: a setuid starter does that directly, otherwise
# it needs a uid mapping, which AppArmor can refuse even when the userns
# sysctls read permissive. Probe the operation rather than the binary.
apptainer_can_build() {
    [ -u /usr/libexec/apptainer/bin/starter-suid ] && return 0
    unshare --user --map-root-user true 2>/dev/null
}

# See container_build.sh for why an explicit override exists.
# Apptainer is disabled for now: the CI hosts either cannot build with it
# (no setuid starter, no usable user namespace) or do not have it at all,
# and the runner has no root to install it. Remove this line to restore
# autodetection.
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"

if [ -n "$CONTAINER_RUNTIME" ]; then
    if ! command -v "$CONTAINER_RUNTIME" &> /dev/null; then
        echo "[ERROR] CONTAINER_RUNTIME=$CONTAINER_RUNTIME but it is not installed" >&2
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
    echo "[ERROR] Neither Apptainer nor Docker is available" >&2
    exit 1
fi

if [ "$CONTAINER_RUNTIME" = "apptainer" ]; then
    IMAGE=~/apptainer/intellikit-dev.sif
    if [ ! -f "$IMAGE" ]; then
        echo "[ERROR] Apptainer image not found at $IMAGE" >&2
        exit 1
    fi

    # Temporary overlay in the workspace (auto-cleaned below)
    OVERLAY="./intellikit_overlay_$$_$(date +%s%N).img"
    if ! apptainer overlay create --size 16384 --create-dir /var/cache/intellikit "${OVERLAY}" > /dev/null 2>&1; then
        echo "[ERROR] Failed to create Apptainer overlay" >&2
        exit 1
    fi

    EXEC_CMD="apptainer exec --overlay ${OVERLAY} --no-home --cleanenv"
    EXEC_CMD="$EXEC_CMD --bind ${PWD}:/intellikit_workspace --cwd /intellikit_workspace"

    EXIT_CODE=0
    $EXEC_CMD "$IMAGE" bash -c "set -e; $COMMAND" || EXIT_CODE=$?

    rm -f "${OVERLAY}" 2>/dev/null || true
    exit $EXIT_CODE

elif [ "$CONTAINER_RUNTIME" = "docker" ]; then
    IMAGE_NAME=${DOCKER_IMAGE_NAME:-"intellikit-dev"}
    if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
        echo "[ERROR] Docker image $IMAGE_NAME not found; run container_build.sh first" >&2
        exit 1
    fi

    # --device/--group-add give the container the GPUs; SYS_PTRACE and an
    # unconfined seccomp profile are required by the profiling tools, which
    # attach to and trace the processes they launch.
    RUN_CMD="docker run --rm --network=host --device=/dev/kfd --device=/dev/dri"
    RUN_CMD="$RUN_CMD --group-add video --group-add render"
    RUN_CMD="$RUN_CMD --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    RUN_CMD="$RUN_CMD --shm-size=16G --ulimit memlock=-1 --ulimit stack=67108864"
    RUN_CMD="$RUN_CMD -v ${PWD}:/intellikit_workspace -w /intellikit_workspace"
    RUN_CMD="$RUN_CMD -e HOME=/intellikit_workspace"
    RUN_CMD="$RUN_CMD --entrypoint bash"

    # The container runs as root, so anything it writes into the bind-mounted
    # workspace is root-owned. The runner is an ordinary user and must be able
    # to delete those files when it checks out again, so hand them back before
    # exiting -- including when the command failed. Apptainer does not need
    # this: it runs as the invoking user.
    EXIT_CODE=0
    $RUN_CMD "$IMAGE_NAME" -c "set -e; $COMMAND" || EXIT_CODE=$?

    # The container runs as root, so anything it wrote into the bind-mounted
    # workspace is root-owned, and actions/checkout -- running as an ordinary
    # user -- cannot delete it on the next job. Hand ownership back in a
    # separate invocation: wrapping the command instead would splice shell
    # syntax into a caller-supplied, possibly multi-line, command string.
    # Runs after failures too, since a failed job still leaves artifacts.
    docker run --rm -v "${PWD}:/intellikit_workspace" \
        --entrypoint chown "$IMAGE_NAME" \
        -R "$(id -u):$(id -g)" /intellikit_workspace 2>/dev/null || true

    exit $EXIT_CODE
fi
