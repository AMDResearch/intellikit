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

# A git transfer has no timeout by default, so a stalled connection does not
# fail -- it waits, and the job dies on timeout-minutes with the log stopped
# mid-build naming nothing. The installs run in here clone four things:
# accordo's CMake fetches spdlog, fmt and nlohmann/json while configuring, and
# pip installs kerneldb from a git URL.
#
# Abort a transfer that moves under 1 KB/s for ten minutes. git then exits with
# an error identifying the URL, which is the point -- the job timeout alone says
# only that something took too long.
#
# Ten minutes, not one: git moves no bytes while the server enumerates objects,
# so a healthy clone of a large-history repo can sit under any rate threshold
# for a while, and a short window would abort good builds. This is deliberately
# far looser than needed to catch a stall -- it still fires well inside both job
# timeouts (20m for tests, 60m for container builds), and the value here is the
# error naming the URL, not the minutes saved.
#
# Set as environment variables rather than `git config` -- --global would write
# a .gitconfig into the bind-mounted workspace, since HOME points there under
# docker, and --system is unwritable in a read-only Apptainer image.
GIT_STALL_GUARD='export GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=600;'

# Apptainer being installed does not mean it can build. Building runs %post as
# root inside a user namespace: a setuid starter does that directly, otherwise
# it needs a uid mapping, which AppArmor can refuse even when the userns
# sysctls read permissive. Probe the operation rather than the binary.
apptainer_can_build() {
    [ -u /usr/libexec/apptainer/bin/starter-suid ] && return 0
    unshare --user --map-root-user true 2>/dev/null
}

# See container_build.sh: CONTAINER_RUNTIME is honoured only when the caller sets
# it. Defaulting it here had the same effect it had there -- the branch below is
# `if [ -n "$CONTAINER_RUNTIME" ]`, so a default made every run take the "forced"
# path and left the autodetection beneath it unreachable. This file is the twin of
# container_build.sh and has to move with it: with only the build side fixed, the
# image built under apptainer and then every exec step still died demanding docker.

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
    # Mirrors APPTAINER_IMAGE_NAME in container_build.sh so exec picks up the
    # image that build just produced for this matrix leg.
    IMAGE=~/apptainer/${APPTAINER_IMAGE_NAME:-intellikit-dev}.sif
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
    $EXEC_CMD "$IMAGE" bash -c "set -e; $GIT_STALL_GUARD $COMMAND" || EXIT_CODE=$?

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
    $RUN_CMD "$IMAGE_NAME" -c "set -e; $GIT_STALL_GUARD $COMMAND" || EXIT_CODE=$?

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
