#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"

cd "$parent_dir"

mkdir -p "${parent_dir}/apptainer/overlays"

size=1024
while getopts "s:" opt; do
    case $opt in
        s)
            size=$OPTARG
            ;;
        *)
            echo "Usage: $0 [-s size]"
            exit 1
            ;;
    esac
done

# Writable overlay under apptainer/overlays/ (same convention as .github/scripts/container_exec.sh)
timestamp=$(date +%s)
overlay="${parent_dir}/apptainer/overlays/intellikit_overlay_$(whoami)_${timestamp}.img"
echo "[Log] Creating overlay (${size} MiB): ${overlay}"
apptainer overlay create --size "${size}" --create-dir /var/cache/intellikit "${overlay}"
echo "[Log] Overlay persists until you remove the .img; /var/cache/intellikit inside the overlay can hold writable state."

# Run the container
image="apptainer/images/intellikit.sif"
apptainer exec --overlay ${overlay} --cleanenv $image bash --rcfile /etc/bash.bashrc
