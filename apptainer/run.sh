#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"

cd $parent_dir

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

# Create a new filesystem image overlay for this run
timestamp=$(date +%s)
overlay="/tmp/intellikit_overlay_$(whoami)_${timestamp}.img"
echo "[Log] Overlay image ${overlay} does not exist. Creating overlay of ${size} MiB..."
apptainer overlay create --size ${size} --create-dir /var/cache/intellikit ${overlay}
echo "[Log] Utilize the directory /var/cache/intellikit as a sandbox to store data you'd like to persist between container runs."

# Run the container
image="apptainer/images/intellikit.sif"
apptainer exec --overlay ${overlay} --cleanenv $image bash --rcfile /etc/bash.bashrc
