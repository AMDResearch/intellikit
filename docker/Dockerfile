# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

FROM rocm/pytorch:rocm7.1_ubuntu24.04_py3.13_pytorch_release_2.9.1

# Use bash shell for RUN commands
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV ROCM_PATH=/opt/rocm

ENV LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH \
    PATH="$ROCM_PATH/bin:$PATH"

# Install system packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git wget ninja-build cmake python3-pip python3-dev build-essential libdwarf-dev && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r video 2>/dev/null || true && \
    groupadd -r render 2>/dev/null || true

# Set up workspace
WORKDIR /workspace