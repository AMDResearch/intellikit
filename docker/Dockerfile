# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Kept in step with apptainer/intellikit.def -- CI may run under either runtime,
# so the two must describe the same environment. Change both together.

FROM rocm/pytorch:rocm7.1_ubuntu24.04_py3.13_pytorch_release_2.9.1

SHELL ["/bin/bash", "-c"]

ENV TRITON_PATH=/opt/triton
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH \
    PATH="$ROCM_PATH/bin:$PATH"
ENV PYTHONPATH=$TRITON_PATH

# Install system packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git wget ninja-build cmake python3-pip python3-dev build-essential jq \
    libhwloc-dev libhwloc15 hwloc-nox libnuma-dev libdwarf-dev libzstd-dev && \
    rm -rf /var/lib/apt/lists/*

# Create groups if they don't exist
RUN groupadd -r video 2>/dev/null || true && \
    groupadd -r render 2>/dev/null || true

# Install Python packages with pip
RUN pip3 install --upgrade pip && \
    pip3 install wheel jupyter pytest

# Clone and install Triton
RUN cd /opt && \
    git clone https://github.com/triton-lang/triton.git $TRITON_PATH && \
    cd $TRITON_PATH && \
    git checkout 715f6b1d442601436bf8d462db6ff8e17aec8cfb && \
    pip3 install -e .

# Set up workspace
WORKDIR /workspace
