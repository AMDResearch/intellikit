# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Per-metric microbenchmarks for metrix.

Each test module exercises a specific GPU subsystem (HBM, cache, LDS,
compute) with purpose-built HIP kernels and validates that metrix-derived
metrics fall within expected ranges on real hardware.
"""
