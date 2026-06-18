#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
PyTorch example: element-wise addition on GPU (ROCm / CUDA).

Structured like the numbered HIP examples: small workload, explicit steps,
synchronize, then a checksum line for sanity (used by optional pytest).
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=int,
        default=4096,
        metavar="N",
        help="Square matrix side length (default: 4096).",
    )
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("Install PyTorch (ROCm build on AMD GPUs): pip install torch", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("No GPU visible to PyTorch (set HIP/CUDA device).", file=sys.stderr)
        return 1

    n = max(64, args.size)
    device = "cuda"

    print("=" * 60)
    print("Example: PyTorch tensor add (element-wise)")
    print("=" * 60)
    print(f"Matrix size: {n} x {n}")
    print("Step 1: Allocating random tensors on GPU...")
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    print("Step 2: Running c = a + b ...")
    c = a + b
    print("Step 3: Synchronizing and printing checksum...")
    torch.cuda.synchronize()
    print(f"sum(a+b) = {float(c.sum().cpu()):.6f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
