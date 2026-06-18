#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
PyTorch example: matrix multiply on GPU (ROCm / CUDA), C = A @ B.

Same narrative as other IntelliKit packages. Accordo validation uses the HIP
pair in ``validate.py``; this script is optional cross-tool parity.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=int,
        default=2048,
        metavar="N",
        help="Square matrix side length for A, B (default: 2048).",
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
    print("Example: PyTorch matrix multiply (GEMM-style)")
    print("=" * 60)
    print(f"Matrix size: {n} x {n}")
    print("Step 1: Allocating random matrices on GPU...")
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    print("Step 2: Running C = A @ B ...")
    c = a @ b
    print("Step 3: Synchronizing and printing checksum...")
    torch.cuda.synchronize()
    print(f"sum(A@B) = {float(c.sum().cpu()):.6f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
