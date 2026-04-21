#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Kerncap driver: profile the sibling PyTorch script; optionally extract/replay/validate.

Usage:
  python3 profile_with_kerncap.py
  python3 profile_with_kerncap.py --kernel <substring-from-profile>
  python3 profile_with_kerncap.py --kernel <substring> --output ./my_reproducer
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from kerncap import Kerncap


def _resolve_workload(here: Path) -> tuple[Path, str]:
    for name, label in (
        ("tensor_add.py", "tensor add"),
        ("tensor_matmul.py", "matrix multiply"),
    ):
        p = here / name
        if p.is_file():
            return p, label
    print(f"No tensor_add.py or tensor_matmul.py in {here}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    here = Path(__file__).resolve().parent
    script, workload_label = _resolve_workload(here)

    parser = argparse.ArgumentParser(
        description=f"Profile PyTorch {workload_label} with kerncap; optionally extract a kernel.",
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help="Substring of kernel name from profile; if set, runs extract → replay → validate",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Replay iterations when --kernel is set (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Reproducer output dir when --kernel is set (default: temp directory)",
    )
    args = parser.parse_args()

    cmd = [sys.executable, str(script)]
    kc = Kerncap()

    print("=" * 60)
    print(f"Kerncap example: PyTorch {workload_label}")
    print("=" * 60)
    print("--- Profile ---")
    print(f"  Command: {' '.join(cmd)}\n")
    profile = kc.profile(cmd)
    print(f"  Found {len(profile)} kernel(s):")
    for k in profile[:20]:
        print(f"    {k.name}: {k.total_duration_ns / 1e6:.2f} ms ({k.percentage:.1f}%)")
    if len(profile) > 20:
        print(f"    ... ({len(profile) - 20} more)")

    if not args.kernel:
        print("\n  Profile only. Re-run with --kernel <substring> to extract a kernel from the list.")
        return 0

    matching = [k for k in profile if args.kernel in k.name]
    if not matching:
        print(f"\nERROR: kernel substring '{args.kernel}' not found in profile.", file=sys.stderr)
        print("Available:", ", ".join(k.name for k in profile[:30]), file=sys.stderr)
        return 1

    target = matching[0].name
    print(f"\n  Target kernel: {target}")

    def extract_replay_validate(output_dir: str) -> int:
        print(f"\n--- Extract (kernel substring={args.kernel!r}) ---")
        result = kc.extract(
            kernel_name=args.kernel,
            cmd=cmd,
            source_dir=str(here),
            output=output_dir,
        )
        print(f"  Reproducer: {result.output_dir}")
        print(f"  Has source: {result.has_source}")

        print(f"\n--- Replay (iterations={args.iterations}) ---")
        replay = kc.replay(result.output_dir, iterations=args.iterations)
        if replay.returncode != 0:
            print(f"  Replay failed (rc={replay.returncode}):", file=sys.stderr)
            print(replay.stderr, file=sys.stderr)
            return 1
        if replay.timing_us is not None:
            print(f"  Average kernel time: {replay.timing_us:.1f} us")
        else:
            print(f"  Replay output:\n{replay.stdout}")

        print("\n--- Validate ---")
        validation = kc.validate(result.output_dir)
        print(f"  Passed: {validation.passed}")
        if not validation.passed:
            print(f"  Details: {validation.details}")
            return 1

        if args.output:
            print(f"\nReproducer saved to: {result.output_dir}")
            print("To replay manually:")
            print(f"  cd {result.output_dir} && make run")
        return 0

    if args.output:
        return extract_replay_validate(args.output)
    with tempfile.TemporaryDirectory(prefix="kerncap_pytorch_ex_") as tmpdir:
        return extract_replay_validate(str(Path(tmpdir) / "reproducer"))


if __name__ == "__main__":
    raise SystemExit(main())
