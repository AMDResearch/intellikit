# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Extract and replay a GPU kernel from the mini_pipeline example.

Demonstrates the full kerncap workflow using the Python API:
  1. Compile the mini_pipeline HIP application
  2. Profile it to find hot kernels
  3. Extract a target kernel into a standalone reproducer
  4. Replay the captured kernel in isolation
  5. Validate the reproducer for correctness

Prerequisites:
  - ROCm installed (hipcc, rocprofv3 on PATH)
  - AMD GPU (MI300+ recommended)
  - kerncap installed: pip install -e kerncap/

Usage (from `kerncap/` package root):
  python3 examples/01_extract_and_replay/extract_and_replay.py
  python3 examples/01_extract_and_replay/extract_and_replay.py --kernel histogram_atomic
  python3 examples/01_extract_and_replay/extract_and_replay.py --kernel vector_add --iterations 50
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from kerncap import Kerncap


def compile_mini_pipeline(source: Path, build_dir: Path) -> Path:
    """Compile mini_pipeline.hip and return the path to the binary."""
    if not shutil.which("hipcc"):
        print("ERROR: hipcc not found. Install ROCm and ensure hipcc is on PATH.")
        sys.exit(1)

    binary = build_dir / "mini_pipeline"
    print(f"Compiling {source.name} ...")
    result = subprocess.run(
        ["hipcc", "-O2", "-g", "-o", str(binary), str(source)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        sys.exit(1)

    print(f"  Built: {binary}")
    return binary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and replay a kernel from the mini_pipeline example"
    )
    parser.add_argument(
        "--kernel",
        default="vector_scale",
        help="Kernel name to extract (default: vector_scale)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of replay iterations for benchmarking (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for the reproducer (default: temp directory)",
    )
    args = parser.parse_args()

    examples_dir = Path(__file__).resolve().parent
    hip_source = examples_dir / "mini_pipeline.hip"
    if not hip_source.exists():
        print(f"ERROR: {hip_source} not found")
        sys.exit(1)

    kc = Kerncap()

    with tempfile.TemporaryDirectory(prefix="kerncap_example_") as tmpdir:
        build_dir = Path(tmpdir) / "build"
        build_dir.mkdir()
        binary = compile_mini_pipeline(hip_source, build_dir)

        # --- 1. Profile ---
        print("\n--- Profile ---")
        profile = kc.profile([str(binary)])
        print(f"  Found {len(profile)} kernel(s):")
        for k in profile[:10]:
            print(f"    {k.name}: {k.total_duration_ns / 1e6:.2f} ms ({k.percentage:.1f}%)")

        matching = [k for k in profile if args.kernel in k.name]
        if not matching:
            print(f"\nERROR: kernel '{args.kernel}' not found in profile.")
            print("Available kernels:", ", ".join(k.name for k in profile))
            sys.exit(1)

        target = matching[0].name
        print(f"\n  Target kernel: {target}")

        # --- 2. Extract ---
        output_dir = args.output or str(Path(tmpdir) / "reproducer")
        print(f"\n--- Extract (kernel={args.kernel}) ---")
        result = kc.extract(
            kernel_name=args.kernel,
            cmd=[str(binary)],
            source_dir=str(build_dir),
            output=output_dir,
        )
        print(f"  Reproducer: {result.output_dir}")
        print(f"  Has source: {result.has_source}")

        # --- 3. Replay ---
        print(f"\n--- Replay (iterations={args.iterations}) ---")
        replay = kc.replay(result.output_dir, iterations=args.iterations)
        if replay.returncode != 0:
            print(f"  Replay failed (rc={replay.returncode}):")
            print(f"  {replay.stderr}")
            sys.exit(1)

        if replay.timing_us is not None:
            print(f"  Average kernel time: {replay.timing_us:.1f} us")
        else:
            print(f"  Replay output:\n{replay.stdout}")

        # --- 4. Validate ---
        print("\n--- Validate ---")
        validation = kc.validate(result.output_dir)
        print(f"  Passed: {validation.passed}")
        if not validation.passed:
            print(f"  Details: {validation.details}")

        # Keep the reproducer around if the user specified --output
        if args.output:
            print(f"\nReproducer saved to: {result.output_dir}")
            print("To replay manually:")
            print(f"  cd {result.output_dir} && make run")


if __name__ == "__main__":
    main()
