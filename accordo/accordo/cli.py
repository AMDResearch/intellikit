# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Accordo CLI -- thin wrapper around the Python API for subprocess invocation.

Usage:
    accordo validate \\
        --kernel-name "reduce_sum" \\
        --ref-binary ./ref \\
        --opt-binary ./opt \\
        [--tolerance 1e-6] \\
        [--timeout 30] \\
        [--working-dir .] \\
        [--kernel-args "input:const float*,output:float*"] \\
        [--log-level WARNING]

Outputs JSON to stdout.  Logging goes to stderr so it never contaminates the
JSON payload.  Exit code is 0 when the tool completes (regardless of validation
pass/fail); non-zero only on fatal errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Tuple


def _parse_kernel_args(raw: str) -> List[Tuple[str, str]]:
    """Parse ``name:type,name:type,...`` into a list of (name, type) tuples."""
    pairs: List[Tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":", 1)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid kernel-arg format '{item}'. Expected 'name:type'."
            )
        pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def _build_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("validate", help="Validate kernel correctness")
    p.add_argument("--kernel-name", required=True, help="Kernel function name to intercept")
    p.add_argument("--ref-binary", required=True, help="Reference binary path/command")
    p.add_argument("--opt-binary", required=True, help="Optimized binary path/command")
    p.add_argument("--tolerance", type=float, default=1e-6, help="Absolute tolerance (default: 1e-6)")
    p.add_argument("--timeout", type=int, default=30, help="Timeout per snapshot in seconds (default: 30)")
    p.add_argument("--working-dir", default=".", help="Working directory for binary execution (default: .)")
    p.add_argument("--kernel-args", type=_parse_kernel_args, default=None,
                   help="Manual kernel args as 'name:type,name:type,...'")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging level (default: WARNING)")


def _run_validate(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        from .validator import Accordo
        from .exceptions import AccordoError
    except ImportError as exc:
        json.dump({"error": f"Failed to import accordo: {exc}"}, sys.stdout)
        return 1

    # The C++ interception library and KernelDB write to stdout.  Redirect
    # the fd-level stdout to stderr for the duration of the validation so
    # only the final JSON reaches the real stdout.
    real_stdout_fd = os.dup(1)
    os.dup2(2, 1)

    try:
        validator = Accordo(
            binary=args.ref_binary,
            kernel_name=args.kernel_name,
            kernel_args=args.kernel_args,
            working_directory=args.working_dir,
            log_level=args.log_level,
        )

        ref_snapshot = validator.capture_snapshot(
            binary=args.ref_binary, timeout_seconds=args.timeout,
        )
        opt_snapshot = validator.capture_snapshot(
            binary=args.opt_binary, timeout_seconds=args.timeout,
        )
        result = validator.compare_snapshots(
            ref_snapshot, opt_snapshot, tolerance=args.tolerance,
        )

        mismatches_serialized = []
        for m in (result.mismatches or []):
            mismatches_serialized.append({
                "arg_index": m.arg_index,
                "arg_name": m.arg_name,
                "arg_type": m.arg_type,
                "max_difference": m.max_difference,
                "mean_difference": m.mean_difference,
            })

        output = {
            "is_valid": result.is_valid,
            "num_arrays_validated": result.num_arrays_validated,
            "num_mismatches": result.num_mismatches,
            "summary": result.summary(),
            "error_message": result.error_message,
            "matched_arrays": result.matched_arrays or {},
            "mismatches": mismatches_serialized,
        }

    except AccordoError as exc:
        output = {"error": str(exc)}
    except Exception as exc:
        output = {"error": f"Unexpected error: {exc}"}

    # Restore real stdout and write JSON there.  Use os._exit() afterwards
    # to prevent KernelDB / atexit handlers from printing to stdout.
    os.dup2(real_stdout_fd, 1)
    os.close(real_stdout_fd)
    real_stdout = os.fdopen(1, "w", closefd=False)

    json.dump(output, real_stdout, indent=2)
    real_stdout.write("\n")
    real_stdout.flush()

    exit_code = 0 if "error" not in output else 1
    os._exit(exit_code)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="accordo",
        description="Accordo: GPU kernel correctness validation",
    )
    subparsers = parser.add_subparsers(dest="command")
    _build_validate_parser(subparsers)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "validate":
        sys.exit(_run_validate(args))


if __name__ == "__main__":
    main()
