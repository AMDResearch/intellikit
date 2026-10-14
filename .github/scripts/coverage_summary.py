#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Aggregate per-package Cobertura XML reports into a single markdown table.
# Usage: coverage_summary.py <directory-of-coverage-xml-files>

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

EXPECTED_ARGC = 2


def read_report(path: Path) -> tuple[str, int, int] | None:
    """Extract ``(package, covered, valid)`` from one Cobertura XML file.

    Args:
        path: Report to read. Named ``coverage-<package>.xml`` by the workflow.

    Returns:
        The package name with its covered and total line counts, or ``None``
        if the file could not be parsed.
    """
    try:
        # Trusted input: these files are produced by pytest-cov in this same
        # workflow run, not supplied by a third party.
        root = ET.parse(path).getroot()  # noqa: S314
        covered = int(root.get("lines-covered", "0"))
        valid = int(root.get("lines-valid", "0"))
    except (ET.ParseError, ValueError) as exc:
        print(f"warning: could not parse {path.name}: {exc}", file=sys.stderr)
        return None
    return path.stem.removeprefix("coverage-"), covered, valid


def percent(covered: int, valid: int) -> float:
    """Return ``covered`` as a percentage of ``valid``, or 0.0 when empty."""
    return 100.0 * covered / valid if valid else 0.0


def main() -> int:
    """Print a markdown coverage table and return a process exit code."""
    if len(sys.argv) != EXPECTED_ARGC:
        print("usage: coverage_summary.py <coverage-dir>", file=sys.stderr)
        return 2

    reports = sorted(Path(sys.argv[1]).glob("coverage-*.xml"))
    rows = [row for row in (read_report(path) for path in reports) if row is not None]

    print("## Test Coverage\n")

    if not rows:
        print("No coverage reports were produced.")
        return 0

    total_covered = sum(covered for _, covered, _ in rows)
    total_valid = sum(valid for _, _, valid in rows)

    print("| Package | Lines | Covered | Coverage |")
    print("| --- | ---: | ---: | ---: |")
    for package, covered, valid in rows:
        print(f"| {package} | {valid} | {covered} | {percent(covered, valid):.1f}% |")
    print(
        f"| **Overall** | **{total_valid}** | **{total_covered}** "
        f"| **{percent(total_covered, total_valid):.1f}%** |"
    )
    print(
        "\nMeasured on the editable install leg only. Packages not listed were "
        "unchanged in this run and did not execute."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
