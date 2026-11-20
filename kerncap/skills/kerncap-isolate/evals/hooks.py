# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Install Kerncap for the `kerncap-isolate` behavior case.

Prompts and expectations live in ``evals.json``; this file holds only the
environment plumbing the dataset format cannot express. The runner calls
``setup_session`` once before the cases run.

When Kerncap is already importable by ``python3`` on the runner it is used as is.
Otherwise it is installed into a throwaway venv whose ``bin`` is put first on
``PATH``, which the agent inherits. Inside the IntelliKit repo the venv gets the
checked-out sources, so a pull request is graded against its own code; in a
vendored copy (the amd/skills catalog) it gets IntelliKit ``main``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

PACKAGE = "kerncap"
GIT_SOURCE = "git+https://github.com/AMDResearch/intellikit.git#subdirectory=kerncap"

# Building from source can compile C++ (KernelDB, the HSA tool library), which
# takes several minutes on a cold runner; 30 minutes leaves room for a slow mirror.
INSTALL_TIMEOUT_S = 30 * 60


def _local_source() -> Path | None:
    """The tool's directory when this file sits in an IntelliKit checkout."""
    # <tool>/skills/<skill>/evals/hooks.py -> <tool>
    tool_dir = Path(__file__).resolve().parents[3]
    if tool_dir.name == PACKAGE and (tool_dir / "pyproject.toml").is_file():
        return tool_dir
    return None


def setup_session(cache_dir: str) -> dict[str, str]:
    """Make the tool importable and its commands runnable for the agent."""
    python = shutil.which("python3")
    if python is None:
        msg = "python3 is not on PATH; the runner needs Python 3.10 or later."
        raise RuntimeError(msg)
    # Run the probe outside the repo: from the IntelliKit root, the bare tool
    # directory would import as a namespace package and look installed.
    probe = subprocess.run(
        [python, "-c", f"import {PACKAGE}"],
        check=False,
        capture_output=True,
        cwd=cache_dir,
    )
    if probe.returncode == 0:
        return {}

    venv_dir = Path(cache_dir) / "venv"
    subprocess.run([python, "-m", "venv", str(venv_dir)], check=True)
    source = _local_source() or GIT_SOURCE
    subprocess.run(
        [str(venv_dir / "bin" / "pip"), "install", "--quiet", str(source)],
        check=True,
        timeout=INSTALL_TIMEOUT_S,
    )
    os.environ["VIRTUAL_ENV"] = str(venv_dir)
    os.environ["PATH"] = f"{venv_dir / 'bin'}{os.pathsep}{os.environ['PATH']}"
    return {}
