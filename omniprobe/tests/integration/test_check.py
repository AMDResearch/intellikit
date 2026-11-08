# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""`omniprobe-ik check` against a real Omniprobe install found only through PATH."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .conftest import skip_no_hipcc, skip_no_omniprobe


def _check(path: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "ROCM_PATH")}
    env["PATH"] = path
    code = "import sys; from omniprobe_ik.cli.main import main; sys.argv[1:] = ['check', '--json']; sys.exit(main())"
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120
    )


@skip_no_omniprobe
@skip_no_hipcc
def test_install_on_path_alone_is_usable():
    """PATH holding <prefix>/omniprobe/bin and ROCm's bin is the whole requirement."""
    exe = Path(shutil.which("omniprobe"))
    rocm_bin = Path(shutil.which("hipcc")).resolve().parent
    result = _check(f"{exe.parent}:{rocm_bin}:/usr/bin:/bin")
    report = json.loads(result.stdout)
    assert result.returncode == 0, report["problems"]
    assert report["plugins"] == {"memory": True, "basic_blocks": True}
    assert report["python_api"] and Path(report["python_api"], "__init__.py").is_file()
    assert report["gpu_arch"].startswith("gfx")
    assert report["omniprobe_rocm"] == report["compiler_rocm"]


def test_no_omniprobe_on_path_fails_with_install_pointer():
    result = _check("/usr/bin:/bin")
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["executable"] is None
    assert report["install_doc"] in " ".join(report["problems"])
