# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Install Accordo for the `accordo-validation` behavior case.

Prompts and expectations live in ``evals.json``; this file holds only the
environment plumbing the dataset format cannot express. The runner calls
``setup_session`` once before the cases run.

Inside the IntelliKit repo the checked-out sources are always installed into a
throwaway venv, so a pull request is graded against its own code. In a vendored
copy (the amd/skills catalog) a working Accordo already on the runner is reused;
otherwise IntelliKit ``main`` is installed into the venv. The venv's ``bin`` is
put first on ``PATH``, which the agent inherits.

Accordo builds KernelDB, which compiles against libdwarf, libelf, and libzstd.
IntelliKit's dev image installs their Ubuntu packages; a CI runner may have
neither the headers nor the right to install them. When any of them is missing,
the same Ubuntu packages are downloaded with a private apt index, unpacked into
the cache, and pointed to through the compiler and loader search paths, so
nothing on the runner changes and no root is needed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

PACKAGE = "accordo"
GIT_SOURCE = "git+https://github.com/AMDResearch/intellikit.git#subdirectory=accordo"
# Accordo imports without KernelDB (it only flags it as unavailable), so
# check KernelDB too.
PROBE = "import accordo, kerneldb"

# Building from source can compile C++ (KernelDB, the HSA tool library), which
# takes several minutes on a cold runner; 30 minutes leaves room for a slow mirror.
INSTALL_TIMEOUT_S = 30 * 60

# The development packages KernelDB builds against (docker/Dockerfile installs
# the same set), and the directories its CMake searches for libdwarf.h.
NATIVE_DEV_PACKAGES = ("libdwarf-dev", "libelf-dev", "libzstd-dev")
NATIVE_RUNTIME_PREFIXES = ("libdwarf", "libelf", "libzstd")
LIBDWARF_HEADER_DIRS = (
    "/usr/include",
    "/usr/include/libdwarf",
    "/usr/include/libdwarf-0",
    "/usr/local/include/libdwarf",
    "/usr/local/include/libdwarf-0",
)
NATIVE_HEADERS = {
    "libdwarf.h": LIBDWARF_HEADER_DIRS,
    "libelf.h": ("/usr/include", "/usr/local/include"),
    "zstd.h": ("/usr/include", "/usr/local/include"),
}


def _local_source() -> Path | None:
    """The tool's directory when this file sits in an IntelliKit checkout."""
    # <tool>/skills/<skill>/evals/hooks.py -> <tool>
    tool_dir = Path(__file__).resolve().parents[3]
    if tool_dir.name == PACKAGE and (tool_dir / "pyproject.toml").is_file():
        return tool_dir
    return None


def _prepend_env(name: str, *paths: Path) -> None:
    """Put ``paths`` ahead of whatever ``name`` already holds."""
    parts = [str(p) for p in paths]
    if os.environ.get(name):
        parts.append(os.environ[name])
    os.environ[name] = os.pathsep.join(parts)


def _unpack_native_deps(cache_dir: str) -> None:
    """Unpack KernelDB's Ubuntu build dependencies into the cache, unprivileged."""
    apt_get, apt_cache, dpkg = (shutil.which(tool) for tool in ("apt-get", "apt-cache", "dpkg"))
    if apt_get is None or apt_cache is None or dpkg is None:
        msg = (
            "KernelDB's build headers are missing and this runner has no apt-get/dpkg to "
            f"fetch them; install {' '.join(NATIVE_DEV_PACKAGES)} on the runner."
        )
        raise RuntimeError(msg)

    root = Path(cache_dir) / "native"
    lists, archives, debs, sysroot = (
        root / "lists",
        root / "archives",
        root / "debs",
        root / "sysroot",
    )
    for d in (lists / "partial", archives / "partial", debs, sysroot):
        d.mkdir(parents=True, exist_ok=True)
    # A private index and cache: `update` and `download` then need no root and
    # leave the runner's own apt state alone.
    apt_opts = [
        "-o",
        f"Dir::State::Lists={lists}",
        "-o",
        f"Dir::Cache={root}",
        "-o",
        f"Dir::Cache::archives={archives}",
        "-o",
        "Debug::NoLocking=1",
    ]
    # Not checked: one unreachable or unsigned third-party source fails the whole
    # update even when the distribution's own indexes arrived. The download
    # below fails clearly if the packages are really unavailable.
    subprocess.run([apt_get, *apt_opts, "-qq", "update"], check=False, timeout=INSTALL_TIMEOUT_S)

    # The -dev packages' .so links point into their runtime packages, whose
    # names change between releases (libdwarf1 vs libdwarf1t64), so read them.
    depends = subprocess.run(
        [
            apt_cache,
            *apt_opts,
            "depends",
            "--no-recommends",
            "--no-suggests",
            *NATIVE_DEV_PACKAGES,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    runtime = {
        name
        for name in re.findall(r"^\s+Depends: (\S+)$", depends, re.MULTILINE)
        if name.startswith(NATIVE_RUNTIME_PREFIXES) and name not in NATIVE_DEV_PACKAGES
    }
    subprocess.run(
        [apt_get, *apt_opts, "-qq", "download", *NATIVE_DEV_PACKAGES, *sorted(runtime)],
        check=True,
        cwd=debs,
        timeout=INSTALL_TIMEOUT_S,
    )
    for deb in sorted(debs.glob("*.deb")):
        subprocess.run([dpkg, "-x", str(deb), str(sysroot)], check=True)

    include = sysroot / "usr" / "include"
    libdir = next(sysroot.glob("usr/lib/*/libdwarf.so")).parent
    # KernelDB finds libdwarf.h through CMAKE_INCLUDE_PATH and the other headers
    # through the compiler's path (libdwarf's own directory included, whichever
    # include style a source uses), links `dwarf elf zstd` by bare name, and the
    # agent's tools load the libraries at run time.
    header_dirs = [d for d in (include / "libdwarf-0", include / "libdwarf", include) if d.is_dir()]
    _prepend_env("CMAKE_INCLUDE_PATH", *header_dirs)
    _prepend_env("CPATH", *header_dirs)
    _prepend_env("LIBRARY_PATH", libdir)
    _prepend_env("LD_LIBRARY_PATH", libdir)


def setup_session(cache_dir: str) -> dict[str, str]:
    """Make the tool importable and its commands runnable for the agent."""
    python = shutil.which("python3")
    if python is None:
        msg = "python3 is not on PATH; the runner needs Python 3.10 or later."
        raise RuntimeError(msg)
    local = _local_source()
    if local is None:
        # Vendored copy: reuse a working installation if the runner has one.
        # Probe outside any checkout, where a bare tool directory would import
        # as a namespace package and look installed.
        probe = subprocess.run(
            [python, "-c", PROBE],
            check=False,
            capture_output=True,
            cwd=cache_dir,
        )
        if probe.returncode == 0:
            return {}

    headers_present = all(
        any(Path(d, header).is_file() for d in dirs) for header, dirs in NATIVE_HEADERS.items()
    )
    if not headers_present:
        _unpack_native_deps(cache_dir)

    venv_dir = Path(cache_dir) / "venv"
    subprocess.run([python, "-m", "venv", str(venv_dir)], check=True)
    source = local or GIT_SOURCE
    subprocess.run(
        [str(venv_dir / "bin" / "pip"), "install", "--quiet", str(source)],
        check=True,
        timeout=INSTALL_TIMEOUT_S,
    )
    os.environ["VIRTUAL_ENV"] = str(venv_dir)
    os.environ["PATH"] = f"{venv_dir / 'bin'}{os.pathsep}{os.environ['PATH']}"
    return {}
