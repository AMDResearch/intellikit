"""kerncap — Kernel extraction and isolation tool for HIP and Triton on AMD GPUs."""

import os
import pathlib
import shutil
import site
import sys


def _get_replay_path() -> str:
    """Return the absolute path to the installed kerncap-replay binary.

    Search order mirrors _get_lib_path(): package bin/ dir, site-packages,
    then PATH.
    """
    bin_name = "kerncap-replay"
    pkg_dir = pathlib.Path(__file__).resolve().parent

    candidates = [
        pkg_dir / "bin" / bin_name,
        pkg_dir.parent / "bin" / bin_name,
    ]

    sp_dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    user_sp = getattr(site, "getusersitepackages", lambda: None)()
    if user_sp:
        sp_dirs.append(user_sp)

    for sp in sp_dirs:
        candidates.append(pathlib.Path(sp) / "kerncap" / "bin" / bin_name)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    found = shutil.which(bin_name)
    if found:
        return found

    searched = [str(c) for c in candidates]
    raise FileNotFoundError(
        f"Could not locate {bin_name}. "
        f"Ensure the package was built correctly (pip install .) "
        f"or ensure kerncap-replay is on PATH.\n"
        f"Searched: {searched}"
    )


def _get_lib_path() -> str:
    """Return the absolute path to the installed libkerncap.so.

    The shared library is installed alongside the Python package by
    scikit-build-core.  We search in order:
      1. KERNCAP_LIB_PATH environment variable (explicit override)
      2. Relative to this file (works when importing the installed package)
      3. Installed site-packages (works when the local source tree shadows
         the installed package, e.g. running tests from the repo root)
    """
    env_path = os.environ.get("KERNCAP_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    lib_name = "libkerncap.so"
    pkg_dir = pathlib.Path(__file__).resolve().parent

    # Check relative to this __init__.py (normal installed case)
    candidates = [
        pkg_dir / "lib" / lib_name,
        pkg_dir / lib_name,
    ]

    # Also check site-packages in case the local source tree is shadowing
    # the installed package (common when running tests from the repo root)
    sp_dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    user_sp = getattr(site, "getusersitepackages", lambda: None)()
    if user_sp:
        sp_dirs.append(user_sp)

    for sp in sp_dirs:
        candidates.append(pathlib.Path(sp) / "kerncap" / "lib" / lib_name)
        candidates.append(pathlib.Path(sp) / "kerncap" / lib_name)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    searched = [str(c) for c in candidates]
    raise FileNotFoundError(
        f"Could not locate {lib_name}. "
        f"Ensure the package was built correctly (pip install .) "
        f"or set KERNCAP_LIB_PATH to the library location.\n"
        f"Searched: {searched}"
    )
