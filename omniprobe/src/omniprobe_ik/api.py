#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Python API for Omniprobe GPU memory and basic-block analysis.

Wraps Omniprobe's Python API (``omniprobe.api``) to provide memory coalescing
analysis and basic-block execution counts through IntelliKit's standard interface.

Omniprobe must be installed separately: put ``<prefix>/omniprobe/bin`` of an
Omniprobe install on ``PATH``. The executable, its instrumentation plugins and its
Python API are all found from there (see :data:`INSTALL_DOC` and
:func:`check_installation`).

Omniprobe only analyzes kernels that were compiled with one of its LLVM
instrumentation plugins. Build the application with the flags returned by
:func:`instrumentation_flags` first; a binary built without them is reported
as :class:`NoInstrumentedKernelsError`, not as an empty result.

Example::

    from omniprobe_ik import Omniprobe, instrumentation_flags

    # hipcc my_gpu_app.hip -o my_gpu_app -g <flags>
    flags = instrumentation_flags("memory")

    probe = Omniprobe()
    result = probe.analyze_memory("./my_gpu_app")
    for kernel in result.kernels:
        for access in kernel.accesses:
            print(f"{access.source_file}:{access.line} bank_conflicts={access.bank_conflicts}")
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compile-time instrumentation
# ---------------------------------------------------------------------------

# Which Omniprobe LLVM pass plugin each analysis needs. The analyzers consume
# the messages a specific plugin injects, so a kernel compiled with the wrong
# plugin is as invisible to an analysis as one compiled with none.
ANALYSIS_PLUGINS = {
    "memory": "libAMDGCNSubmitAddressMessages-rocm.so",
    "basic_blocks": "libAMDGCNSubmitBBStart-rocm.so",
}


# How to build and install Omniprobe; every "not found" message points here.
INSTALL_DOC = "https://github.com/AMDResearch/intellikit/blob/main/docs/getting-started/installing-omniprobe.md"
_NOT_INSTALLED = (
    "Put <prefix>/omniprobe/bin of an Omniprobe install on PATH (in an MCP setup, in the "
    "server's 'env' configuration); to install Omniprobe see " + INSTALL_DOC
)


class NoInstrumentedKernelsError(RuntimeError):
    """Omniprobe ran but reported no instrumented kernel.

    Almost always means the application was compiled without the Omniprobe
    instrumentation plugin; see :func:`instrumentation_flags`.
    """


class KernelFilterMatchedNothingError(NoInstrumentedKernelsError):
    """The kernel filter matched none of the dispatched instrumented kernels.

    The application is instrumented; the filter is wrong. ``dispatched`` holds
    the names Omniprobe listed as candidates.
    """

    def __init__(self, message: str, dispatched: list[str]):
        super().__init__(message)
        self.dispatched = dispatched


def _normalize_analysis(analysis: str) -> str:
    key = analysis.replace("-", "_")
    if key not in ANALYSIS_PLUGINS:
        choices = ", ".join(sorted(ANALYSIS_PLUGINS))
        raise ValueError(f"unknown analysis {analysis!r}; expected one of: {choices}")
    return key


def find_plugin_dir(omniprobe_path: Optional[str] = None) -> Path:
    """Locate the directory holding Omniprobe's instrumentation plugins.

    Honors ``OMNIPROBE_PLUGIN_DIR`` if set. Otherwise looks in ``lib/plugins``
    next to the ``omniprobe`` executable, which is where both Omniprobe's build
    tree and its install tree put them. The executable's path is tried as found
    first and then with symlinks resolved: in a build tree ``bin/omniprobe`` is
    a symlink into the source tree, so resolving it first would look in the
    wrong place.

    Raises:
        FileNotFoundError: If no plugin directory can be found.
    """
    override = os.environ.get("OMNIPROBE_PLUGIN_DIR")
    if override:
        return Path(override)

    exe = omniprobe_path or shutil.which("omniprobe")
    if exe is None:
        raise FileNotFoundError(
            "omniprobe executable not found on PATH, so its instrumentation plugins "
            f"cannot be located. {_NOT_INSTALLED}"
        )

    candidates = [Path(exe).parent.parent / "lib" / "plugins"]
    candidates.append(Path(exe).resolve().parent.parent / "lib" / "plugins")
    for candidate in candidates:
        if any((candidate / name).is_file() for name in ANALYSIS_PLUGINS.values()):
            return candidate

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"no Omniprobe instrumentation plugins found (searched {searched}); "
        "set OMNIPROBE_PLUGIN_DIR to the directory holding libAMDGCNSubmit*-rocm.so"
    )


def detect_offload_arch() -> Optional[str]:
    """Return the gfx architecture of the first GPU ``rocminfo`` reports, or *None*."""
    rocminfo = shutil.which("rocminfo")
    if rocminfo is None:
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        candidate = Path(rocm_path) / "bin" / "rocminfo"
        rocminfo = str(candidate) if candidate.is_file() else None
    if rocminfo is None:
        return None

    try:
        output = subprocess.run([rocminfo], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    match = re.search(r"^\s*Name:\s+(gfx[0-9a-z]+)\s*$", output, re.MULTILINE)
    return match.group(1) if match else None


def instrumentation_flags(
    analysis: str = "memory",
    offload_arch: Optional[str] = None,
    omniprobe_path: Optional[str] = None,
    rdc: bool = False,
) -> list[str]:
    """Return the ``hipcc`` flags that make kernels visible to an Omniprobe analysis.

    Omniprobe works by cloning each kernel at compile time and injecting
    instrumentation into the clone; at run time it swaps the clone in. Kernels
    compiled without these flags have no clone, and Omniprobe reports nothing
    for them. Add ``-g`` as well to get source-line attribution.

    Args:
        analysis: ``"memory"`` or ``"basic_blocks"`` (``"basic-blocks"`` is
            accepted too). Each needs a different plugin.
        offload_arch: GPU architecture, e.g. ``"gfx90a"``. Detected with
            ``rocminfo`` if omitted.
        omniprobe_path: Path to the ``omniprobe`` executable, used to find the
            plugins. Defaults to the one on ``PATH``.
        rdc: Add ``-fgpu-rdc``. Instrumentation does not need it; set it only
            when the application already builds with relocatable device code,
            and keep its build mode otherwise. Measured on gfx90a (Omniprobe
            13ca707): every kernel -- including ``static`` and
            anonymous-namespace ones, across translation units -- is matched
            with or without RDC. Omniprobe older than 3003058 fails to link
            RDC builds of two or more translation units (``duplicate symbol:
            __hip_cuid_...``).

    Returns:
        A list such as ``["-fpass-plugin=/.../libAMDGCNSubmitAddressMessages-rocm.so",
        "--offload-arch=gfx90a"]``. With ``rdc=True``, ``-fgpu-rdc`` is added;
        pass it to the link step too when compiling and linking separately.

    Raises:
        ValueError: Unknown analysis.
        FileNotFoundError: The plugins cannot be located.
        RuntimeError: No GPU architecture was given and none could be detected.
    """
    plugin = find_plugin_dir(omniprobe_path) / ANALYSIS_PLUGINS[_normalize_analysis(analysis)]
    if not plugin.is_file():
        raise FileNotFoundError(f"Omniprobe plugin not found: {plugin}")

    arch = offload_arch or detect_offload_arch()
    if arch is None:
        raise RuntimeError(
            "could not detect the GPU architecture with rocminfo; pass offload_arch, e.g. 'gfx90a'"
        )

    flags = [f"-fpass-plugin={plugin}", f"--offload-arch={arch}"]
    if rdc:
        flags.insert(1, "-fgpu-rdc")
    return flags


def _upstream_source_root(omniprobe_path: Optional[str] = None) -> Optional[Path]:
    """The directory holding ``omniprobe/api`` for the given or ``PATH`` executable, or None."""
    exe = omniprobe_path or shutil.which("omniprobe")
    if exe is None:
        return None
    for start in (Path(exe), Path(exe).resolve()):
        for parent in start.parents:
            if (parent / "omniprobe" / "api" / "__init__.py").is_file():
                return parent
    return None


def _add_upstream_source_to_path(omniprobe_path: Optional[str] = None) -> bool:
    """Find Omniprobe's Python API from its executable and put it on ``sys.path``.

    The API is ``<root>/omniprobe/api``: in an install tree ``<root>`` is the
    install prefix (``<prefix>/omniprobe/bin/omniprobe``); in a build tree
    ``bin/omniprobe`` is a symlink to ``<source>/omniprobe/omniprobe``.
    Returns True if ``omniprobe.api`` is importable afterwards.
    """
    import importlib
    import sys

    source_root = _upstream_source_root(omniprobe_path)
    if source_root is None:
        return False
    sys.path.insert(0, str(source_root))
    sys.modules.pop("omniprobe", None)
    try:
        importlib.import_module("omniprobe.api")
    except ImportError:
        sys.path.remove(str(source_root))
        return False
    logger.info("Using Omniprobe's Python API from %s", source_root)
    return True


def omniprobe_rocm(omniprobe_path: Optional[str] = None) -> Optional[Path]:
    """The ROCm installation an Omniprobe install runs against, or *None* if unknown.

    That is the ROCm its core library resolves ``libhsa-runtime64.so`` from -- the
    one its instrumentation plugins were built for, and the same rule Omniprobe's
    launcher uses to pick a ROCm.
    """
    exe = omniprobe_path or shutil.which("omniprobe")
    if exe is None:
        return None
    for root in (Path(exe).parent.parent, Path(exe).resolve().parent.parent):
        core = root / "lib" / "liblogDuration64.so"
        if not core.is_file():
            continue
        try:
            out = subprocess.run(["ldd", str(core)], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        for line in out.stdout.splitlines():
            if "libhsa-runtime64.so" in line and "=>" in line:
                lib = Path(line.split("=>")[1].split()[0]).resolve()
                return lib.parent.parent
    return None


def compiler_rocm() -> Optional[Path]:
    """The ROCm installation the ``hipcc`` on ``PATH`` belongs to, or *None*."""
    hipcc = shutil.which("hipcc")
    return Path(hipcc).resolve().parent.parent if hipcc else None


def check_installation(omniprobe_path: Optional[str] = None) -> dict:
    """Report whether Omniprobe is usable from here, and what is missing if not.

    Checks the ``omniprobe`` executable, the instrumentation plugins, Omniprobe's
    Python API, the GPU architecture (``rocminfo``), and whether Omniprobe, the
    ``hipcc`` on ``PATH`` and ``ROCM_PATH`` (if set) refer to the same ROCm. A
    mismatch is a warning, not a problem: the plugins are built with one ROCm's
    LLVM and LLVM does not promise that another version can load them, but on
    gfx90a with Omniprobe 22c4d95 a trivial kernel compiled by ROCm 7.1.0's and
    6.4.1's hipcc with 7.2.0-built plugins gave the same report as 7.2.0's.

    Returns:
        A dict with ``ok`` (bool: no problems), ``problems`` and ``warnings``
        (lists of str; a ROCm mismatch is a warning), and the fields
        ``executable``, ``executable_runs``, ``plugin_dir``, ``plugins``, ``python_api``,
        ``gpu_arch``, ``omniprobe_rocm``, ``compiler_rocm``, ``rocm_path``.
    """
    import importlib.util

    problems: list[str] = []
    report: dict = {"install_doc": INSTALL_DOC}

    exe = omniprobe_path or shutil.which("omniprobe")
    report["executable"] = exe
    if exe is None:
        problems.append(f"omniprobe executable not found on PATH. {_NOT_INSTALLED}")

    report["executable_runs"] = None
    if exe is not None:
        try:
            run = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=60)
            report["executable_runs"] = run.returncode == 0
            if run.returncode != 0:
                last = (run.stderr.strip().splitlines() or ["no output"])[-1]
                problems.append(
                    f"{exe} --help failed ({last}); its Python requirements may be missing: "
                    "python3 -m pip install -r <omniprobe source>/omniprobe/requirements.txt"
                )
        except (OSError, subprocess.SubprocessError) as e:
            report["executable_runs"] = False
            problems.append(f"{exe} could not be run: {e}")

    report["plugin_dir"] = None
    report["plugins"] = {}
    if exe is not None:
        try:
            plugin_dir = find_plugin_dir(exe)
            report["plugin_dir"] = str(plugin_dir)
            report["plugins"] = {
                analysis: (plugin_dir / name).is_file()
                for analysis, name in ANALYSIS_PLUGINS.items()
            }
            missing = [a for a, present in report["plugins"].items() if not present]
            if missing:
                problems.append(f"instrumentation plugins missing in {plugin_dir}: {missing}")
        except FileNotFoundError as e:
            problems.append(str(e))

    # Located, not imported: asking the question must not run Omniprobe's code.
    api = None
    try:
        spec = importlib.util.find_spec("omniprobe.api")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        api = str(Path(spec.origin).parent)
    elif exe is not None:
        root = _upstream_source_root(exe)
        if root is not None:
            api = str(root / "omniprobe" / "api")
    report["python_api"] = api
    if api is None and exe is not None:
        problems.append(
            f"Omniprobe's Python API was not found next to {exe} (expected "
            "<prefix>/omniprobe/api); the install looks incomplete"
        )

    arch = detect_offload_arch()
    report["gpu_arch"] = arch
    if arch is None:
        problems.append("no GPU architecture from rocminfo: put ROCm's bin directory on PATH")

    op_rocm = omniprobe_rocm(exe) if exe else None
    cc_rocm = compiler_rocm()
    env_rocm = os.environ.get("ROCM_PATH")
    report["omniprobe_rocm"] = str(op_rocm) if op_rocm else None
    report["compiler_rocm"] = str(cc_rocm) if cc_rocm else None
    report["rocm_path"] = env_rocm
    warnings: list[str] = []
    if cc_rocm is None:
        problems.append("hipcc not found on PATH: instrumented builds need ROCm's hipcc")
    elif op_rocm is not None and op_rocm != cc_rocm:
        warnings.append(
            f"ROCm mismatch: Omniprobe was built for {op_rocm} but the hipcc on PATH is from "
            f"{cc_rocm}. Its instrumentation plugins are LLVM pass plugins built with "
            f"{op_rocm}'s LLVM; LLVM gives plugins no stable interface across versions, so "
            "a mismatched compiler may reject or crash in them"
        )
    if env_rocm and op_rocm is not None and Path(env_rocm).resolve() != op_rocm:
        warnings.append(
            f"ROCM_PATH={env_rocm} is not the ROCm Omniprobe was built for ({op_rocm}); "
            "Omniprobe uses ROCM_PATH's llvm-objdump when it has one"
        )

    report["problems"] = problems
    report["warnings"] = warnings
    report["ok"] = not problems
    return report


_NO_ALTERNATIVE = re.compile(r"No instrumented alternative found for (.+?)\s*$", re.MULTILINE)


def parse_uninstrumented_kernels(log: str) -> list[str]:
    """Kernels Omniprobe reported as dispatched without an instrumented clone.

    The HIP runtime's own helper kernels (``__amd_rocclr_*``, e.g. buffer
    copies) are never instrumented and are left out. Order is first appearance;
    duplicates are dropped.
    """
    seen: list[str] = []
    for name in _NO_ALTERNATIVE.findall(log):
        name = name.strip()
        if name.startswith("__amd_rocclr_") or name in seen:
            continue
        seen.append(name)
    return seen


class _CapturingSubprocess:
    """Stands in for the ``subprocess`` module inside upstream's API module.

    Upstream's ``Omniprobe._run`` captures omniprobe's output and discards it,
    so the "No instrumented alternative found" lines never reach the caller.
    This proxy records what ``run`` returned. Interim, until upstream exposes
    the information itself.
    """

    def __init__(self, real):
        self._real = real
        self.log = None

    def __getattr__(self, name):
        return getattr(self._real, name)

    def run(self, *args, **kwargs):
        completed = self._real.run(*args, **kwargs)
        self.log = f"{completed.stdout or ''}\n{completed.stderr or ''}"
        return completed


def _uninstrumented(upstream_result, log: Optional[str]) -> Optional[list[str]]:
    """Uninstrumented kernel names: upstream's own list and the per-dispatch log lines, merged.

    Upstream parses the summary Omniprobe prints at exit, but other Omniprobe threads
    still write to stderr then and can split an entry mid-line ("... (1 dispatchComms
    Runner shutting down"), which drops it (measured with Omniprobe 13ca707: 1 of 24
    runs). The per-dispatch "No instrumented alternative found for" lines are written
    earlier and stay intact, so both sources are used.
    """
    reported = getattr(upstream_result, "uninstrumented_kernels", None)
    if not isinstance(reported, dict):
        return None if log is None else parse_uninstrumented_kernels(log)
    names = [k for k in reported if not k.startswith("__amd_rocclr_")]
    if log is not None:
        names += [k for k in parse_uninstrumented_kernels(log) if k not in names]
    return names


def _call_capturing_log(func, **kwargs):
    """Call an upstream analysis method; return (result, omniprobe log or None)."""
    try:
        import omniprobe.api.omniprobe as upstream_module
    except ImportError:
        return func(**kwargs), None
    real = getattr(upstream_module, "subprocess", None)
    if real is None:
        return func(**kwargs), None
    proxy = _CapturingSubprocess(real)
    upstream_module.subprocess = proxy
    try:
        result = func(**kwargs)
    finally:
        upstream_module.subprocess = real
    return result, proxy.log


def uninstrumented_warning(kernels: Optional[list[str]]) -> Optional[str]:
    """The warning text for kernels that ran uninstrumented, or None."""
    if not kernels:
        return None
    names = "; ".join(kernels)
    return (
        f"{len(kernels)} dispatched kernel(s) ran WITHOUT instrumentation and are missing from "
        f"these results: {names}. Rebuild the files that define them with the instrumentation "
        "flags (every file that defines a kernel needs them)."
    )


def _warn_uninstrumented(kernels: Optional[list[str]]) -> None:
    message = uninstrumented_warning(kernels)
    if message:
        logger.warning(message)


_MISSING_TOOL = re.compile(
    r"^sh: (?:line \d+: )?(\S+): (?:No such file or directory|command not found)", re.M
)


def environment_problem(log: Optional[str]) -> Optional[str]:
    """Why Omniprobe's own environment, not the build, kept it from reporting, or None.

    Omniprobe 149cf2c and later pick a ROCm with llvm-objdump themselves and print
    ``omniprobe: ERROR: no ROCm installation ...`` or ``kernelDB could not read ...``
    when that fails. Older builds (measured with Omniprobe 13ca707): without ROCM_PATH it
    runs ``/opt/rocm/llvm/bin/llvm-objdump`` and, where that does not exist, prints
    ``sh: ...: No such file or directory`` and ``[hsaInterceptor] Command failed``, then
    reports nothing although it found the instrumented clones. If the interceptor never
    printed ``HSA intercept registered``, it did not load at all.
    """
    if not log:
        return None
    rocm = os.environ.get("ROCM_PATH")
    where = f"ROCM_PATH is {rocm!r}" if rocm else "ROCM_PATH is not set, so /opt/rocm is assumed"
    if "no ROCm installation with llvm/bin/llvm-objdump found" in log or (
        "kernelDB could not read" in log
    ):
        # Omniprobe 149cf2c+ finds its ROCm itself and says so when it cannot.
        return (
            f"Omniprobe could not find a ROCm installation with llvm-objdump ({where}), so it "
            "has no source attribution. Set ROCM_PATH to the ROCm installation Omniprobe was "
            "built against (e.g. /opt/rocm-<version>) in the shell, or in the MCP server's "
            "'env' configuration, and rerun."
        )
    missing = _MISSING_TOOL.search(log)
    if missing or "[hsaInterceptor] Command failed" in log:
        what = f"could not run {missing.group(1)}" if missing else "had a helper command fail"
        return (
            f"Omniprobe {what} ({where}). Set ROCM_PATH to the ROCm installation Omniprobe was "
            "built against (e.g. /opt/rocm-<version>) in the shell, or in the MCP server's "
            "'env' configuration, and rerun."
        )
    if "HSA intercept registered" not in log and "instrumented alternative" not in log:
        return (
            "Omniprobe's interceptor did not load, so no dispatch was observed. Check the "
            f"environment Omniprobe runs in ({where}; PATH, LD_LIBRARY_PATH, and in an MCP setup "
            "the server's 'env' configuration)."
        )
    return None


_FILTER_MISSED = re.compile(
    r"no dispatched kernel matched the kernel filter -k '(?P<filter>.*?)', so none ran "
    r"instrumented\..*?dispatched were:\n(?P<names>(?:  .*\n?)*)",
    re.S,
)
_CANDIDATE = re.compile(r"^  (.+?)(?: \(\d+ dispatch(?:es)?\))?$")


def kernel_filter_missed(log: Optional[str]) -> Optional[tuple[str, list[str]]]:
    """(filter, dispatched kernel names) if Omniprobe said the filter matched nothing, else None.

    Omniprobe b71927b and later print this at exit when instrumented kernels ran but
    the ``-k`` filter excluded every one of them.
    """
    if not log:
        return None
    match = _FILTER_MISSED.search(log)
    if not match:
        return None
    names = []
    for line in match.group("names").splitlines():
        candidate = _CANDIDATE.match(line)
        if candidate and not candidate.group(1).startswith("... and "):
            names.append(candidate.group(1))
    return match.group("filter"), names


def kernel_filter_message(command: str, kernel_filter: str, dispatched: list[str]) -> str:
    """The error text for a kernel filter that matched no dispatched kernel."""
    return "\n".join(
        [
            f"The kernel filter {kernel_filter!r} matched none of the instrumented kernels "
            f"{command!r} dispatched, so Omniprobe analyzed nothing. The application is "
            "instrumented; do not rebuild it.",
            "The filter is an ECMAScript regex searched anywhere in the demangled kernel name, "
            "e.g. 'void name<4>(double*)' for a template kernel, so '^name' does not match it.",
            "Kernels dispatched: " + ("; ".join(dispatched) if dispatched else "(none listed)"),
        ]
    )


_OMNIPROBE_MESSAGE = re.compile(r"omniprobe: (?:WARNING|ERROR): .*(?:\n  .*)*")
_INSTRUMENTED = re.compile(r"^Found instrumented alternative for (.+?)\s*$", re.M)
_EXCLUDED = re.compile(r"^Kernel filter excluded (.+?); running uninstrumented\s*$", re.M)


def omniprobe_messages(log: Optional[str]) -> list[str]:
    """Omniprobe's own ``omniprobe: WARNING:`` / ``omniprobe: ERROR:`` messages in a run's log.

    Each message keeps its indented continuation lines (e.g. the kernel list after
    the kernel-filter warning). Passed through so that callers see why Omniprobe
    reported what it did, including warnings this wrapper does not know yet.
    """
    if not log:
        return []
    seen: list[str] = []
    for match in _OMNIPROBE_MESSAGE.finditer(log):
        text = match.group(0).rstrip()
        if text not in seen:
            seen.append(text)
    return seen


def random_dispatch_missed(
    log: Optional[str], kernel_filter: Optional[str], dispatches: str
) -> bool:
    """True if ``dispatches="random"`` with a filter is the likely reason nothing was captured.

    Omniprobe 22c4d95 draws the random dispatch from all dispatches, including the
    ones the kernel filter excludes (measured: 9 of 12 runs empty in pnimpl's repro,
    4 of 4 here), so a kernel the filter matches can run uninstrumented.
    """
    if dispatches != "random" or kernel_filter is None or not log:
        return False
    excluded = set(_EXCLUDED.findall(log))
    return any(name not in excluded for name in _INSTRUMENTED.findall(log))


def _raise_no_data(
    analysis, command, kernel_filter, uninstrumented, log, dispatches: str = "all"
) -> None:
    messages = omniprobe_messages(log)
    missed = kernel_filter_missed(log)
    if missed is not None:
        used_filter, dispatched = missed
        text = kernel_filter_message(command, kernel_filter or used_filter, dispatched)
        if messages:
            text += "\nOmniprobe's own messages:\n" + "\n".join(messages)
        raise KernelFilterMatchedNothingError(text, dispatched)
    if random_dispatch_missed(log, kernel_filter, dispatches):
        lines = [
            f"Omniprobe captured nothing for {command!r} with dispatches='random' and "
            f"kernel_filter {kernel_filter!r}. The filter matches an instrumented kernel, but "
            "Omniprobe picks the random dispatch among all dispatches, including the ones the "
            "filter excludes, so the matching kernel can run uninstrumented. The application is "
            "instrumented; do not rebuild it. Use dispatches='all' (or '1') with a filter."
        ]
    else:
        lines = [_no_data_message(analysis, command, kernel_filter, uninstrumented, log)]
    if messages:
        lines.append("Omniprobe's own messages:\n" + "\n".join(messages))
    raise NoInstrumentedKernelsError("\n".join(lines))


def _no_data_message(
    analysis: str,
    command: str,
    kernel_filter: Optional[str],
    uninstrumented: Optional[list[str]] = None,
    log: Optional[str] = None,
) -> str:
    env_problem = environment_problem(log)
    if env_problem:
        lines = [f"Omniprobe reported no results for {command!r}: {env_problem}"]
        if log and "Found instrumented alternative" in log:
            lines.append(
                "The application is instrumented (Omniprobe found its instrumented kernels); "
                "do not rebuild it."
            )
        return "\n".join(lines)
    plugin = ANALYSIS_PLUGINS[analysis]
    try:
        flags = " ".join(instrumentation_flags(analysis))
    except (FileNotFoundError, RuntimeError):
        flags = f"-fpass-plugin=<omniprobe>/lib/plugins/{plugin} --offload-arch=<gfx arch>"

    lines = [
        f"Omniprobe reported no instrumented kernels for {command!r}.",
        f"Omniprobe only sees kernels compiled with its {plugin} instrumentation plugin.",
        f"Rebuild the application with: hipcc -g {flags} ...",
        "(instrumentation_flags() / 'omniprobe-ik flags' print these flags.)",
    ]
    if uninstrumented:
        lines.append("Kernels dispatched without instrumentation: " + "; ".join(uninstrumented))
    if kernel_filter is not None:
        lines.append(f"Also check that kernel_filter {kernel_filter!r} matches a launched kernel.")
    lines.append("Otherwise the application launched no GPU kernel.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Result dataclasses (IntelliKit-style, mirroring upstream with clean names)
# ---------------------------------------------------------------------------


@dataclass
class MemoryAccess:
    """A single observed memory access from Omniprobe's memory analysis."""

    source_file: str
    line: int
    column: int
    excess_cache_lines: int
    bank_conflicts: int
    memory_space: str = ""
    access_type: str = ""
    execution_count: int = 0
    cache_lines_needed: int = 0
    cache_lines_used: int = 0
    isa_instruction: str = ""
    ir_bytes: int = 0
    isa_bytes: int = 0


@dataclass
class KernelMemoryResult:
    """Memory analysis results for a single kernel dispatch."""

    kernel_name: str
    dispatch_id: int
    accesses: list[MemoryAccess] = field(default_factory=list)
    # Generic (flat) accesses left unanalyzed: waves whose lanes pointed into different
    # address spaces (Omniprobe e807927+), or every flat access of a binary instrumented
    # by an older Omniprobe.
    unanalyzed_flat_accesses: int = 0
    # Global accesses dropped because no ISA instruction was found for their source
    # line: all of them when the binary has no debug info (-g). Omniprobe 821ba07+.
    unattributed_accesses: int = 0


@dataclass
class MemoryAnalysisResult:
    """Top-level result from ``analyze_memory``.

    ``uninstrumented_kernels`` names kernels that were dispatched but ran
    without instrumentation, so they are missing from ``kernels``. ``None``
    means Omniprobe's log could not be inspected, not that nothing was missed.
    """

    kernels: list[KernelMemoryResult] = field(default_factory=list)
    uninstrumented_kernels: Optional[list[str]] = field(default_factory=list)
    # Omniprobe's own "omniprobe: WARNING/ERROR:" messages from the run, verbatim.
    omniprobe_messages: list[str] = field(default_factory=list)


@dataclass
class BasicBlock:
    """Exact execution counts and source location for one basic block.

    ``wave_visits`` is how many times a wave entered the block; ``active_lanes``
    sums the active lanes over those entries, so ``active_lanes / wave_visits``
    is the mean lane occupancy (divergence). There is no timing.
    """

    bb_id: int
    source_file: str
    line: int
    column: int = 0
    wave_visits: int = 0
    active_lanes: int = 0


@dataclass
class KernelBBResult:
    """Basic-block analysis results for a single kernel dispatch."""

    kernel_name: str
    dispatch_id: int
    basic_blocks: list[BasicBlock] = field(default_factory=list)


@dataclass
class BasicBlockResult:
    """Top-level result from ``analyze_basic_blocks``.

    ``uninstrumented_kernels`` is as for :class:`MemoryAnalysisResult`.
    """

    kernels: list[KernelBBResult] = field(default_factory=list)
    uninstrumented_kernels: Optional[list[str]] = field(default_factory=list)
    # Omniprobe's own "omniprobe: WARNING/ERROR:" messages from the run, verbatim.
    omniprobe_messages: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main API class
# ---------------------------------------------------------------------------


class Omniprobe:
    """High-level API for Omniprobe GPU memory and basic-block analysis.

    This class wraps Omniprobe's upstream Python API (``omniprobe.api.Omniprobe``)
    and converts results into IntelliKit-style dataclasses.

    Args:
        omniprobe_path: Explicit path to the ``omniprobe`` executable.
            If *None*, Omniprobe will locate it via ``PATH``.

    Raises:
        ImportError: If the ``omniprobe.api`` module is not importable.
        FileNotFoundError: If the ``omniprobe`` executable cannot be found.
    """

    def __init__(self, omniprobe_path: Optional[str] = None) -> None:
        try:
            from omniprobe.api import Omniprobe as _UpstreamOmniprobe
        except ImportError:
            if not _add_upstream_source_to_path(omniprobe_path):
                where = (
                    "no omniprobe executable on PATH"
                    if (omniprobe_path or shutil.which("omniprobe")) is None
                    else "the omniprobe executable found has no omniprobe/api next to it"
                )
                msg = f"Could not import omniprobe.api, Omniprobe's Python API ({where}). {_NOT_INSTALLED}"
                raise ImportError(msg) from None
            from omniprobe.api import Omniprobe as _UpstreamOmniprobe

        self._upstream = _UpstreamOmniprobe(omniprobe_path=omniprobe_path)
        logger.info("Omniprobe initialized (exe: %s)", self._upstream._exe)

    # -- public methods -----------------------------------------------------

    def analyze_memory(
        self,
        command: str,
        kernel_filter: Optional[str] = None,
        dispatches: str = "all",
    ) -> MemoryAnalysisResult:
        """Run memory analysis on *command* and return structured results.

        Args:
            command: Application command line to instrument.
            kernel_filter: Optional ECMAScript regex, searched anywhere in the
                demangled kernel name (a template kernel's name starts with its
                return type, e.g. ``void name<4>(...)``, so ``^name`` misses it).
            dispatches: Which dispatches to capture — ``"all"``, ``"random"``,
                or ``"1"``. With a *kernel_filter*, ``"random"`` often captures
                nothing (Omniprobe 22c4d95 draws from all dispatches, including the
                excluded ones); the error then says so.

        Returns:
            A :class:`MemoryAnalysisResult` containing per-kernel memory
            access observations.

        Raises:
            KernelFilterMatchedNothingError: *kernel_filter* matched none of the
                instrumented kernels the application dispatched.
            NoInstrumentedKernelsError: No kernel was reported, typically
                because the application was not compiled with
                ``instrumentation_flags("memory")``.
        """
        upstream_result, log = _call_capturing_log(
            self._upstream.analyze_memory,
            command=command,
            kernel_filter=kernel_filter,
            dispatches=dispatches,
        )
        result = self._convert_memory_result(upstream_result)
        result.uninstrumented_kernels = _uninstrumented(upstream_result, log)
        result.omniprobe_messages = omniprobe_messages(log)
        if not result.kernels:
            _raise_no_data(
                "memory", command, kernel_filter, result.uninstrumented_kernels, log, dispatches
            )
        _warn_uninstrumented(result.uninstrumented_kernels)
        return result

    def analyze_basic_blocks(
        self,
        command: str,
        kernel_filter: Optional[str] = None,
        dispatches: str = "all",
    ) -> BasicBlockResult:
        """Run basic-block analysis on *command* and return structured results.

        Args:
            command: Application command line to instrument.
            kernel_filter: Optional ECMAScript regex, searched anywhere in the
                demangled kernel name (a template kernel's name starts with its
                return type, e.g. ``void name<4>(...)``, so ``^name`` misses it).
            dispatches: Which dispatches to capture — ``"all"``, ``"random"``,
                or ``"1"``. With a *kernel_filter*, ``"random"`` often captures
                nothing (Omniprobe 22c4d95 draws from all dispatches, including the
                excluded ones); the error then says so.

        Returns:
            A :class:`BasicBlockResult` containing per-kernel basic-block
            timing data.

        Raises:
            KernelFilterMatchedNothingError: *kernel_filter* matched none of the
                instrumented kernels the application dispatched.
            NoInstrumentedKernelsError: No kernel was reported, typically
                because the application was not compiled with
                ``instrumentation_flags("basic_blocks")``.
        """
        upstream_result, log = _call_capturing_log(
            self._upstream.analyze_basic_blocks,
            command=command,
            kernel_filter=kernel_filter,
            dispatches=dispatches,
        )
        result = self._convert_bb_result(upstream_result)
        result.uninstrumented_kernels = _uninstrumented(upstream_result, log)
        result.omniprobe_messages = omniprobe_messages(log)
        if not result.kernels:
            _raise_no_data(
                "basic_blocks",
                command,
                kernel_filter,
                result.uninstrumented_kernels,
                log,
                dispatches,
            )
        _warn_uninstrumented(result.uninstrumented_kernels)
        return result

    # -- conversion helpers -------------------------------------------------

    @staticmethod
    def _convert_memory_result(upstream) -> MemoryAnalysisResult:
        kernels = []
        for k in upstream.kernels:
            accesses = [
                MemoryAccess(
                    source_file=a.source_file,
                    line=a.line,
                    column=a.column,
                    excess_cache_lines=a.excess_cache_lines,
                    bank_conflicts=a.bank_conflicts,
                    memory_space=a.memory_space,
                    access_type=a.access_type,
                    execution_count=a.execution_count,
                    cache_lines_needed=getattr(a, "cache_lines_needed", 0),
                    cache_lines_used=getattr(a, "cache_lines_used", 0),
                    isa_instruction=getattr(a, "isa_instruction", ""),
                    ir_bytes=getattr(a, "ir_bytes", 0),
                    isa_bytes=getattr(a, "isa_bytes", 0),
                )
                for a in k.accesses
            ]
            kernels.append(
                KernelMemoryResult(
                    kernel_name=k.kernel_name,
                    dispatch_id=k.dispatch_id,
                    accesses=accesses,
                    unanalyzed_flat_accesses=getattr(k, "unanalyzed_flat_accesses", 0) or 0,
                    unattributed_accesses=getattr(k, "unattributed_accesses", 0) or 0,
                )
            )
        return MemoryAnalysisResult(kernels=kernels)

    @staticmethod
    def _convert_bb_result(upstream) -> BasicBlockResult:
        kernels = []
        for k in upstream.kernels:
            blocks = [
                BasicBlock(
                    bb_id=b.bb_id,
                    source_file=b.source_file,
                    line=b.line,
                    column=getattr(b, "column", 0),
                    wave_visits=getattr(b, "wave_visits", 0),
                    active_lanes=getattr(b, "active_lanes", 0),
                )
                for b in k.basic_blocks
            ]
            kernels.append(
                KernelBBResult(
                    kernel_name=k.kernel_name,
                    dispatch_id=k.dispatch_id,
                    basic_blocks=blocks,
                )
            )
        return BasicBlockResult(kernels=kernels)


# ---------------------------------------------------------------------------
# Ranked summary
# ---------------------------------------------------------------------------

HOW_TO_READ_MEMORY = (
    "excess_per_execution: cache lines fetched beyond the minimum, per wave execution of the "
    "instruction. 0 = fully coalesced; for 4-byte accesses by 64 lanes on 128-byte lines the "
    "worst case is 62 (every lane on its own line); a lane stride of S floats gives "
    "min(64, 2S) - 2. conflicts_per_execution: extra LDS cycles from bank conflicts per wave "
    "execution; 0 = conflict-free, 62 = worst case for 4-byte accesses (32 banks, the wave "
    "checked as two halves of 32 lanes). Rank by the totals to decide what to fix first; "
    "use the per-execution values to judge how bad each access pattern is."
)

KNOWN_BLIND_SPOTS = [
    "Only global and LDS (shared) accesses are analyzed; scratch (private) and constant "
    "accesses are not reported.",
    "Code built without -g: global accesses cannot be matched to source lines and are "
    "dropped (counted in unattributed_accesses). Build with -g.",
    "Generic (flat) pointer accesses -- every access at -O0, and accesses through a pointer "
    "loaded from memory -- are classified per wave at run time (Omniprobe e807927 and later) "
    "and analyzed as global or LDS. A wave whose lanes point into different address spaces "
    "is counted in unanalyzed_flat_accesses and not analyzed. Binaries instrumented by an "
    "older Omniprobe count all their flat accesses there.",
    "Kernels built without the instrumentation plugin, and kernels excluded by the kernel "
    "filter, run uninstrumented and are missing from the results.",
]


def flat_accesses_warning(per_kernel: dict) -> Optional[str]:
    """The warning for flat accesses Omniprobe counted but could not analyze, or None."""
    if not per_kernel:
        return None
    detail = "; ".join(f"{name}: {n}" for name, n in per_kernel.items())
    return (
        "Some generic (flat) pointer accesses were NOT analyzed, so they are missing from these "
        f"results ({detail} wave executions). Omniprobe e807927 and later classify flat accesses "
        "per wave at run time and leave out only waves whose lanes point into different address "
        "spaces; binaries instrumented by an older Omniprobe count every flat access here "
        "(all accesses at -O0, and accesses through pointers loaded from memory)."
    )


def unattributed_accesses_warning(
    per_kernel: dict, attributed_kernels: Optional[set] = None
) -> Optional[str]:
    """The warning for accesses dropped because they had no source line, or None.

    A kernel that also has attributed accesses was built with debug info, so
    missing ``-g`` is not the cause there.
    """
    if not per_kernel:
        return None
    attributed_kernels = attributed_kernels or set()
    no_debug = {k: n for k, n in per_kernel.items() if k not in attributed_kernels}
    partial = {k: n for k, n in per_kernel.items() if k in attributed_kernels}
    parts = []
    if no_debug:
        detail = "; ".join(f"{name}: {n}" for name, n in no_debug.items())
        parts.append(
            "Global memory accesses could not be matched to a source line and were NOT analyzed, "
            f"so they are missing from these results ({detail} wave executions). None of these "
            "kernels' accesses has a source line: the binary was most likely built without debug "
            "info. Rebuild with -g."
        )
    if partial:
        detail = "; ".join(f"{name}: {n}" for name, n in partial.items())
        parts.append(
            "Some global memory accesses could not be matched to a source line and were NOT "
            f"analyzed ({detail} wave executions). These kernels have debug info (their other "
            "accesses were attributed), so -g is not missing; Omniprobe could not map these "
            "particular instructions to a line."
        )
    return " ".join(parts)


def silent_kernels_warning(kernel_names: list[str]) -> Optional[str]:
    """The warning for instrumented kernels that reported no global or LDS access, or None."""
    if not kernel_names:
        return None
    return (
        f"{len(kernel_names)} instrumented kernel(s) reported no global or LDS memory access: "
        f"{'; '.join(kernel_names)}. Omniprobe analyzes global and LDS accesses only; scratch "
        "(private) and constant accesses are not reported."
    )


def summarize_memory(result: MemoryAnalysisResult, top: int = 10) -> dict:
    """Rank the access sites of a memory analysis, worst first.

    Entries that share kernel, location, memory space and access type are
    merged first (Omniprobe before 3c6182d split one LDS location into several).
    Sites with nothing to report are left out.

    Returns:
        A dict with ``global_sites`` (ranked by total ``excess_cache_lines``),
        ``lds_sites`` (ranked by total ``bank_conflicts``), per-kernel
        ``kernels`` totals and a ``how_to_read`` explanation.
    """
    merged: dict = {}
    for kernel in result.kernels:
        for a in kernel.accesses:
            key = (
                kernel.kernel_name,
                a.source_file,
                a.line,
                a.column,
                a.memory_space,
                a.access_type,
            )
            site = merged.setdefault(
                key,
                {
                    "kernel": kernel.kernel_name,
                    "source_file": a.source_file,
                    "line": a.line,
                    "column": a.column,
                    "memory_space": a.memory_space,
                    "access_type": a.access_type,
                    "execution_count": 0,
                    "excess_cache_lines": 0,
                    "bank_conflicts": 0,
                },
            )
            site["execution_count"] += a.execution_count
            site["excess_cache_lines"] += a.excess_cache_lines
            site["bank_conflicts"] += a.bank_conflicts

    def per_execution(total: int, executions: int) -> float:
        return round(total / executions, 2) if executions else 0.0

    global_sites = []
    lds_sites = []
    kernels: dict = {}
    for site in merged.values():
        totals = kernels.setdefault(
            site["kernel"], {"kernel": site["kernel"], "excess_cache_lines": 0, "bank_conflicts": 0}
        )
        totals["excess_cache_lines"] += site["excess_cache_lines"]
        totals["bank_conflicts"] += site["bank_conflicts"]
        if site["excess_cache_lines"] > 0:
            global_sites.append(
                {
                    **{k: v for k, v in site.items() if k != "bank_conflicts"},
                    "excess_per_execution": per_execution(
                        site["excess_cache_lines"], site["execution_count"]
                    ),
                }
            )
        if site["bank_conflicts"] > 0:
            lds_sites.append(
                {
                    **{k: v for k, v in site.items() if k != "excess_cache_lines"},
                    "conflicts_per_execution": per_execution(
                        site["bank_conflicts"], site["execution_count"]
                    ),
                }
            )

    global_sites.sort(key=lambda s: s["excess_cache_lines"], reverse=True)
    lds_sites.sort(key=lambda s: s["bank_conflicts"], reverse=True)
    flat = {
        k.kernel_name: k.unanalyzed_flat_accesses
        for k in result.kernels
        if getattr(k, "unanalyzed_flat_accesses", 0)
    }
    unattributed = {
        k.kernel_name: k.unattributed_accesses
        for k in result.kernels
        if getattr(k, "unattributed_accesses", 0)
    }
    silent = [
        k.kernel_name
        for k in result.kernels
        if not k.accesses and k.kernel_name not in flat and k.kernel_name not in unattributed
    ]
    warnings = [
        w
        for w in (
            uninstrumented_warning(getattr(result, "uninstrumented_kernels", None)),
            flat_accesses_warning(flat),
            unattributed_accesses_warning(
                unattributed, {k.kernel_name for k in result.kernels if k.accesses}
            ),
            silent_kernels_warning(silent),
        )
        if w
    ]
    return {
        "warnings": warnings,
        "uninstrumented_kernels": getattr(result, "uninstrumented_kernels", None),
        "omniprobe_messages": getattr(result, "omniprobe_messages", []),
        "global_sites": global_sites[:top],
        "lds_sites": lds_sites[:top],
        "kernels": sorted(
            kernels.values(),
            key=lambda k: (k["excess_cache_lines"], k["bank_conflicts"]),
            reverse=True,
        ),
        "how_to_read": HOW_TO_READ_MEMORY,
    }
