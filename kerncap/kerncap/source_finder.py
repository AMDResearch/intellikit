"""Kernel source locator for HIP and Triton kernels.

Given a kernel name, finds the source code in a project directory tree.
"""

import ast
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


import logging

_logger = logging.getLogger(__name__)


@dataclass
class KernelSource:
    """Describes located kernel source code and its dependencies."""

    language: str  # "hip" or "triton"
    kernel_name: str
    main_file: str  # primary file containing the kernel
    source_files: List[str]  # all files needed
    include_paths: List[str] = field(default_factory=list)
    link_libraries: List[str] = field(default_factory=list)
    link_paths: List[str] = field(default_factory=list)
    compile_defines: List[str] = field(default_factory=list)
    kernel_function: str = ""  # the actual function/class name
    translation_unit: str = ""  # .cu file that compiles the kernel
    compile_command: str = ""  # full compile command from compile_commands.json
    compile_dir: str = ""  # directory the compile command was run from


def detect_language(
    kernel_name: str,
    source_dir: str,
) -> str:
    """Heuristic to detect whether a kernel is HIP or Triton.

    Returns "hip", "triton", or "unknown".
    """
    # Check for Triton kernels
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r") as f:
                    content = f.read()
                if "@triton.jit" in content or "@triton.autotune" in content:
                    # Check if the kernel name matches any function
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if kernel_name in node.name or node.name in kernel_name:
                                return "triton"
            except (SyntaxError, UnicodeDecodeError):
                continue

    # Check for HIP kernels
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if not fname.endswith((".hip", ".cpp", ".cu", ".hpp", ".h")):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r") as f:
                    content = f.read()
                if "__global__" in content:
                    return "hip"
            except UnicodeDecodeError:
                continue

    return "unknown"


# ---------------------------------------------------------------------------
# Triton source finder
# ---------------------------------------------------------------------------


def _find_triton_kernel(
    kernel_name: str,
    source_dir: str,
) -> Optional[KernelSource]:
    """Locate a @triton.jit kernel in a Python source tree."""
    matches: List[Tuple[str, str]] = []  # (file_path, function_name)

    for root, _, files in os.walk(source_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r") as f:
                    content = f.read()
            except (UnicodeDecodeError, PermissionError):
                continue

            if "@triton.jit" not in content and "@triton.autotune" not in content:
                continue

            try:
                tree = ast.parse(content, filename=fpath)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check decorators for @triton.jit or @triton.autotune
                    for dec in node.decorator_list:
                        dec_name = _get_decorator_name(dec)
                        if dec_name in ("triton.jit", "triton.autotune", "jit", "autotune"):
                            if kernel_name in node.name or node.name in kernel_name:
                                matches.append((fpath, node.name))

    if not matches:
        return None

    # Prefer exact match
    best = matches[0]
    for fpath, fname in matches:
        if fname == kernel_name:
            best = (fpath, fname)
            break

    main_file, func_name = best

    # Trace imports to find helper functions
    deps = _trace_triton_deps(main_file, source_dir)

    return KernelSource(
        language="triton",
        kernel_name=kernel_name,
        main_file=main_file,
        source_files=[main_file] + deps,
        kernel_function=func_name,
    )


def _get_decorator_name(node: ast.expr) -> str:
    """Extract a decorator name string from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    elif isinstance(node, ast.Call):
        return _get_decorator_name(node.func)
    return ""


def _trace_triton_deps(main_file: str, source_dir: str) -> List[str]:
    """Trace Python imports from a Triton kernel file to find dependencies."""
    deps: List[str] = []
    visited: Set[str] = {os.path.abspath(main_file)}

    try:
        with open(main_file, "r") as f:
            tree = ast.parse(f.read(), filename=main_file)
    except (SyntaxError, UnicodeDecodeError):
        return deps

    main_dir = os.path.dirname(main_file)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level and node.level > 0:
            # Relative import (e.g. from .common import ..., from . import foo)
            rel_dir = main_dir
            for _ in range(node.level - 1):
                rel_dir = os.path.dirname(rel_dir)

            # from .module import names
            if node.module:
                parts = node.module.split(".")
                candidate = os.path.join(rel_dir, *parts) + ".py"
                if os.path.isfile(candidate):
                    abs_path = os.path.abspath(candidate)
                    if abs_path not in visited:
                        visited.add(abs_path)
                        deps.append(candidate)

            # from . import name  (each name may be a sibling module)
            if node.names:
                for alias in node.names:
                    candidate = os.path.join(rel_dir, alias.name) + ".py"
                    if os.path.isfile(candidate):
                        abs_path = os.path.abspath(candidate)
                        if abs_path not in visited:
                            visited.add(abs_path)
                            deps.append(candidate)

        elif node.module:
            # Absolute import — convert module path to file path
            parts = node.module.split(".")
            for i in range(len(parts), 0, -1):
                candidate = os.path.join(source_dir, *parts[:i]) + ".py"
                if os.path.isfile(candidate):
                    abs_path = os.path.abspath(candidate)
                    if abs_path not in visited:
                        visited.add(abs_path)
                        deps.append(candidate)
                    break
                # Check for __init__.py in package
                candidate_pkg = os.path.join(source_dir, *parts[:i], "__init__.py")
                if os.path.isfile(candidate_pkg):
                    abs_path = os.path.abspath(candidate_pkg)
                    if abs_path not in visited:
                        visited.add(abs_path)
                        deps.append(candidate_pkg)
                    break

    return deps


# ---------------------------------------------------------------------------
# HIP source finder
# ---------------------------------------------------------------------------


def _find_hip_kernel(
    kernel_name: str,
    source_dir: str,
    extra_defines: Optional[List[str]] = None,
    mangled_name: str = "",
) -> Optional[KernelSource]:
    """Locate a __global__ HIP kernel in a C++/HIP source tree."""
    if mangled_name:
        cc_path = _find_compile_commands_from_source_dir(source_dir)
        if cc_path:
            result = _find_hip_kernel_via_dwarf(
                kernel_name, mangled_name, source_dir, cc_path, extra_defines
            )
            if result:
                _logger.info("Source discovery via DWARF debug info succeeded")
                return result
            _logger.debug("DWARF discovery failed, falling back to source grep")

    search_files: List[str] = []
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if fname.endswith((".hip", ".cpp", ".cu", ".cxx", ".cc", ".hpp", ".h", ".cuh")):
                fpath = os.path.join(root, fname)
                if fpath not in search_files:
                    search_files.append(fpath)

    # Demangle the kernel name to extract the base function name
    base_name = _extract_base_name(kernel_name)

    # Two-pass search: strict __global__ definition first, then fallback.
    # This prevents a driver file that merely *references* the kernel name
    # (e.g. via a host wrapper like ggml_cuda_mul_mat_vec_q) from shadowing
    # the file that actually *defines* the __global__ function.
    global_pattern = re.compile(rf"__global__\s+\w+\s+{re.escape(base_name)}\s*[\(<]")
    main_file = None
    fallback_file = None

    for fpath in search_files:
        try:
            with open(fpath, "r") as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError):
            continue

        if "__global__" in content and global_pattern.search(content):
            main_file = fpath
            break

        if fallback_file is None and base_name in content:
            fallback_file = fpath

    if main_file is None:
        main_file = fallback_file

    if not main_file:
        return None

    # Trace #include dependencies
    includes = _trace_hip_includes(main_file, source_dir)
    include_paths = _detect_include_paths(source_dir)
    all_sources = [main_file] + includes

    # Post-collection verification: ensure the __global__ kernel definition
    # is actually present in the collected source set.  If main_file was
    # chosen by fallback (or if the definition lives in a separate
    # translation unit not reachable via #include), locate the definition
    # file and pull it in.
    if not _has_kernel_definition(base_name, all_sources):
        abs_collected = {os.path.abspath(f) for f in all_sources}
        def_file = _find_definition_file(
            base_name,
            search_files,
            abs_collected,
        )
        if def_file:
            def_includes = _trace_hip_includes(def_file, source_dir)
            main_file = def_file
            merged: List[str] = [def_file]
            seen = {os.path.abspath(def_file)}
            for f in def_includes + all_sources:
                af = os.path.abspath(f)
                if af not in seen:
                    seen.add(af)
                    merged.append(f)
            all_sources = merged

    compile_defines = _detect_compile_defines(
        all_sources,
        None,
        extra_defines,
    )
    link_paths, link_libraries = _detect_link_libraries(
        None,
        source_dir,
    )

    # Find the .cu translation unit that actually compiles the kernel
    cc_path = _find_compile_commands_from_source_dir(source_dir)
    tu_path, compile_cmd, compile_dir = _find_translation_unit(
        main_file,
        source_dir,
        mangled_name=mangled_name,
        compile_commands_path=cc_path,
    )

    # If we found a TU via compile_commands.json, extract its -I/-D flags
    # which are more accurate than heuristic detection.
    if compile_cmd:
        import shlex

        cc_defines = _extract_defines_from_command(compile_cmd, [])
        cc_includes = _extract_includes_from_command(
            compile_cmd,
            [],
            working_dir=compile_dir,
        )
        if cc_defines:
            merged_defs: Dict[str, Optional[str]] = {}
            for d in cc_defines:
                k, _, v = d.partition("=")
                merged_defs[k] = v or None
            for d in extra_defines or []:
                k, _, v = d.partition("=")
                merged_defs[k] = v or None
            compile_defines = [f"{k}={v}" if v else k for k, v in merged_defs.items()]
        if cc_includes:
            include_paths = cc_includes

    if tu_path:
        _logger.info("Translation unit: %s", tu_path)

    return KernelSource(
        language="hip",
        kernel_name=kernel_name,
        main_file=main_file,
        source_files=all_sources,
        include_paths=include_paths,
        link_libraries=link_libraries,
        link_paths=link_paths,
        compile_defines=compile_defines,
        kernel_function=base_name,
        translation_unit=tu_path or "",
        compile_command=compile_cmd,
        compile_dir=compile_dir,
    )


def _has_kernel_definition(base_name: str, source_files: List[str]) -> bool:
    """Check whether any file in *source_files* contains a ``__global__``
    function definition matching *base_name*.
    """
    pattern = re.compile(rf"__global__\s+\w+\s+{re.escape(base_name)}\s*[\(<]")
    for fpath in source_files:
        try:
            with open(fpath, "r") as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError, FileNotFoundError):
            continue
        if pattern.search(content):
            return True
    return False


def _find_definition_file(
    base_name: str,
    search_files: List[str],
    exclude: Set[str],
) -> Optional[str]:
    """Search *search_files* for the ``__global__`` definition of
    *base_name*, skipping any paths already in *exclude*.
    """
    pattern = re.compile(rf"__global__\s+\w+\s+{re.escape(base_name)}\s*[\(<]")
    for fpath in search_files:
        if os.path.abspath(fpath) in exclude:
            continue
        try:
            with open(fpath, "r") as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError, FileNotFoundError):
            continue
        if pattern.search(content):
            return fpath
    return None


def _extract_base_name(demangled_name: str) -> str:
    """Extract the base function name from a demangled kernel name.

    E.g., "void kernel_name<float>(float*, ...)" -> "kernel_name"
    E.g., "void ck::GridwiseGemm<float>::Run()" -> "Run"
    """
    # Remove return type prefix
    name = demangled_name.strip()
    if name.startswith("void "):
        name = name[5:]

    # Remove parameter list (everything after the outermost '(')
    # but be careful with template args containing nested parens
    paren_depth = 0
    angle_depth = 0
    paren_start = -1
    for i, ch in enumerate(name):
        if ch == "<":
            angle_depth += 1
        elif ch == ">":
            angle_depth -= 1
        elif ch == "(" and angle_depth == 0:
            paren_start = i
            break
    if paren_start >= 0:
        name = name[:paren_start]

    # Remove template args — strip balanced <...> from the end
    while name.endswith(">"):
        depth = 0
        for i in range(len(name) - 1, -1, -1):
            if name[i] == ">":
                depth += 1
            elif name[i] == "<":
                depth -= 1
                if depth == 0:
                    name = name[:i]
                    break

    # Handle unbalanced template brackets (e.g. truncated profiler output
    # like "mul_mat_vec_q<(ggml_type)39" with no closing >)
    if "<" in name:
        depth = 0
        for ch in name:
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
        if depth > 0:
            name = name[: name.index("<")]

    name = name.strip()

    # Take the last component if namespaced
    parts = name.split("::")
    result = parts[-1] if parts else name
    return result.strip()


def _trace_hip_includes(
    main_file: str,
    source_dir: str,
    max_depth: int = 5,
) -> List[str]:
    """Recursively trace #include directives from a HIP source file."""
    includes: List[str] = []
    visited: Set[str] = {os.path.abspath(main_file)}

    def _trace(fpath: str, depth: int):
        if depth > max_depth:
            return
        try:
            with open(fpath, "r") as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError, FileNotFoundError):
            return

        # Match #include "..." (local includes, not system includes)
        for match in re.finditer(r'#include\s+"([^"]+)"', content):
            inc_name = match.group(1)
            # Resolve relative to the including file's directory
            inc_dir = os.path.dirname(fpath)
            candidates = [
                os.path.join(inc_dir, inc_name),
                os.path.join(source_dir, inc_name),
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    abs_cand = os.path.abspath(cand)
                    if abs_cand not in visited:
                        visited.add(abs_cand)
                        includes.append(cand)
                        _trace(cand, depth + 1)
                    break

    _trace(main_file, 0)
    return includes


def _detect_include_paths(source_dir: str) -> List[str]:
    """Detect standard include paths needed for the project.

    Searches up to 3 levels deep for directories named ``include``,
    ``inc``, or ``src`` — covers project layouts like
    ``project/ggml/include/`` or ``project/lib/foo/inc/``.

    Also checks sibling directories of *source_dir* (e.g. if source_dir
    is ``ggml/src``, also finds ``ggml/include``).
    """
    source_dir = os.path.abspath(source_dir)
    paths = [source_dir]
    seen: set = {source_dir}

    include_dir_names = {"include", "inc", "src"}

    # Check siblings of source_dir (parent's children).
    parent = os.path.dirname(source_dir)
    if parent and parent != source_dir:
        try:
            for d in os.listdir(parent):
                if d in include_dir_names:
                    candidate = os.path.join(parent, d)
                    if os.path.isdir(candidate) and candidate not in seen:
                        seen.add(candidate)
                        paths.append(candidate)
        except OSError:
            pass

    for root, dirs, _ in os.walk(source_dir):
        depth = root[len(source_dir) :].count(os.sep)
        if depth >= 3:
            dirs.clear()
            continue
        for d in dirs:
            if d in include_dir_names:
                candidate = os.path.join(root, d)
                if candidate not in seen:
                    seen.add(candidate)
                    paths.append(candidate)

    # Check for ROCm / CK includes
    rocm_inc = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "include")
    if os.path.isdir(rocm_inc):
        paths.append(rocm_inc)

    return paths


# ---------------------------------------------------------------------------
# Translation-unit discovery
# ---------------------------------------------------------------------------


def _find_compile_commands_from_source_dir(
    source_dir: str,
) -> Optional[str]:
    """Locate ``compile_commands.json`` by walking up from *source_dir*.

    Checks siblings (``build/``) and ancestors until the filesystem root.
    """
    cur = os.path.abspath(source_dir)
    prev = None
    while cur != prev:
        candidate = os.path.join(cur, "compile_commands.json")
        if os.path.isfile(candidate):
            return candidate
        build_candidate = os.path.join(cur, "build", "compile_commands.json")
        if os.path.isfile(build_candidate):
            return build_candidate
        prev = cur
        cur = os.path.dirname(cur)
    return None


def _find_translation_unit(
    kernel_header: str,
    source_dir: str,
    mangled_name: str = "",
    compile_commands_path: Optional[str] = None,
) -> Tuple[Optional[str], str, str]:
    """Find the ``.cu`` translation unit that compiles *kernel_header*.

    Returns ``(tu_path, compile_command, compile_dir)`` where *compile_command* is the
    full compiler invocation from ``compile_commands.json`` (empty string
    if unavailable) and *compile_dir* is the execution directory.

    The search proceeds in three phases:

    0. **Direct match** — if the kernel file itself is a ``.cu``/``.hip``/
       ``.cpp`` file with a direct entry in ``compile_commands.json``, it IS
       its own translation unit (no include search needed).
    1. **compile_commands.json** — scan entries for ``.cu`` files whose
       source content ``#include``s the kernel header.  If multiple match
       (template-instance files), prefer the one whose content relates to
       the captured template parameters encoded in *mangled_name*.
    2. **Reverse-include grep** — if no ``compile_commands.json`` is
       available, scan ``.cu`` files in *source_dir* directly.
    """
    import json as _json
    import shlex

    header_basename = os.path.basename(kernel_header)
    kernel_header_abs = os.path.abspath(kernel_header)

    candidates: List[Tuple[str, str, str]] = []  # (tu_path, compile_command, compile_dir)
    include_pattern = re.compile(rf'#include\s+"[^"]*{re.escape(header_basename)}\s*"')

    # --- Phase 0: direct match --------------------------------------------
    # When the kernel is defined in a .cu/.hip/.cpp file (not a header),
    # it is its own translation unit.  Look for it directly in
    # compile_commands.json before doing an include search.
    if compile_commands_path and os.path.isfile(compile_commands_path):
        try:
            with open(compile_commands_path, "r") as f:
                entries = _json.load(f)
        except (OSError, ValueError):
            entries = []

        if kernel_header_abs.endswith((".cu", ".hip", ".cpp")):
            for entry in entries:
                entry_file = entry.get("file", "")
                if not os.path.isabs(entry_file):
                    entry_file = os.path.join(entry.get("directory", ""), entry_file)
                if os.path.abspath(entry_file) == kernel_header_abs:
                    cmd = entry.get("command", "")
                    if not cmd:
                        args = entry.get("arguments", [])
                        cmd = " ".join(shlex.quote(a) for a in args) if args else ""
                    directory = entry.get("directory", "")
                    _logger.debug("Kernel file is its own TU: %s", kernel_header)
                    return kernel_header_abs, cmd, directory

        # --- Phase 1: compile_commands.json include search ----------------
        for entry in entries:
            entry_file = entry.get("file", "")
            if not os.path.isabs(entry_file):
                entry_file = os.path.join(entry.get("directory", ""), entry_file)
            entry_abs = os.path.abspath(entry_file)

            if not entry_abs.endswith((".cu", ".hip", ".cpp")):
                continue

            try:
                with open(entry_abs, "r") as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError):
                continue

            if not include_pattern.search(content):
                continue

            cmd = entry.get("command", "")
            if not cmd:
                args = entry.get("arguments", [])
                cmd = " ".join(shlex.quote(a) for a in args) if args else ""
            directory = entry.get("directory", "")
            candidates.append((entry_abs, cmd, directory))

    # --- Phase 2: reverse-include grep ------------------------------------
    if not candidates:
        for root, _, files in os.walk(source_dir):
            for fname in files:
                if not fname.endswith((".cu", ".hip", ".cpp")):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r") as f:
                        content = f.read()
                except (OSError, UnicodeDecodeError):
                    continue
                if include_pattern.search(content):
                    candidates.append((fpath, "", ""))

    if not candidates:
        return None, "", ""

    if len(candidates) == 1:
        _logger.debug("Found translation unit: %s", candidates[0][0])
        return candidates[0]

    # Multiple candidates — use nm on the build's object files to find
    # which one contains the exact mangled symbol.  This is definitive
    # because the compiler already resolved enum values and template
    # parameters.
    if mangled_name and compile_commands_path:
        nm_match = _match_tu_via_object_symbols(
            mangled_name,
            candidates,
            compile_commands_path,
        )
        if nm_match:
            return nm_match

    # Fallback: smallest file by size (focused instantiation files are
    # typically ~3 lines, monolithic TUs are thousands).
    candidates.sort(key=lambda c: os.path.getsize(c[0]))
    _logger.debug("Falling back to smallest TU: %s", candidates[0][0])
    return candidates[0]


def _nm_has_symbol(nm_output: str, symbol: str) -> bool:
    """Check whether *symbol* appears as an exact symbol in ``nm`` output.

    Each ``nm`` output line has the form ``[address] type symbol``.  We compare
    the symbol column for exact equality to avoid false positives when the
    target is a prefix/substring of another symbol.
    """
    for line in nm_output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == symbol:
            return True
        if len(parts) == 2 and parts[1] == symbol:
            return True
    return False


def _match_tu_via_object_symbols(
    mangled_name: str,
    candidates: List[Tuple[str, str, str]],
    compile_commands_path: str,
) -> Optional[Tuple[str, str, str]]:
    """Match a mangled kernel symbol to a TU by checking object files.

    Parses ``compile_commands.json`` to find the ``-o`` output path for
    each candidate TU, then runs ``nm`` on the ``.o`` file to check if
    it contains the exact mangled symbol.
    """
    import json as _json
    import shlex

    try:
        with open(compile_commands_path, "r") as f:
            entries = _json.load(f)
    except (OSError, ValueError):
        return None

    # Build a map: absolute source path → object file path
    source_to_obj: Dict[str, str] = {}
    for entry in entries:
        entry_file = entry.get("file", "")
        if not os.path.isabs(entry_file):
            entry_file = os.path.join(
                entry.get("directory", ""),
                entry_file,
            )
        entry_abs = os.path.abspath(entry_file)

        cmd_str = entry.get("command", "")
        args = entry.get("arguments", [])
        tokens = args if args else shlex.split(cmd_str) if cmd_str else []

        obj_path = _extract_output_path(tokens, entry.get("directory", ""))
        if obj_path:
            source_to_obj[entry_abs] = obj_path

    candidate_sources = {os.path.abspath(c[0]) for c in candidates}

    for tu_path, cmd, comp_dir in candidates:
        tu_abs = os.path.abspath(tu_path)
        obj_path = source_to_obj.get(tu_abs)
        if not obj_path or not os.path.isfile(obj_path):
            continue

        try:
            proc = subprocess.run(
                ["nm", obj_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                continue
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue

        if _nm_has_symbol(proc.stdout, mangled_name):
            _logger.debug(
                "Matched translation unit %s via nm on %s",
                tu_path,
                obj_path,
            )
            return tu_path, cmd, comp_dir

    _logger.debug("nm lookup did not match any candidate object file")
    return None


def _extract_output_path(tokens: List[str], working_dir: str) -> Optional[str]:
    """Extract the ``-o`` output path from a compiler command."""
    for i, tok in enumerate(tokens):
        if tok == "-o" and i + 1 < len(tokens):
            path = tokens[i + 1]
            if not os.path.isabs(path) and working_dir:
                path = os.path.join(working_dir, path)
            return os.path.normpath(path)
        if tok.startswith("-o") and len(tok) > 2:
            path = tok[2:]
            if not os.path.isabs(path) and working_dir:
                path = os.path.join(working_dir, path)
            return os.path.normpath(path)
    return None


# ---------------------------------------------------------------------------
# DWARF-based source discovery
# ---------------------------------------------------------------------------


def _find_tu_by_symbol(
    mangled_name: str,
    compile_commands_path: str,
) -> Optional[Tuple[str, str, str, str]]:
    """Find the translation unit that compiled a mangled symbol.

    Scans all entries in ``compile_commands.json``, extracts object file
    paths via ``-o``, and runs ``nm`` on each to find which one contains
    *mangled_name*.

    Returns ``(tu_path, compile_command, compile_dir, object_file_path)``
    or ``None``.
    """
    import json as _json
    import shlex

    try:
        with open(compile_commands_path, "r") as f:
            entries = _json.load(f)
    except (OSError, ValueError):
        return None

    for entry in entries:
        entry_file = entry.get("file", "")
        if not os.path.isabs(entry_file):
            entry_file = os.path.join(entry.get("directory", ""), entry_file)
        entry_abs = os.path.abspath(entry_file)

        cmd_str = entry.get("command", "")
        args = entry.get("arguments", [])
        tokens = args if args else shlex.split(cmd_str) if cmd_str else []

        obj_path = _extract_output_path(tokens, entry.get("directory", ""))
        if not obj_path or not os.path.isfile(obj_path):
            continue

        try:
            proc = subprocess.run(
                ["nm", obj_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                continue
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue

        if _nm_has_symbol(proc.stdout, mangled_name):
            cmd = cmd_str
            if not cmd:
                cmd = " ".join(shlex.quote(a) for a in args) if args else ""
            directory = entry.get("directory", "")
            _logger.debug(
                "Found TU %s for symbol %s via nm on %s",
                entry_abs,
                mangled_name,
                obj_path,
            )
            return entry_abs, cmd, directory, obj_path

    return None


def _extract_dwarf_source_files(obj_path: str) -> Optional[List[str]]:
    """Extract source file paths from DWARF debug info in an object file.

    Tries ``llvm-dwarfdump`` first (ships with ROCm at
    ``/opt/rocm/llvm/bin/``), falls back to ``readelf``.
    Returns a list of source file paths, or ``None`` if no debug info
    is available.
    """
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    for dwarfdump in [
        os.path.join(rocm, "llvm", "bin", "llvm-dwarfdump"),
        "/opt/rocm/llvm/bin/llvm-dwarfdump",
        "llvm-dwarfdump",
    ]:
        try:
            proc = subprocess.run(
                [dwarfdump, "--debug-line", obj_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                files = _parse_llvm_dwarfdump_output(proc.stdout)
                if files:
                    return files
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue

    try:
        proc = subprocess.run(
            ["readelf", "--debug-dump=decodedline", obj_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            files = _parse_readelf_output(proc.stdout)
            if files:
                return files
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return None


def _parse_llvm_dwarfdump_output(output: str) -> List[str]:
    """Parse ``llvm-dwarfdump --debug-line`` output to extract source paths.

    Extracts ``include_directories`` and ``file_names`` from the line table
    prologue, then joins directory + filename to reconstruct full paths.
    """
    directories: Dict[int, str] = {}
    files: List[str] = []

    dir_re = re.compile(r'include_directories\[\s*(\d+)\]\s*=\s*"([^"]+)"')
    for m in dir_re.finditer(output):
        directories[int(m.group(1))] = m.group(2)

    file_re = re.compile(
        r"file_names\[\s*\d+\]:\s*\n"
        r'\s+name:\s*"([^"]+)"\s*\n'
        r"\s+dir_index:\s*(\d+)",
    )
    for m in file_re.finditer(output):
        name = m.group(1)
        dir_idx = int(m.group(2))
        if os.path.isabs(name):
            files.append(name)
        elif dir_idx in directories:
            files.append(os.path.join(directories[dir_idx], name))
        else:
            files.append(name)

    seen: Set[str] = set()
    unique: List[str] = []
    for f in files:
        norm = os.path.normpath(f)
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)

    return unique


def _parse_readelf_output(output: str) -> List[str]:
    """Parse ``readelf --debug-dump=decodedline`` output for source paths.

    Extracts file paths from ``CU:`` headers and decoded line-table data
    rows.
    """
    files: Set[str] = set()

    cu_re = re.compile(r"^CU:\s+(.+?):\s*$", re.MULTILINE)
    for m in cu_re.finditer(output):
        path = m.group(1).strip()
        if path:
            files.add(os.path.normpath(path))

    line_re = re.compile(r"^(\S+)\s+\d+\s+0x", re.MULTILINE)
    for m in line_re.finditer(output):
        path = m.group(1).strip()
        if path and "/" in path:
            files.add(os.path.normpath(path))

    return list(files)


def _pick_main_file(user_files: List[str], base_name: str) -> str:
    """Pick the best main_file from DWARF-discovered user source files.

    Preference order:
    1. Compilable source (.cu/.hip/.cpp) whose basename contains *base_name*
    2. Any compilable source file
    3. Header whose basename contains *base_name*
    4. First file in the list
    """
    source_exts = (".cu", ".hip", ".cpp", ".cxx", ".cc")

    for f in user_files:
        if f.endswith(source_exts) and base_name in os.path.basename(f):
            return f

    for f in user_files:
        if f.endswith(source_exts):
            return f

    for f in user_files:
        if base_name in os.path.basename(f):
            return f

    return user_files[0]


def _find_hip_kernel_via_dwarf(
    kernel_name: str,
    mangled_name: str,
    source_dir: str,
    compile_commands_path: str,
    extra_defines: Optional[List[str]] = None,
) -> Optional[KernelSource]:
    """Find a HIP kernel source via DWARF debug info in object files.

    Works backward from the compiled object: use the mangled symbol to
    find the object file, read its DWARF line tables to discover which
    source files contributed, then filter by *source_dir* to separate
    user code from framework/system headers.
    """
    tu_match = _find_tu_by_symbol(mangled_name, compile_commands_path)
    if not tu_match:
        _logger.debug("DWARF: no TU found for symbol %s", mangled_name)
        return None

    tu_path, compile_cmd, compile_dir, obj_path = tu_match

    dwarf_files = _extract_dwarf_source_files(obj_path)
    if not dwarf_files:
        _logger.debug("DWARF: no debug info in %s", obj_path)
        return None

    if compile_dir:
        dwarf_files = [
            os.path.join(compile_dir, f) if not os.path.isabs(f) else f
            for f in dwarf_files
        ]

    abs_source_dir = os.path.abspath(source_dir)
    user_files = [
        f
        for f in dwarf_files
        if os.path.abspath(f).startswith(abs_source_dir + os.sep) and os.path.isfile(f)
    ]

    if not user_files:
        _logger.debug(
            "DWARF: none of %d source files are under %s",
            len(dwarf_files),
            source_dir,
        )
        return None

    base_name = _extract_base_name(kernel_name)
    main_file = _pick_main_file(user_files, base_name)

    compile_defines = list(extra_defines or [])
    include_paths: List[str] = []

    if compile_cmd:
        cc_defines = _extract_defines_from_command(compile_cmd, [])
        cc_includes = _extract_includes_from_command(
            compile_cmd,
            [],
            working_dir=compile_dir,
        )
        if cc_defines:
            merged_defs: Dict[str, Optional[str]] = {}
            for d in cc_defines:
                k, _, v = d.partition("=")
                merged_defs[k] = v or None
            for d in extra_defines or []:
                k, _, v = d.partition("=")
                merged_defs[k] = v or None
            compile_defines = [f"{k}={v}" if v else k for k, v in merged_defs.items()]
        if cc_includes:
            include_paths = cc_includes

    _logger.info(
        "DWARF discovery: %d user files, main_file=%s, TU=%s",
        len(user_files),
        main_file,
        tu_path,
    )

    return KernelSource(
        language="hip",
        kernel_name=kernel_name,
        main_file=main_file,
        source_files=user_files,
        include_paths=include_paths or _detect_include_paths(source_dir),
        compile_defines=compile_defines,
        kernel_function=base_name,
        translation_unit=tu_path,
        compile_command=compile_cmd,
        compile_dir=compile_dir,
    )


# ---------------------------------------------------------------------------
# Link-library detection
# ---------------------------------------------------------------------------


def _detect_link_libraries(
    binary_path: Optional[str],
    source_dir: str,
) -> Tuple[List[str], List[str]]:
    """Detect project-local shared libraries needed to link the reproducer.

    When a kernel source file is ``#include``-d into the reproducer it may
    pull in host-side wrapper functions that reference symbols defined in
    the project's shared libraries (e.g. ``libggml-base.so``).  This
    function finds those libraries so the Makefile can link against them.

    Strategy: run ``ldd`` on *binary_path*, filter for libraries whose
    paths fall under the same directory tree as *binary_path* or
    *source_dir* (i.e. project-local, not system libraries).

    Returns ``(lib_paths, lib_names)`` — directory paths for ``-L`` and
    bare library names for ``-l``.
    """
    if not binary_path or not os.path.isfile(binary_path):
        return [], []

    try:
        proc = subprocess.run(
            ["ldd", binary_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return [], []
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return [], []

    abs_binary_dir = os.path.abspath(os.path.dirname(binary_path))
    abs_source_dir = os.path.abspath(source_dir)

    lib_dirs: Dict[str, None] = {}  # ordered set
    lib_names: List[str] = []

    for line in proc.stdout.splitlines():
        # ldd output: "\tlibfoo.so.1 => /path/to/libfoo.so.1 (0x...)"
        parts = line.strip().split()
        if "=>" not in parts or len(parts) < 3:
            continue
        lib_file = parts[parts.index("=>") + 1]
        if not os.path.isfile(lib_file):
            continue

        abs_lib = os.path.abspath(lib_file)
        abs_lib_dir = os.path.dirname(abs_lib)

        # Keep only project-local libraries (share a common prefix with
        # the binary or source directory).
        is_local = (
            abs_lib_dir.startswith(abs_binary_dir)
            or abs_lib_dir.startswith(abs_source_dir)
            or abs_binary_dir.startswith(abs_lib_dir)
        )
        if not is_local:
            # Also check if they share a common project root (walk up to
            # find a shared ancestor that isn't / or /usr etc.)
            common = os.path.commonpath([abs_lib, abs_binary_dir])
            is_local = len(common) > 10  # heuristic: non-trivial prefix

        if not is_local:
            continue

        basename = os.path.basename(lib_file)
        if not basename.startswith("lib"):
            continue

        # libfoo.so.1.2.3 → foo
        name = basename[3:]
        so_idx = name.find(".so")
        if so_idx >= 0:
            name = name[:so_idx]

        if abs_lib_dir not in lib_dirs:
            lib_dirs[abs_lib_dir] = None
        if name not in lib_names:
            lib_names.append(name)

    return list(lib_dirs), lib_names


# ---------------------------------------------------------------------------
# Compile-define detection
# ---------------------------------------------------------------------------

# Substrings that identify HIP/ROCm-related preprocessor guards.
_HIP_KEYWORDS = ("HIP", "ROCM", "AMD")


def _detect_compile_defines(
    source_files: List[str],
    binary_path: Optional[str] = None,
    extra_defines: Optional[List[str]] = None,
) -> List[str]:
    """Detect preprocessor defines needed for the HIP reproducer.

    1. Try ``compile_commands.json`` near *binary_path* for exact flags.
    2. Fall back to scanning *source_files* for ``#ifdef`` / ``#if defined()``
       guards whose names contain HIP/ROCM/AMD substrings.
    3. Merge any caller-supplied *extra_defines*.
    """
    defines: Dict[str, Optional[str]] = {}

    if binary_path:
        for d in _defines_from_compile_commands(binary_path, source_files):
            k, _, v = d.partition("=")
            defines[k] = v or None

    if not defines:
        defines.update(_defines_from_source_scan(source_files))

    for d in extra_defines or []:
        k, _, v = d.partition("=")
        defines[k] = v or None

    return [f"{k}={v}" if v else k for k, v in defines.items()]


def _defines_from_compile_commands(
    binary_path: str,
    source_files: List[str],
) -> List[str]:
    """Extract ``-D`` flags from a ``compile_commands.json`` near the binary."""
    import json as _json

    cc_path = _find_compile_commands(binary_path)
    if cc_path is None:
        return []

    source_basenames = {os.path.basename(f) for f in source_files}
    source_abspaths = {os.path.abspath(f) for f in source_files}

    try:
        with open(cc_path, "r") as f:
            entries = _json.load(f)
    except (OSError, ValueError):
        return []

    for entry in entries:
        entry_file = entry.get("file", "")
        if not os.path.isabs(entry_file):
            entry_file = os.path.join(entry.get("directory", ""), entry_file)
        entry_abs = os.path.abspath(entry_file)

        if entry_abs in source_abspaths or os.path.basename(entry_file) in source_basenames:
            return _extract_defines_from_command(
                entry.get("command", ""),
                entry.get("arguments", []),
            )
    return []


def _find_compile_commands(binary_path: str) -> Optional[str]:
    """Walk up from *binary_path* looking for ``compile_commands.json``."""
    cur = os.path.dirname(os.path.abspath(binary_path))
    prev = None
    while cur != prev:
        candidate = os.path.join(cur, "compile_commands.json")
        if os.path.isfile(candidate):
            return candidate
        if os.path.basename(cur) == "build":
            break
        prev = cur
        cur = os.path.dirname(cur)
    return None


def _extract_defines_from_command(
    command: str,
    arguments: List[str],
) -> List[str]:
    """Pull ``-D`` flags out of a compiler command string or argument list."""
    import shlex

    tokens = arguments if arguments else shlex.split(command)
    defines: List[str] = []
    skip_next = False
    for i, tok in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if tok.startswith("-D"):
            val = tok[2:]
            if val:
                defines.append(val)
            elif i + 1 < len(tokens):
                defines.append(tokens[i + 1])
                skip_next = True
    return defines


def _includes_from_compile_commands(
    binary_path: str,
    source_files: List[str],
) -> List[str]:
    """Extract ``-I`` paths from a ``compile_commands.json`` near the binary."""
    import json as _json

    cc_path = _find_compile_commands(binary_path)
    if cc_path is None:
        return []

    source_basenames = {os.path.basename(f) for f in source_files}
    source_abspaths = {os.path.abspath(f) for f in source_files}

    try:
        with open(cc_path, "r") as f:
            entries = _json.load(f)
    except (OSError, ValueError):
        return []

    cc_dir = os.path.dirname(os.path.abspath(cc_path))

    for entry in entries:
        entry_file = entry.get("file", "")
        if not os.path.isabs(entry_file):
            entry_file = os.path.join(entry.get("directory", ""), entry_file)
        entry_abs = os.path.abspath(entry_file)

        if entry_abs in source_abspaths or os.path.basename(entry_file) in source_basenames:
            return _extract_includes_from_command(
                entry.get("command", ""),
                entry.get("arguments", []),
                entry.get("directory", cc_dir),
            )
    return []


def _extract_includes_from_command(
    command: str,
    arguments: List[str],
    working_dir: str = "",
) -> List[str]:
    """Pull ``-I`` paths out of a compiler command string or argument list.

    Resolves relative paths against *working_dir* and returns absolute paths.
    """
    import shlex

    tokens = arguments if arguments else shlex.split(command)
    includes: List[str] = []
    skip_next = False
    for i, tok in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if tok.startswith("-I"):
            val = tok[2:]
            if val:
                path = val
            elif i + 1 < len(tokens):
                path = tokens[i + 1]
                skip_next = True
            else:
                continue
            if not os.path.isabs(path) and working_dir:
                path = os.path.join(working_dir, path)
            path = os.path.normpath(path)
            if os.path.isdir(path):
                includes.append(path)
    return includes


def _defines_from_source_scan(source_files: List[str]) -> Dict[str, Optional[str]]:
    """Scan source files for ``#ifdef`` / ``#if defined()`` guards that
    contain HIP/ROCM/AMD substrings and therefore need to be defined when
    compiling with ``hipcc``.
    """
    found: Dict[str, Optional[str]] = {}
    pattern = re.compile(
        r"#\s*if(?:def\s+(\w+)"
        r"|\s+defined\s*\(\s*(\w+)\s*\))"
    )
    for fpath in source_files:
        try:
            with open(fpath, "r") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        for m in pattern.finditer(content):
            name = m.group(1) or m.group(2)
            if name and any(kw in name.upper() for kw in _HIP_KEYWORDS):
                found[name] = None
    return found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_kernel_source(
    kernel_name: str,
    source_dir: str,
    language: Optional[str] = None,
    extra_defines: Optional[List[str]] = None,
    mangled_name: str = "",
) -> Optional[KernelSource]:
    """Find the source code for a kernel.

    Parameters
    ----------
    kernel_name : str
        Kernel name (possibly demangled).
    source_dir : str
        Root directory to search for source files.
    language : str, optional
        "hip" or "triton". Auto-detected if None.
    extra_defines : list of str, optional
        Additional preprocessor defines to include in the reproducer
        (e.g. ``["GGML_USE_HIP", "FOO=1"]``).
    mangled_name : str, optional
        Mangled kernel symbol name from ``dispatch.json``.  Used to
        disambiguate template-instantiation translation units.

    Returns
    -------
    KernelSource or None
        The located kernel source, or None if not found.
    """
    if language is None:
        language = detect_language(kernel_name, source_dir)

    if language == "triton":
        return _find_triton_kernel(kernel_name, source_dir)
    elif language == "hip":
        return _find_hip_kernel(
            kernel_name,
            source_dir,
            extra_defines,
            mangled_name,
        )
    else:
        result = _find_triton_kernel(kernel_name, source_dir)
        if result:
            return result
        return _find_hip_kernel(
            kernel_name,
            source_dir,
            extra_defines,
            mangled_name,
        )
