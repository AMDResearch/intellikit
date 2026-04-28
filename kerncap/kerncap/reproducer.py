"""Standalone reproducer generator.

Takes captured kernel data and located source, generates a self-contained
project that uses kerncap-replay for VA-faithful kernel replay.
"""

import ast
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jinja2

from kerncap.source_finder import KernelSource

logger = logging.getLogger(__name__)


# Triton dtype ``str()`` representations as they appear in legacy
# name_map.json files (captured before the dtype-tagging fix in
# ``triton_capture_hsa.py``).  ``str(tl.bfloat16)`` is ``"bf16"`` etc;
# we map those back to the qualified ``tl.<name>`` Python expression
# the reproducer can pass to ``tl.zeros(dtype=...)``.  New captures
# tag the value as ``{"__triton_dtype__": "<name>"}`` and are
# unambiguous; this table only exists so old captures keep working
# without re-running the (expensive) workload.
_LEGACY_TRITON_DTYPE_STRS = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
    "fp64": "float64",
    "i1":   "int1",
    "i8":   "int8",
    "i16":  "int16",
    "i32":  "int32",
    "i64":  "int64",
    "u8":   "uint8",
    "u16":  "uint16",
    "u32":  "uint32",
    "u64":  "uint64",
    "fp8e4nv":   "float8e4nv",
    "fp8e4b8":   "float8e4b8",
    "fp8e4b15":  "float8e4b15",
    "fp8e5":     "float8e5",
    "fp8e5b16":  "float8e5b16",
}


def _constexpr_repr(v: object) -> str:
    """Render a constexpr value as Python source for the reproducer.

    Triton dtype constexprs (e.g. ``compute_type=tl.bfloat16``) need to
    appear as the qualified ``tl.<name>`` expression in the generated
    file -- not a string literal -- because the kernel passes them to
    APIs like ``tl.zeros(dtype=...)`` that reject bare strings.

    Two encodings are honoured:

    1. ``{"__triton_dtype__": "<name>"}`` -- written by the dtype-aware
       compile-shim in ``triton_capture_hsa.py`` (preferred, lossless).
    2. Bare strings like ``"bf16"`` -- written by older captures whose
       compile shim ``str()``'d the dtype object.  Mapped back to the
       canonical ``tl.<name>`` form via ``_LEGACY_TRITON_DTYPE_STRS``.

    Anything else falls back to ``repr`` so plain literals keep working.
    """
    if isinstance(v, dict) and "__triton_dtype__" in v:
        return f"tl.{v['__triton_dtype__']}"
    if isinstance(v, str) and v in _LEGACY_TRITON_DTYPE_STRS:
        return f"tl.{_LEGACY_TRITON_DTYPE_STRS[v]}"
    return repr(v)


def _get_template_env() -> jinja2.Environment:
    """Create a Jinja2 environment pointing at our templates directory."""
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(templates_dir),
        keep_trailing_newline=True,
    )
    env.filters["pyrepr"] = repr
    env.filters["constrepr"] = _constexpr_repr
    return env


def generate_hsaco_reproducer(
    capture_dir: str,
    output_dir: str,
    kernel_source: Optional[KernelSource] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Generate an HSACO-based reproducer project.

    Uses kerncap-replay for VA-faithful HSA dispatch.  The project
    contains ``capture/``, an unflattened ``kernel_variant.cpp``,
    a ``vfs.yaml`` overlay file, and a Makefile with run/replay/recompile/validate targets.

    Parameters
    ----------
    capture_dir : str
        Directory containing captured data (dispatch.json, kernarg.bin,
        kernel.hsaco, memory_regions.json, memory/).
    output_dir : str
        Where to write the reproducer project.
    kernel_source : KernelSource, optional
        Located kernel source information.
    metadata : dict, optional
        Pre-loaded dispatch.json metadata.

    Returns
    -------
    str
        Path to the reproducer project directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load capture metadata
    if metadata is None:
        dispatch_path = os.path.join(capture_dir, "dispatch.json")
        meta_path = os.path.join(capture_dir, "metadata.json")
        if os.path.isfile(dispatch_path):
            with open(dispatch_path, "r") as f:
                metadata = json.load(f)
        elif os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        else:
            raise FileNotFoundError(f"No dispatch.json or metadata.json in {capture_dir}")

    # Copy capture data
    capture_dest = os.path.join(output_dir, "capture")
    if os.path.realpath(capture_dir) != os.path.realpath(capture_dest):
        if os.path.exists(capture_dest):
            shutil.rmtree(capture_dest)
        shutil.copytree(capture_dir, capture_dest)

    # Derive gpu_arch
    isa_name = metadata.get("isa_name", "")
    if isa_name and "--" in isa_name:
        gpu_arch = isa_name.rsplit("--", 1)[-1]
    elif isa_name and isa_name.startswith("gfx"):
        gpu_arch = isa_name
    else:
        gpu_arch = metadata.get("gpu_arch", "gfx90a")

    kernel_name = metadata.get("demangled_name", metadata.get("kernel_name", "unknown"))

    # Generate Makefile
    makefile_path = os.path.join(output_dir, "Makefile")
    _write_replay_makefile(
        makefile_path,
        kernel_name,
        gpu_arch,
        kernel_source=kernel_source,
    )

    if kernel_source and kernel_source.main_file:
        variant_path = os.path.join(output_dir, "kernel_variant.cpp")
        shutil.copy2(kernel_source.main_file, variant_path)

        original_path = os.path.abspath(kernel_source.main_file)

        # Group all files by original directory for VFS roots.
        vfs_map: Dict[str, List[Tuple[str, str]]] = {}

        main_dir = os.path.dirname(original_path)
        vfs_map.setdefault(main_dir, []).append(
            (os.path.basename(original_path), os.path.abspath(variant_path))
        )

        # Copy dependency headers into deps/ so the reproducer is
        # self-contained for inspection and editing.
        dep_files = [
            f
            for f in kernel_source.source_files
            if os.path.abspath(f) != os.path.abspath(kernel_source.main_file)
        ]
        if dep_files:
            deps_dir = os.path.join(output_dir, "deps")
            os.makedirs(deps_dir, exist_ok=True)

            used_names: Dict[str, str] = {}  # dest_name -> original abs path
            for dep in dep_files:
                dep_abs = os.path.abspath(dep)
                basename = os.path.basename(dep)

                dest_name = basename
                if dest_name in used_names:
                    stem, ext = os.path.splitext(basename)
                    src_dir = os.path.dirname(dep_abs)
                    dir_name = os.path.basename(src_dir) or "root"
                    safe_prefix = dir_name.replace(os.sep, "_")
                    candidate = f"{safe_prefix}__{basename}"
                    if candidate in used_names and used_names[candidate] != dep_abs:
                        hash_digest = hashlib.sha1(dep_abs.encode("utf-8")).hexdigest()[:8]
                        candidate = f"{stem}_{hash_digest}{ext}"
                    dest_name = candidate
                    logger.warning(
                        "Dependency name collision: storing deps/%s as deps/%s",
                        basename,
                        dest_name,
                    )

                used_names[dest_name] = dep_abs

                dest_path = os.path.join(deps_dir, dest_name)
                shutil.copy2(dep_abs, dest_path)

                dep_dir = os.path.dirname(dep_abs)
                vfs_map.setdefault(dep_dir, []).append((basename, os.path.abspath(dest_path)))
                logger.debug("Copied dependency %s -> deps/%s", basename, dest_name)

        vfs_roots = []
        for dir_path, entries in vfs_map.items():
            contents = [
                {"type": "file", "name": name, "external-contents": local}
                for name, local in entries
            ]
            vfs_roots.append(
                {
                    "type": "directory",
                    "name": dir_path,
                    "contents": contents,
                }
            )

        vfs_content = {"version": 0, "roots": vfs_roots}
        vfs_path = os.path.join(output_dir, "vfs.yaml")
        with open(vfs_path, "w") as f:
            json.dump(vfs_content, f, indent=2)

    return output_dir


def _write_replay_makefile(
    path: str,
    kernel_name: str,
    gpu_arch: str,
    kernel_source: Optional[KernelSource] = None,
) -> None:
    """Write a Makefile that uses kerncap-replay."""
    try:
        from kerncap import _get_replay_path

        replay_default = _get_replay_path()
    except (ImportError, FileNotFoundError):
        replay_default = "kerncap-replay"

    has_compilable = kernel_source is not None and bool(kernel_source.compile_command)

    phony_targets = ["run", "replay", "validate"]
    if has_compilable:
        phony_targets.extend(["recompile", "run-variant", "validate-variant"])

    lines = [
        "# Makefile — generated by kerncap (VA-faithful replay)",
        f"# Kernel: {kernel_name}",
        f"# GPU:    {gpu_arch}",
        "",
        f"REPLAY ?= {replay_default}",
        "CAPTURE_DIR ?= capture",
        f"GPU_ARCH ?= {gpu_arch}",
        "",
        ".PHONY: " + " ".join(phony_targets),
        "",
        "# Replay the captured kernel (baseline)",
        "run:",
        "\t$(REPLAY) $(CAPTURE_DIR)",
        "",
        "replay:",
        "\t$(REPLAY) $(CAPTURE_DIR) --json",
        "",
        "validate:",
        "\t$(REPLAY) $(CAPTURE_DIR) --dump-output",
        "",
    ]

    if has_compilable:
        import shlex

        tokens = shlex.split(kernel_source.compile_command)
        new_tokens = []
        skip_next = False
        for i, tok in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue
            if tok == "-c":
                continue
            if tok == "-o":
                skip_next = True
                continue
            if tok.startswith("-o") and len(tok) > 2:
                continue
            new_tokens.append(tok)

        clean_cmd = " ".join(shlex.quote(t) for t in new_tokens)
        compile_dir = kernel_source.compile_dir

        lines.extend(
            [
                "# Recompile the edited kernel_variant.cpp into a new HSACO via Clang VFS overlay.",
                "# The VFS overlay tricks the compiler into using the edited file in place of the original.",
                "# --no-gpu-bundle-output produces a raw code object, so no unbundling is needed.",
                "recompile: kernel_variant.cpp vfs.yaml",
                '\t@echo "Recompiling optimized HSACO via VFS overlay..."',
                f"\tcd {shlex.quote(compile_dir)} && \\",
                f"\t{clean_cmd} -ivfsoverlay $(PWD)/vfs.yaml --cuda-device-only --no-gpu-bundle-output -o $(PWD)/optimized.hsaco",
                "",
                "# Replay with the recompiled HSACO",
                "run-variant: optimized.hsaco",
                "\t$(REPLAY) $(CAPTURE_DIR) --hsaco optimized.hsaco",
                "",
                "# Dump post-execution output for the recompiled HSACO",
                "validate-variant: optimized.hsaco",
                "\t$(REPLAY) $(CAPTURE_DIR) --hsaco optimized.hsaco --dump-output",
                "",
            ]
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


_TRITON_SAFE_IMPORT_ROOTS = frozenset(
    {
        "triton",
        "math",
        "functools",
        "os",
        "sys",
        "typing",
        "dataclasses",
        "enum",
        "itertools",
        "operator",
        "abc",
        "collections",
        "builtins",
    }
)


def _extract_triton_kernel_standalone(
    source_file: str,
    kernel_function: str,
    output_path: str,
) -> None:
    """Extract a minimal standalone Triton kernel module from a source file.

    Parses *source_file*, collects every ``@triton.jit`` / ``@triton.autotune``
    decorated function plus only the triton/stdlib imports, and writes a new
    ``output_path`` that can be imported without pulling in any heavy
    framework code (e.g. vLLM custom-op registrations).

    Parameters
    ----------
    source_file : str
        Path to the Python file that contains the target kernel.
    kernel_function : str
        Name (or substring) of the kernel function to extract.
    output_path : str
        Destination file path for the generated standalone module.
    """
    with open(source_file, "r") as f:
        source = f.read()
    source_lines = source.splitlines()

    tree = ast.parse(source, filename=source_file)

    # ------------------------------------------------------------------ #
    # Collect all @triton.jit / @triton.autotune decorated functions.      #
    # We include all of them because helper kernels called by the target   #
    # are typically also decorated, and the set is usually small.          #
    # ------------------------------------------------------------------ #
    triton_decorator_names = {"triton.jit", "jit", "triton.autotune", "autotune"}

    def _decorator_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: List[str] = []
            cur: ast.expr = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            return ".".join(reversed(parts))
        if isinstance(node, ast.Call):
            return _decorator_name(node.func)
        return ""

    _autotune_decorator_names = {"triton.autotune", "autotune"}

    def _node_source_with_decorators(func_node: ast.FunctionDef) -> str:
        """Return the full source text for *func_node*, stripping @triton.autotune.

        Autotune decorators reference module-level variables (e.g.
        ``autotune_configs``) that the standalone module won't have.
        We keep ``@triton.jit`` while removing ``@triton.autotune`` so the
        extracted kernel remains directly callable with pinned meta-parameters.
        """
        kept_decorator_lines: List[str] = []
        for dec in func_node.decorator_list:
            if _decorator_name(dec) in _autotune_decorator_names:
                continue
            dec_start = dec.lineno - 1
            dec_end = dec.end_lineno  # AST end_lineno is 1-based inclusive; using it as the slice end makes source_lines[dec_start:dec_end] include the final decorator line.
            kept_decorator_lines.extend(source_lines[dec_start:dec_end])

        func_start = func_node.lineno - 1
        func_end = func_node.end_lineno
        func_body_lines = source_lines[func_start:func_end]

        return "\n".join(kept_decorator_lines + func_body_lines)

    triton_funcs: List[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if _decorator_name(dec) in triton_decorator_names:
                triton_funcs.append(node)
                break

    if not any(f.name == kernel_function for f in triton_funcs):
        found = [f.name for f in triton_funcs]
        raise ValueError(
            f"Kernel function '{kernel_function}' not found among Triton-decorated "
            f"functions in {source_file}. Found: {found}"
        )

    # ------------------------------------------------------------------ #
    # Collect safe (triton / stdlib) imports from the entire AST.          #
    # ast.walk finds imports inside try/except, if-guards, etc. that a     #
    # top-level-only scan would miss (e.g. vLLM wraps triton imports in    #
    # try/except blocks).                                                  #
    # ------------------------------------------------------------------ #
    import_lines: List[str] = []
    seen_imports: set = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            safe_aliases = [
                alias
                for alias in node.names
                if alias.name.split(".")[0] in _TRITON_SAFE_IMPORT_ROOTS
            ]
            if not safe_aliases:
                continue
            for alias in safe_aliases:
                seg = f"import {alias.name}"
                if alias.asname:
                    seg += f" as {alias.asname}"
                if seg not in seen_imports:
                    seen_imports.add(seg)
                    import_lines.append(seg)
            continue
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # skip relative imports
            root = (node.module or "").split(".")[0]
            if root not in _TRITON_SAFE_IMPORT_ROOTS:
                continue
        else:
            continue

        seg = ast.get_source_segment(source, node)
        if seg is None:
            seg = ast.unparse(node)
        if seg and seg not in seen_imports:
            seen_imports.add(seg)
            import_lines.append(seg)

    # Every standalone Triton kernel file needs these unconditionally.
    if "import triton" not in seen_imports:
        import_lines.insert(0, "import triton")
    if not any("triton.language" in s for s in seen_imports):
        import_lines.insert(1, "import triton.language as tl")

    # ------------------------------------------------------------------ #
    # Build and write the standalone file.                                 #
    # ------------------------------------------------------------------ #
    parts: List[str] = [
        "# Standalone Triton kernel module — generated by kerncap.",
        "# Contains only the captured kernel and its direct Triton helpers.",
        "",
        *import_lines,
        "",
    ]
    for func_node in triton_funcs:
        parts.append(_node_source_with_decorators(func_node))
        parts.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(parts) + "\n")

    logger.debug(
        "Wrote standalone kernel module: %s (%d triton function(s))",
        output_path,
        len(triton_funcs),
    )


def generate_triton_reproducer(
    capture_dir: str,
    kernel_source: KernelSource,
    output_dir: str,
) -> str:
    """Generate a Triton reproducer project.

    Parameters
    ----------
    capture_dir : str
        Directory containing captured data (metadata.json, arg_*.bin).
    kernel_source : KernelSource
        Located kernel source information.
    output_dir : str
        Where to write the reproducer project.

    Returns
    -------
    str
        Path to the reproducer project directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    meta_path = os.path.join(capture_dir, "metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    capture_dest = os.path.join(output_dir, "capture")
    if os.path.realpath(capture_dir) != os.path.realpath(capture_dest):
        if os.path.exists(capture_dest):
            shutil.rmtree(capture_dest)
        shutil.copytree(capture_dir, capture_dest)

    os.makedirs(os.path.join(output_dir, "reference_output"), exist_ok=True)
    env = _get_template_env()

    main_file = kernel_source.main_file
    main_dir = os.path.dirname(main_file)

    # If the kernel lives inside a Python package (directory has __init__.py),
    # extract a minimal standalone Triton module instead of copying the whole
    # package.  Copying the entire package risks pulling in module-level side
    # effects (e.g. vLLM custom-op registrations via direct_register_custom_op)
    # that collide with the already-registered ops from the editable install.
    pkg_init = os.path.join(main_dir, "__init__.py")
    is_standalone = os.path.isfile(pkg_init)
    if is_standalone:
        variant_path = os.path.join(output_dir, "kernel_variant.py")
        _extract_triton_kernel_standalone(
            main_file,
            kernel_source.kernel_function,
            variant_path,
        )
        kernel_module = "kernel_variant"
        logger.info(
            "Extracted standalone kernel module: %s (avoided full package copy)",
            variant_path,
        )
    else:
        # No package structure: copy individual files to flat directory.
        # Unlike the HIP path, Triton files use Python module imports
        # (not filesystem-relative #include), so flattening is safe here.
        for src_file in kernel_source.source_files:
            dest = os.path.join(output_dir, os.path.basename(src_file))
            if not os.path.exists(dest):
                shutil.copy2(src_file, dest)
        kernel_module = Path(main_file).stem

    autotune_config = metadata.get("autotune_config")
    if autotune_config is None:
        config_args = {
            a["name"]: a["value"] for a in metadata.get("args", []) if a.get("is_autotune_config")
        }
        if config_args:
            autotune_config = {"kwargs": config_args}

    context = {
        "kernel_name": metadata["kernel_name"],
        "grid": metadata["grid"],
        "block": metadata["block"],
        "args": metadata.get("args", []),
        "kernel_module": kernel_module,
        "kernel_function": kernel_source.kernel_function,
        "autotune_config": autotune_config,
        "autotune_stripped": is_standalone and autotune_config is not None,
    }

    # Render reproducer.py
    template = env.get_template("triton_reproducer.py.j2")
    reproducer_path = os.path.join(output_dir, "reproducer.py")
    with open(reproducer_path, "w") as f:
        f.write(template.render(**context))

    # Make it executable
    os.chmod(reproducer_path, 0o755)

    return output_dir


# ---------------------------------------------------------------------------
# HSA-backed Triton reproducer (editable Python loop, but driven by the
# byte-faithful artifacts captured at the HSA layer).
# ---------------------------------------------------------------------------


# Triton scalar (by_value) types -> struct format char + Python decoder
_TRITON_SCALAR_FORMATS: Dict[str, Tuple[str, type]] = {
    "i1": ("?", bool),
    "i8": ("b", int),
    "u8": ("B", int),
    "i16": ("h", int),
    "u16": ("H", int),
    "i32": ("i", int),
    "u32": ("I", int),
    "i64": ("q", int),
    "u64": ("Q", int),
    "fp16": ("e", float),
    "bf16": ("H", int),  # raw bits; user can reinterpret if they care
    "fp32": ("f", float),
    "f32": ("f", float),
    "fp64": ("d", float),
    "f64": ("d", float),
}


def _decode_scalar(kernarg_bytes: bytes, offset: int, size: int, value_type: str):
    """Decode a single by_value kernarg slot into a Python literal."""
    import struct

    fmt = _TRITON_SCALAR_FORMATS.get(value_type)
    if fmt is None:
        # Fall back to size-based heuristic: signed integer
        size_to_fmt = {1: "b", 2: "h", 4: "i", 8: "q"}
        fmt_char = size_to_fmt.get(size, "")
        if not fmt_char:
            return None
        return struct.unpack_from("<" + fmt_char, kernarg_bytes, offset)[0]
    fmt_char, _decoder = fmt
    try:
        return struct.unpack_from("<" + fmt_char, kernarg_bytes, offset)[0]
    except struct.error:
        return None


def _find_region_for_pointer(pointer: int, regions: List[dict]) -> Optional[Tuple[dict, int]]:
    """Return the (region, byte_offset_within_region) for *pointer*."""
    for r in regions:
        base = int(r.get("base", 0))
        size = int(r.get("size", 0))
        if base <= pointer < base + size:
            return r, pointer - base
    return None


def _select_name_map_row(
    name_map: List[dict],
    hsaco_sha256: str,
    user_name: str,
) -> Optional[dict]:
    """Pick the best name_map row for the captured dispatch.

    Prefer exact SHA-256 match; fall back to user_name match (latest
    row wins, since the observer attaches layout to the most recent).
    """
    for row in name_map:
        if hsaco_sha256 and row.get("hsaco_sha256") == hsaco_sha256:
            return row
    matching = [
        r
        for r in name_map
        if user_name and (r.get("user_name") == user_name or user_name in r.get("user_name", ""))
    ]
    if matching:
        return matching[-1]
    return None


def generate_triton_hsa_reproducer(
    capture_dir: str,
    kernel_source: KernelSource,
    output_dir: str,
) -> str:
    """Generate an editable Triton reproducer from HSA-captured artifacts.

    Combines:
      * ``capture/metadata.json`` (kernarg slot table from ELF metadata)
      * ``capture/kernarg_raw.bin`` (exact kernarg buffer bytes)
      * ``capture/memory_regions.json`` + ``capture/memory/*`` (pointer-backed
        buffers, byte-exact)
      * ``capture/name_map.json`` (Triton signature, constexprs, launch
        attributes, optional tensor_layout from the runtime observer)

    Produces a self-contained ``reproducer.py`` plus ``kernel_variant.py``
    (extracted standalone Triton module).  Editing ``kernel_variant.py``
    and rerunning ``python3 reproducer.py`` re-JITs through Triton's
    compiler -- there is no separate "make recompile" step.

    Returns *output_dir*.
    """
    os.makedirs(output_dir, exist_ok=True)

    capture_dest = os.path.join(output_dir, "capture")
    if os.path.realpath(capture_dir) != os.path.realpath(capture_dest):
        if os.path.exists(capture_dest):
            shutil.rmtree(capture_dest)
        shutil.copytree(capture_dir, capture_dest)

    os.makedirs(os.path.join(output_dir, "reference_output"), exist_ok=True)

    meta_path = os.path.join(capture_dest, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    kernarg_path = os.path.join(capture_dest, "kernarg_raw.bin")
    with open(kernarg_path, "rb") as f:
        kernarg_bytes = f.read()

    regions_path = os.path.join(capture_dest, "memory_regions.json")
    regions: List[dict] = []
    if os.path.isfile(regions_path):
        with open(regions_path) as f:
            regions = json.load(f).get("regions", [])

    name_map_path = os.path.join(capture_dest, "name_map.json")
    name_map: List[dict] = []
    if os.path.isfile(name_map_path):
        with open(name_map_path) as f:
            name_map = json.load(f)

    hsaco_sha256 = metadata.get("hsaco_sha256", "")
    user_name = metadata.get("triton_user_name") or metadata.get("kernel_name", "")
    row = _select_name_map_row(name_map, hsaco_sha256, user_name) or {}

    signature: Dict[str, str] = row.get("signature") or {}
    param_names: List[str] = row.get("param_names") or []
    constexprs: Dict[str, object] = row.get("constexpr_values") or {}
    launch: Dict[str, int] = row.get("launch") or {}
    tensor_layout: List[dict] = row.get("tensor_layout") or []
    layout_by_name = {lay["name"]: lay for lay in tensor_layout if "name" in lay}

    # Build the ordered list of params that *actually* take a kernarg slot,
    # i.e. all JIT params minus the constexprs (which Triton folds into
    # the IR).  The AMDGPU ABI emits these in source order, so a
    # positional walk over the kernarg slots reconstructs the names that
    # ``triton.compiler.compile`` saw -- which is exactly what
    # ``JITFunction.run`` expects as kwargs.
    constexpr_param_names = set(constexprs.keys())
    constexpr_signature_names = {n for n, t in signature.items() if str(t).lower() == "constexpr"}
    constexpr_param_names |= constexpr_signature_names
    runtime_params: List[str] = [p for p in param_names if p not in constexpr_param_names]

    # Walk kernarg slots in offset order, classify each into pointer / scalar.
    slots = metadata.get("kernarg_slots") or []
    pointer_args: List[dict] = []
    scalar_args: List[dict] = []
    runtime_cursor = 0

    for slot in slots:
        kind = (slot.get("value_kind") or "").lower()
        raw_slot_name = slot.get("name", "") or ""
        # Strip names auto-injected by AMDGPU runtime (hidden args, padding).
        if kind.startswith("hidden_"):
            continue
        # Prefer the AMDGPU metadata name when present; otherwise consume
        # the next runtime param in source order.  If the slot table is
        # longer than the JIT signature it usually means Triton emitted
        # trailing ``readnone`` placeholder pointers (a known quirk of
        # the AMDGPU backend) -- those have no user-facing param and
        # passing a bogus kwarg would make ``JITFunction.run`` reject
        # the launch, so we skip them.
        if raw_slot_name:
            slot_name = raw_slot_name
        elif runtime_cursor < len(runtime_params):
            slot_name = runtime_params[runtime_cursor]
            runtime_cursor += 1
        elif runtime_params:
            logger.debug(
                "Skipping kernarg slot %d (kind=%r): no matching JIT param "
                "(likely a trailing readnone placeholder).",
                slot["offset"],
                kind,
            )
            continue
        else:
            slot_name = f"slot_{slot['offset']}"

        if kind in ("global_buffer", "dynamic_shared_pointer"):
            ptr = int.from_bytes(
                kernarg_bytes[slot["offset"] : slot["offset"] + slot["size"]],
                "little",
                signed=False,
            )
            region_info = _find_region_for_pointer(ptr, regions)
            triton_dtype = signature.get(slot_name, "*i8")
            layout = layout_by_name.get(slot_name)
            shape = layout["shape"] if layout else None
            stride = layout["stride"] if layout else None
            storage_offset = int(layout["storage_offset"]) if layout else 0
            if region_info is None:
                logger.warning(
                    "Pointer arg '%s' (0x%x) does not map to any captured memory region; skipping.",
                    slot_name,
                    ptr,
                )
                continue
            region, byte_offset = region_info
            base_hex = format(int(region["base"]), "x")
            pointer_args.append(
                {
                    "index": len(pointer_args),
                    "name": slot_name,
                    "triton_dtype": triton_dtype,
                    "region_file": f"region_{base_hex}.bin",
                    "byte_offset": byte_offset,
                    "shape": shape,
                    "stride": stride,
                    "storage_offset": storage_offset,
                }
            )
        elif kind == "by_value":
            if slot_name in constexpr_param_names:
                # Constexpr: value lives in name_map, not the kernarg buffer.
                continue
            # AMDGPU metadata frequently leaves ``value_type`` empty for
            # Triton-emitted scalars, so fall back to the JIT signature
            # (e.g. ``dropout_p`` -> ``fp32``).  Without this, a 4-byte
            # float would be misdecoded as ``int32`` and the reproducer
            # would launch with a type-mismatched argument.
            value_type = (slot.get("value_type") or "").lower()
            if not value_type and slot_name in signature:
                value_type = signature[slot_name].lstrip("*").lower()
            value = _decode_scalar(kernarg_bytes, slot["offset"], slot["size"], value_type)
            scalar_args.append(
                {
                    "name": slot_name,
                    "value": value,
                    "value_type": value_type or "?",
                    "offset": slot["offset"],
                }
            )
        else:
            logger.debug("Skipping kernarg slot kind=%r name=%r", kind, slot_name)

    # Resolve module / function name for the reproducer's import line.
    main_file = kernel_source.main_file
    main_dir = os.path.dirname(main_file)
    pkg_init = os.path.join(main_dir, "__init__.py")
    is_standalone = os.path.isfile(pkg_init)
    if is_standalone:
        variant_path = os.path.join(output_dir, "kernel_variant.py")
        _extract_triton_kernel_standalone(main_file, kernel_source.kernel_function, variant_path)
        kernel_module = "kernel_variant"
        logger.info(
            "Extracted standalone Triton module: %s (avoided full package copy)",
            variant_path,
        )
    else:
        for src_file in kernel_source.source_files:
            dest = os.path.join(output_dir, os.path.basename(src_file))
            if not os.path.exists(dest):
                shutil.copy2(src_file, dest)
        kernel_module = Path(main_file).stem

    grid = metadata.get("grid", {"x": 1, "y": 1, "z": 1})
    block = metadata.get("block", {"x": 1, "y": 1, "z": 1})

    context = {
        "kernel_name": metadata.get("kernel_name", user_name),
        "kernel_module": kernel_module,
        "kernel_function": kernel_source.kernel_function,
        "grid": grid,
        "block": block,
        "pointer_args": pointer_args,
        "scalar_args": scalar_args,
        "constexprs": constexprs,
        "launch": launch,
        "deps_dir": "deps" if os.path.isdir(os.path.join(output_dir, "deps")) else "",
    }

    env = _get_template_env()
    template = env.get_template("triton_reproducer_hsa.py.j2")
    reproducer_path = os.path.join(output_dir, "reproducer.py")
    with open(reproducer_path, "w") as f:
        f.write(template.render(**context))
    os.chmod(reproducer_path, 0o755)

    return output_dir
