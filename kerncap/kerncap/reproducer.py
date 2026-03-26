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


def _get_template_env() -> jinja2.Environment:
    """Create a Jinja2 environment pointing at our templates directory."""
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(templates_dir),
        keep_trailing_newline=True,
    )


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

    def _node_source_with_decorators(func_node: ast.FunctionDef) -> str:
        """Return the full source text for *func_node* including decorators."""
        if func_node.decorator_list:
            start = func_node.decorator_list[0].lineno - 1
        else:
            start = func_node.lineno - 1
        end = func_node.end_lineno  # already 1-indexed, exclusive as slice end
        return "\n".join(source_lines[start:end])

    triton_funcs: List[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if _decorator_name(dec) in triton_decorator_names:
                triton_funcs.append(node)
                break

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
            safe = any(
                alias.name.split(".")[0] in _TRITON_SAFE_IMPORT_ROOTS for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # skip relative imports
            root = (node.module or "").split(".")[0]
            safe = root in _TRITON_SAFE_IMPORT_ROOTS
        else:
            continue

        if not safe:
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
    if os.path.isfile(pkg_init):
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

    context = {
        "kernel_name": metadata["kernel_name"],
        "grid": metadata["grid"],
        "block": metadata["block"],
        "args": metadata.get("args", []),
        "kernel_module": kernel_module,
        "kernel_function": kernel_source.kernel_function,
        "autotune_config": metadata.get("autotune_config"),
    }

    # Render reproducer.py
    template = env.get_template("triton_reproducer.py.j2")
    reproducer_path = os.path.join(output_dir, "reproducer.py")
    with open(reproducer_path, "w") as f:
        f.write(template.render(**context))

    # Make it executable
    os.chmod(reproducer_path, 0o755)

    return output_dir
