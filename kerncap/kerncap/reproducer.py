"""Standalone reproducer generator.

Takes captured kernel data and located source, generates a self-contained
project that uses kerncap-replay for VA-faithful kernel replay.
"""

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
            raise FileNotFoundError(
                f"No dispatch.json or metadata.json in {capture_dir}"
            )

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
        makefile_path, kernel_name, gpu_arch,
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
            f for f in kernel_source.source_files
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
                    # Collision: rename both the earlier entry and this one
                    stem, ext = os.path.splitext(basename)
                    counter = 1

                    # Rename the previously copied file if it hasn't been
                    # disambiguated yet (its dest_name still equals basename)
                    prev_abs = used_names[dest_name]
                    if prev_abs is not None:
                        prev_new = f"{stem}_{counter}{ext}"
                        while prev_new in used_names:
                            counter += 1
                            prev_new = f"{stem}_{counter}{ext}"
                        prev_dest = os.path.join(deps_dir, dest_name)
                        prev_new_dest = os.path.join(deps_dir, prev_new)
                        os.rename(prev_dest, prev_new_dest)
                        logger.warning(
                            "Dependency name collision: renamed deps/%s -> deps/%s (from %s)",
                            dest_name, prev_new, prev_abs,
                        )
                        used_names[prev_new] = prev_abs
                        used_names[dest_name] = None  # sentinel: slot taken, file moved

                        counter += 1

                    dest_name = f"{stem}_{counter}{ext}"
                    while dest_name in used_names:
                        counter += 1
                        dest_name = f"{stem}_{counter}{ext}"
                    logger.warning(
                        "Dependency name collision: storing deps/%s as deps/%s",
                        basename, dest_name,
                    )

                used_names[dest_name] = dep_abs

                dest_path = os.path.join(deps_dir, dest_name)
                shutil.copy2(dep_abs, dest_path)

                dep_dir = os.path.dirname(dep_abs)
                vfs_map.setdefault(dep_dir, []).append(
                    (basename, os.path.abspath(dest_path))
                )
                logger.debug("Copied dependency %s -> deps/%s", basename, dest_name)

        vfs_roots = []
        for dir_path, entries in vfs_map.items():
            contents = [
                {"type": "file", "name": name, "external-contents": local}
                for name, local in entries
            ]
            vfs_roots.append({
                "type": "directory",
                "name": dir_path,
                "contents": contents,
            })

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

        lines.extend([
            "# Recompile the edited kernel_variant.cpp into a new HSACO via Clang VFS overlay.",
            "# The VFS overlay tricks the compiler into using the edited file in place of the original.",
            "# --no-gpu-bundle-output produces a raw code object, so no unbundling is needed.",
            "recompile: kernel_variant.cpp vfs.yaml",
            f"\t@echo \"Recompiling optimized HSACO via VFS overlay...\"",
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
        ])

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

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
    # copy the entire package directory so relative imports keep working.
    pkg_init = os.path.join(main_dir, "__init__.py")
    if os.path.isfile(pkg_init):
        pkg_name = os.path.basename(main_dir)
        pkg_dest = os.path.join(output_dir, pkg_name)
        if os.path.exists(pkg_dest):
            shutil.rmtree(pkg_dest)
        shutil.copytree(main_dir, pkg_dest)
        kernel_module = f"{pkg_name}.{Path(main_file).stem}"
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
