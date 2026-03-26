"""Python-level capture for Triton kernels.

Intercepts Triton kernel calls at the Python level to capture arguments
including tensors, scalars, and constexpr values that are not visible
at the HSA dispatch level.

The capture works by generating a hook script that monkey-patches
``triton.runtime.jit.JITFunction.run`` before executing the user's
application.  When the target kernel fires, all Python-level arguments
(torch tensors, scalars, None, etc.) are serialised to disk alongside
a ``metadata.json`` that the reproducer generator consumes.
"""

import os
import subprocess
import tempfile
import textwrap
from typing import List, Optional


# The hook script is written to a temp file and executed as the main
# script, with the user's original script passed as argv[1].
_CAPTURE_HOOK = textwrap.dedent('''\
    """kerncap Triton capture hook — auto-generated, do not edit."""
    import sys
    import os
    import json
    import inspect
    import runpy

    _target_kernel = os.environ["KERNCAP_KERNEL"]
    _output_dir = os.environ["KERNCAP_OUTPUT"]
    _dispatch = int(os.environ.get("KERNCAP_DISPATCH", "-1"))
    _call_count = 0
    _captured = False


    def _install_hook():
        import triton
        import triton.runtime.jit

        _autotuner_config_keys = set()
        # Tensor args from the capture, saved so the Autotuner hook can
        # re-snapshot them AFTER the autotuner selects the best config.
        _pending_ref = [None]  # use list for nonlocal mutation in Py 3

        # Hook the Autotuner so we can:
        #   1. Learn which kwargs come from its configs (BLOCK_M, etc.)
        #   2. Save reference outputs AFTER the best-config run (not
        #      after the first benchmark run, which may use config[0]).
        try:
            from triton.runtime.autotuner import Autotuner
            import torch as _torch

            _OrigAutotunerRun = Autotuner.run

            def _hooked_autotuner_run(self_at, *args, **kwargs):
                fn_name = self_at.fn.fn.__name__
                if _target_kernel in fn_name or fn_name in _target_kernel:
                    if self_at.configs:
                        _autotuner_config_keys.update(
                            self_at.configs[0].kwargs.keys()
                        )
                result = _OrigAutotunerRun(self_at, *args, **kwargs)
                # After ALL autotuner configs have been benchmarked and
                # the final best-config run is complete, snapshot the
                # tensors — they now hold the definitive output.
                if _pending_ref[0] is not None:
                    _torch.cuda.synchronize()
                    _save_reference_outputs(_pending_ref[0])
                    # Capture the best autotuner config so the reproducer
                    # can pin it and get bit-identical results.
                    _save_autotune_config(self_at)
                    _pending_ref[0] = None
                return result

            Autotuner.run = _hooked_autotuner_run
        except (ImportError, AttributeError):
            pass

        _OrigRun = triton.runtime.jit.JITFunction.run

        def _hooked_run(self_jit, *args, grid, warmup=False, **kwargs):
            global _call_count, _captured
            fn_name = self_jit.fn.__name__

            should_capture = (
                not _captured and not warmup
                and (_target_kernel in fn_name or fn_name in _target_kernel)
                and (_dispatch < 0 or _call_count == _dispatch)
            )

            if not _captured and not warmup and (
                _target_kernel in fn_name or fn_name in _target_kernel
            ):
                _call_count += 1

            if should_capture:
                _captured = True
                # Save input tensors BEFORE the kernel runs
                tensor_args = _save_capture(
                    self_jit, args, kwargs, grid, _autotuner_config_keys,
                )
                # Store tensor refs BEFORE _OrigRun so the autotuner
                # hook can find them even if this config raises (the
                # autotuner swallows per-config exceptions internally).
                _pending_ref[0] = tensor_args
                # Run the kernel
                result = _OrigRun(
                    self_jit, *args, grid=grid, warmup=warmup, **kwargs,
                )
                # For non-autotuned kernels, save ref outputs now.
                # For autotuned kernels, the autotuner hook will save
                # after the best config run completes.
                if not _autotuner_config_keys:
                    import torch as _torch
                    _torch.cuda.synchronize()
                    _save_reference_outputs(tensor_args)
                    _pending_ref[0] = None
                return result

            return _OrigRun(self_jit, *args, grid=grid, warmup=warmup, **kwargs)

        triton.runtime.jit.JITFunction.run = _hooked_run


    def _resolve_grid(grid, all_args=None):
        """Normalise a Triton grid to a 3-tuple of ints.

        Grid lambdas (e.g. ``lambda meta: (cdiv(N, meta['BLOCK_M']), …)``)
        need a *meta* dict populated with scalar/constexpr values.
        """
        if callable(grid):
            if all_args:
                # Build a meta dict from scalar/constexpr args
                meta = {k: v for k, v in all_args.items()
                        if isinstance(v, (int, float, bool))}
                try:
                    return _normalize_grid(grid(meta))
                except Exception:
                    pass
            try:
                return _normalize_grid(grid({}))
            except Exception:
                return (1, 1, 1)
        return _normalize_grid(grid)


    def _normalize_grid(grid):
        """Convert a grid value to a 3-tuple of ints."""
        if isinstance(grid, int):
            return (grid, 1, 1)
        grid = tuple(int(g) for g in grid)
        return grid + (1,) * max(0, 3 - len(grid))


    def _save_tensor_storage(tensor, filepath):
        """Save a tensor's full underlying storage to preserve stride layout.

        Unlike _save_tensor, this does NOT call .contiguous(), so non-standard
        strides (e.g. padded row layouts) are preserved in the on-disk bytes.
        The storage is moved to CPU first; the caller must record tensor.stride()
        and tensor.storage_offset() in metadata so the reproducer can reconstruct
        the original view via torch.as_strided.
        """
        import torch
        cpu_tensor = tensor.detach().cpu()
        storage = cpu_tensor.untyped_storage()
        flat = torch.empty(storage.nbytes(), dtype=torch.uint8)
        flat.set_(storage)
        flat.numpy().tofile(filepath)

    def _save_tensor(tensor, filepath):
        """Save a tensor's logical values as a contiguous binary file.

        Used for reference outputs where we compare logical values, not layouts.
        """
        import torch
        cpu_tensor = tensor.detach().cpu().contiguous()
        if cpu_tensor.dtype == torch.bfloat16:
            cpu_tensor.view(torch.uint16).numpy().tofile(filepath)
        else:
            cpu_tensor.numpy().tofile(filepath)


    def _save_capture(jit_fn, args, kwargs, grid, autotuner_config_keys):
        """Save pre-kernel tensor inputs and scalar args.

        Returns a list of (index, name, tensor) for all tensor args so
        the caller can save their post-kernel state as reference outputs.
        """
        import torch
        import numpy as np

        os.makedirs(_output_dir, exist_ok=True)

        param_names = list(inspect.signature(jit_fn.fn).parameters.keys())

        # Merge positional + keyword args by parameter name (needed
        # both for metadata and for resolving callable grids).
        all_args = {}
        for i, name in enumerate(param_names):
            if i < len(args):
                all_args[name] = args[i]
        all_args.update(kwargs)

        grid_val = _resolve_grid(grid, all_args)

        metadata = {
            "kernel_name": jit_fn.fn.__name__,
            "grid": {"x": grid_val[0], "y": grid_val[1], "z": grid_val[2]},
            "block": {"x": 1, "y": 1, "z": 1},
            "language": "triton",
            "args": [],
        }

        tensor_args = []

        for i, name in enumerate(param_names):
            val = all_args.get(name)
            is_config = name in autotuner_config_keys

            if isinstance(val, torch.Tensor):
                filename = f"arg_{i}.bin"
                filepath = os.path.join(_output_dir, filename)
                _save_tensor_storage(val, filepath)

                metadata["args"].append({
                    "index": i,
                    "name": name,
                    "is_pointer": True,
                    "is_const": False,
                    "is_autotune_config": is_config,
                    "file": filename,
                    "ref_output_file": f"ref_output_{i}.bin",
                    "buffer_size": val.nelement() * val.element_size(),
                    "shape": list(val.shape),
                    "strides": list(val.stride()),
                    "storage_offset": val.storage_offset(),
                    "torch_dtype": str(val.dtype),
                })
                tensor_args.append((i, name, val))

            elif val is None:
                metadata["args"].append({
                    "index": i,
                    "name": name,
                    "is_pointer": False,
                    "is_const": True,
                    "is_autotune_config": is_config,
                    "value": None,
                    "type": "NoneType",
                })

            elif isinstance(val, (int, float, bool)):
                metadata["args"].append({
                    "index": i,
                    "name": name,
                    "is_pointer": False,
                    "is_const": True,
                    "is_autotune_config": is_config,
                    "value": val,
                    "type": type(val).__name__,
                })

            else:
                metadata["args"].append({
                    "index": i,
                    "name": name,
                    "is_pointer": False,
                    "is_const": True,
                    "is_autotune_config": is_config,
                    "value": str(val),
                    "type": type(val).__name__,
                })

        with open(os.path.join(_output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"[kerncap] Captured {jit_fn.fn.__name__} "
            f"({len(metadata['args'])} args) -> {_output_dir}",
            flush=True,
        )

        return tensor_args


    def _save_reference_outputs(tensor_args):
        """Save post-kernel tensor state as reference outputs."""
        import torch

        for i, name, tensor in tensor_args:
            filepath = os.path.join(_output_dir, f"ref_output_{i}.bin")
            _save_tensor(tensor, filepath)

        print(
            f"[kerncap] Saved {len(tensor_args)} reference outputs",
            flush=True,
        )


    def _save_autotune_config(autotuner):
        """Persist the autotuner's selected best config into metadata."""
        if not autotuner.cache:
            return
        best = list(autotuner.cache.values())[-1]
        config_info = {"kwargs": dict(best.kwargs)}
        for attr in ("num_warps", "num_stages", "num_ctas"):
            val = getattr(best, attr, None)
            if val is not None:
                config_info[attr] = val

        meta_path = os.path.join(_output_dir, "metadata.json")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        metadata["autotune_config"] = config_info
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"[kerncap] Pinned autotune config: {config_info}",
            flush=True,
        )


    # ---- main ---------------------------------------------------------
    _install_hook()

    _user_script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(_user_script, run_name="__main__")
''')


def run_triton_capture(
    kernel_name: str,
    cmd: List[str],
    output_dir: str,
    dispatch: int = -1,
    timeout: int = 300,
) -> str:
    """Capture a Triton kernel's Python-level arguments.

    Generates a temporary hook script that patches ``JITFunction.run``,
    then re-invokes the user's Python command through it so that the
    first non-warmup call to the target kernel is intercepted and all
    arguments are serialised to *output_dir*.

    Parameters
    ----------
    kernel_name : str
        Kernel function name (or substring).
    cmd : list[str]
        The user's command, e.g. ``["python", "bench.py"]``.
    output_dir : str
        Where to write capture artifacts.
    dispatch : int
        Which dispatch to capture (-1 = first match).
    timeout : int
        Maximum seconds to wait.

    Returns
    -------
    str
        Path to *output_dir* (which now contains ``metadata.json``).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find the Python interpreter and the script in the command
    python_bin = None
    script_idx = None
    for i, tok in enumerate(cmd):
        if python_bin is None:
            if "python" in os.path.basename(tok):
                python_bin = tok
                continue
        else:
            # After we've found the Python binary, explicitly reject
            # unsupported invocation styles like `python -m` and `python -c`.
            if tok in ("-m", "-c"):
                raise ValueError(
                    f"Triton capture does not support Python invocation styles like `{tok}`"
                )
            script_idx = i
            break

    if python_bin is None or script_idx is None:
        raise ValueError(
            f"Triton capture requires a 'python <script>' command, got: {' '.join(cmd)}"
        )

    hook_fd, hook_path = tempfile.mkstemp(suffix=".py", prefix="kerncap_triton_hook_")
    try:
        with os.fdopen(hook_fd, "w") as f:
            f.write(_CAPTURE_HOOK)

        # python [flags] hook.py  original_script.py [script_args ...]
        modified_cmd = (
            [python_bin]
            + cmd[1:script_idx]  # any python flags (-u, etc.)
            + [hook_path]
            + cmd[script_idx:]  # original script + its args
        )

        env = os.environ.copy()
        env["KERNCAP_KERNEL"] = kernel_name
        env["KERNCAP_OUTPUT"] = output_dir
        if dispatch >= 0:
            env["KERNCAP_DISPATCH"] = str(dispatch)

        try:
            proc = subprocess.run(
                modified_cmd,
                env=env,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Application did not complete within {timeout}s")

        meta_file = os.path.join(output_dir, "metadata.json")
        if not os.path.exists(meta_file):
            raise RuntimeError(
                f"Triton capture did not produce metadata.json in "
                f"{output_dir}.\n"
                f"stdout: {proc.stdout[:500]}\n"
                f"stderr: {proc.stderr[:500]}"
            )

        return output_dir
    finally:
        try:
            os.unlink(hook_path)
        except OSError:
            pass
