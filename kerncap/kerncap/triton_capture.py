"""Python-level capture for Triton kernels.

Intercepts Triton kernel calls at the Python level to capture arguments
including tensors, scalars, and constexpr values that are not visible
at the HSA dispatch level.

The capture works by injecting a ``sitecustomize.py`` into ``PYTHONPATH``
that installs a monkey-patch on ``triton.runtime.jit.JITFunction.run``
at Python startup.  Because ``sitecustomize`` is imported by every Python
interpreter (via the ``site`` module), this approach works across process
boundaries — including child processes spawned via ``multiprocessing.spawn``
(e.g. vLLM's EngineCore).

When the target kernel fires, all Python-level arguments (torch tensors,
scalars, None, etc.) are serialised to disk alongside a ``metadata.json``
that the reproducer generator consumes.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import textwrap
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hook code that monkey-patches triton.runtime.jit.JITFunction.run.
#
# This is written to a temp file and exec'd at Python startup via
# sitecustomize.py.  It must be self-contained (no kerncap imports).
# ---------------------------------------------------------------------------
_HOOK_INSTALLER = textwrap.dedent('''\
    """kerncap Triton capture hook — auto-generated, do not edit."""
    import sys
    import os
    import json
    import inspect

    _target_kernel = os.environ.get("KERNCAP_KERNEL", "")
    _output_dir = os.environ.get("KERNCAP_OUTPUT", "/tmp/kerncap_output")
    _dispatch = int(os.environ.get("KERNCAP_DISPATCH", "-1"))
    _call_count = 0
    _captured = False

    if not _target_kernel:
        raise RuntimeError("KERNCAP_KERNEL not set")


    def _install_hook():
        import triton
        import triton.runtime.jit

        _autotuner_config_keys = set()
        _pending_ref = [None]

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
                if _pending_ref[0] is not None:
                    _torch.cuda.synchronize()
                    _save_reference_outputs(_pending_ref[0])
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
                tensor_args = _save_capture(
                    self_jit, args, kwargs, grid, _autotuner_config_keys,
                )
                _pending_ref[0] = tensor_args
                result = _OrigRun(
                    self_jit, *args, grid=grid, warmup=warmup, **kwargs,
                )
                if not _autotuner_config_keys:
                    import torch as _torch
                    _torch.cuda.synchronize()
                    _save_reference_outputs(tensor_args)
                    _pending_ref[0] = None
                return result

            return _OrigRun(self_jit, *args, grid=grid, warmup=warmup, **kwargs)

        triton.runtime.jit.JITFunction.run = _hooked_run


    def _resolve_grid(grid, all_args=None):
        """Normalise a Triton grid to a 3-tuple of ints."""
        if callable(grid):
            if all_args:
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
        """Save a tensor's full underlying storage to preserve stride layout."""
        import torch
        storage = tensor.detach().untyped_storage()
        flat_gpu = torch.empty(storage.nbytes(), dtype=torch.uint8,
                               device=tensor.device)
        flat_gpu.set_(storage)
        flat_cpu = flat_gpu.cpu()
        flat_cpu.numpy().tofile(filepath)

    def _save_tensor(tensor, filepath):
        """Save a tensor's logical values as a contiguous binary file."""
        import torch
        cpu_tensor = tensor.detach().cpu().contiguous()
        if cpu_tensor.dtype == torch.bfloat16:
            cpu_tensor.view(torch.uint16).numpy().tofile(filepath)
        else:
            cpu_tensor.numpy().tofile(filepath)


    def _is_triton_dtype(val):
        """Return True if *val* is a triton.language dtype."""
        try:
            import triton.language as _tl
            return isinstance(val, _tl.dtype)
        except (ImportError, AttributeError):
            return False

    def _triton_dtype_attr(val):
        """Return the tl attribute name for a triton dtype object."""
        import triton.language as _tl
        for attr in dir(_tl):
            if getattr(_tl, attr, None) is val:
                return attr
        return str(val)


    def _save_capture(jit_fn, args, kwargs, grid, autotuner_config_keys):
        """Save pre-kernel tensor inputs and scalar args."""
        import torch
        import numpy as np

        os.makedirs(_output_dir, exist_ok=True)

        param_names = list(inspect.signature(jit_fn.fn).parameters.keys())

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

            elif _is_triton_dtype(val):
                metadata["args"].append({
                    "index": i,
                    "name": name,
                    "is_pointer": False,
                    "is_const": True,
                    "is_autotune_config": is_config,
                    "value": _triton_dtype_attr(val),
                    "type": "triton_dtype",
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


    _install_hook()
''')

# ---------------------------------------------------------------------------
# sitecustomize.py template — loaded at startup by every Python interpreter.
#
# Checks for _KERNCAP_TRITON_HOOK env var; if set, exec's the hook
# installer.  Safe to shadow an existing sitecustomize because:
#   - The env var is only set during a capture session
#   - If the hook fails (e.g. triton not installed), the error is
#     swallowed and the process continues normally
# ---------------------------------------------------------------------------
_SITECUSTOMIZE = textwrap.dedent('''\
    import os as _os
    _hook = _os.environ.get("_KERNCAP_TRITON_HOOK")
    if _hook and _os.path.isfile(_hook):
        try:
            exec(compile(open(_hook).read(), _hook, "exec"))
        except Exception:
            pass
''')


def run_triton_capture(
    kernel_name: str,
    cmd: List[str],
    output_dir: str,
    dispatch: int = -1,
    timeout: int = 300,
) -> str:
    """Capture a Triton kernel's Python-level arguments.

    Injects a ``sitecustomize.py`` via ``PYTHONPATH`` that installs the
    Triton capture hook at interpreter startup.  This works across process
    boundaries (e.g. ``multiprocessing.spawn``) because every Python
    interpreter imports ``sitecustomize`` via the ``site`` module.

    The command is run directly — no wrapping with ``runpy`` — so any
    command that eventually spawns a Python process running Triton kernels
    will be instrumented (``python script.py``, ``python -m module``,
    ``vllm serve ...``, etc.).

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

    # Create temp dir for sitecustomize.py and the hook script
    site_dir = tempfile.mkdtemp(prefix="kerncap_site_")
    hook_fd, hook_path = tempfile.mkstemp(
        suffix=".py", prefix="kerncap_triton_hook_", dir=site_dir,
    )

    try:
        with os.fdopen(hook_fd, "w") as f:
            f.write(_HOOK_INSTALLER)

        with open(os.path.join(site_dir, "sitecustomize.py"), "w") as f:
            f.write(_SITECUSTOMIZE)

        env = os.environ.copy()
        env["KERNCAP_KERNEL"] = kernel_name
        env["KERNCAP_OUTPUT"] = output_dir
        if dispatch >= 0:
            env["KERNCAP_DISPATCH"] = str(dispatch)

        # Point every Python process at our sitecustomize.py
        env["_KERNCAP_TRITON_HOOK"] = hook_path
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{site_dir}:{existing_pp}" if existing_pp else site_dir

        logger.debug(
            "Triton capture: sitecustomize at %s, hook at %s",
            site_dir, hook_path,
        )

        try:
            proc = subprocess.run(
                cmd,
                env=env,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Application did not complete within {timeout}s")

        meta_file = os.path.join(output_dir, "metadata.json")
        if not os.path.exists(meta_file):
            tail = 2000
            stdout_tail = proc.stdout[-tail:] if proc.stdout else ""
            stderr_tail = proc.stderr[-tail:] if proc.stderr else ""
            raise RuntimeError(
                f"Triton capture did not produce metadata.json in "
                f"{output_dir}.\n"
                f"stdout (last {tail} chars): {stdout_tail}\n"
                f"stderr (last {tail} chars): {stderr_tail}"
            )

        return output_dir
    finally:
        shutil.rmtree(site_dir, ignore_errors=True)
