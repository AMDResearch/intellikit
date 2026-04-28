"""Triton kernel capture via the HSA layer (no JITFunction monkey-patch).

Two cooperating pieces:

1. ``libkerncap.so`` (loaded via ``LD_PRELOAD``) intercepts HSA AQL
   kernel-dispatch packets, parses kernarg layout from the loaded
   code object's ``NT_AMDGPU_METADATA`` note, and dumps the kernarg
   buffer + memory snapshot exactly the same way the HIP backend does.

2. A small ``triton.compiler.compile`` wrapper installed via the same
   ``sitecustomize.py`` trampoline as ``triton_capture.py``.  It records
   ``(user_name, hsaco_sha256, hsaco_path, signature, param_names,
   constexpr_values, launch={num_warps,num_stages,...}, source_file,
   source_snapshot)`` to ``name_map.json``.  This is *observation only*
   -- the launch path is never altered.

3. (Optional, advisory) A short-lived ``JITFunction.run`` observer that
   fires exactly once for the first launch matching the target kernel,
   records ``tensor_layout`` (shape/stride/dtype/storage_offset) into the
   matching name_map row, and uninstalls itself.  Used solely to give
   the Python reproducer accurate ``torch.as_strided`` views; replay
   correctness does not depend on it (the HSA-layer kernarg buffer is
   the source of truth).

Compared to ``triton_capture.py``:

* No long-lived ``JITFunction.run`` patch.  The optional observer
  uninstalls after one launch so it cannot perturb other workloads.
* No ``Autotuner.run`` / ``CachingAutotuner.run`` patches at all.
* Multi-process / async-compile workloads are supported by the same
  ``sitecustomize.py`` trampoline that ``triton_capture.py`` uses.
* The on-disk schema is the union of the HIP backend's VA-faithful
  artifacts (``dispatch.json`` + ``kernarg_raw.bin`` +
  ``memory_regions.json``) and a Triton-specific ``name_map.json``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import textwrap
from typing import List

from kerncap._subprocess import run_streaming

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The compile-hook installer.  This is exec'd at Python startup via
# sitecustomize.py (works across multiprocessing/spawn and async-compile
# worker subprocesses).  Must be self-contained -- no kerncap imports.
# ---------------------------------------------------------------------------
_HOOK_INSTALLER = textwrap.dedent('''\
    """kerncap Triton HSA-capture hook -- auto-generated."""
    import contextlib
    import fcntl
    import hashlib
    import inspect
    import json
    import os
    import shutil
    import tempfile
    import threading
    from pathlib import Path

    _name_map_path = os.environ.get("KERNCAP_TRITON_NAME_MAP")
    if not _name_map_path:
        raise RuntimeError("KERNCAP_TRITON_NAME_MAP not set")

    _hsaco_dir = os.environ.get("KERNCAP_TRITON_HSACO_DIR", "")
    _source_dir = os.environ.get("KERNCAP_TRITON_SOURCE_DIR", "")
    _target_kernel = os.environ.get("KERNCAP_KERNEL", "")

    # Cross-process+cross-thread exclusion for the name_map.json
    # read-modify-write cycle.  vLLM (and torch.compile) spawn many
    # Python interpreters via multiprocessing; every one of them runs
    # this hook and may want to append a row.  A per-process
    # ``threading.Lock`` would only serialise threads within a single
    # interpreter -- two *processes* would still race and:
    #   1. corrupt the shared ``name_map.json.tmp`` staging file by
    #      both opening it ``"w"`` (truncate) and interleaving writes,
    #      then publishing the hybrid bytes via ``os.replace``; or
    #   2. lose append-only updates because each process reads the
    #      same N rows, appends one, and writes N+1 rows -- the second
    #      writer wins and the first append is dropped.
    # We fix both by holding an ``fcntl.flock(LOCK_EX)`` on a sibling
    # ``.lock`` file across the entire read-modify-write, and by
    # writing to a per-process unique tmp filename via ``mkstemp``.
    _lock_path = _name_map_path + ".lock"
    _thread_lock = threading.Lock()


    @contextlib.contextmanager
    def _locked_rmw():
        """Hold cross-process+thread exclusion around name_map.json RMW."""
        os.makedirs(os.path.dirname(_lock_path) or ".", exist_ok=True)
        # In-process fast path first so threads in the same interpreter
        # don't fight over the lockfile fd unnecessarily.
        with _thread_lock:
            fd = os.open(_lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


    def _sha256_bytes(b):
        return hashlib.sha256(b).hexdigest()


    def _json_safe(v):
        """Recursively replace IEEE special floats with string sentinels.

        Python's ``json.dump`` defaults to ``allow_nan=True`` and writes
        ``Infinity``/``-Infinity``/``NaN`` as bare tokens.  Those are
        valid in Python's ``json`` but NOT valid per RFC 8259, and
        strict parsers (notably nlohmann::json on the C++ side, which
        ``libkerncap.so`` uses to read this file) reject them with
        ``expected digit after '-'; last read: '-I'`` and similar.

        We round-trip via the lowercase string sentinels ``"inf"``,
        ``"-inf"``, ``"nan"`` -- ``float("-inf")`` accepts these, so a
        consumer that knows a constexpr is float-typed can recover the
        original value.  All other values pass through unchanged.
        """
        import math
        if isinstance(v, float):
            if math.isnan(v):
                return "nan"
            if math.isinf(v):
                return "-inf" if v < 0 else "inf"
            return v
        if isinstance(v, dict):
            return {k: _json_safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_json_safe(x) for x in v]
        return v


    def _atomic_write_rows(rows):
        # Per-process unique tmp filename -- never collides with another
        # writer mid-flight even in the (impossible-by-design but
        # defensive) case the flock guard is somehow violated (e.g.
        # NFS without lockd).  ``os.replace`` then atomically publishes
        # the fully-written file so readers (libkerncap.so on the C++
        # side) never observe a partial JSON document.
        #
        # ``allow_nan=False`` makes ``json.dump`` raise instead of
        # silently writing ``Infinity``/``NaN`` tokens that the strict
        # C++ parser rejects.  ``_json_safe`` pre-walks the rows to
        # rewrite IEEE specials as string sentinels so the safety net
        # only fires on something we genuinely missed.
        target_dir = os.path.dirname(_name_map_path) or "."
        prefix = os.path.basename(_name_map_path) + "."
        fd, tmp = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=target_dir)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(
                    _json_safe(rows),
                    f,
                    indent=2,
                    default=str,
                    allow_nan=False,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, _name_map_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


    def _load_rows():
        if not os.path.exists(_name_map_path):
            return []
        try:
            with open(_name_map_path) as f:
                return json.load(f)
        except Exception:
            return []


    def _append_row(row):
        with _locked_rmw():
            rows = _load_rows()
            rows.append(row)
            _atomic_write_rows(rows)


    def _safe_str(x):
        try:
            return str(x)
        except Exception:
            return ""


    def _safe_serializable(v):
        # Triton dtype objects (e.g. ``tl.bfloat16``) carry semantic
        # info the kernel needs at compile time -- ``tl.zeros(dtype=...)``
        # rejects a bare string.  ``str(tl.bfloat16)`` returns ``"bf16"``
        # which is lossy *and* not a valid Triton dtype string, so we
        # tag with a sentinel dict the reproducer renderer recognises
        # and rewrites back to ``tl.<name>`` at codegen time.
        try:
            import triton.language as _tl
            _dtype_t = getattr(_tl, "dtype", None)
            if _dtype_t is not None and isinstance(v, _dtype_t):
                return {"__triton_dtype__": getattr(v, "name", _safe_str(v))}
        except Exception:
            pass
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return _safe_str(v)


    # -----------------------------------------------------------------
    # compile() shim -- captures everything visible at compile time.
    # -----------------------------------------------------------------

    def _harvest(ck):
        """Record one row for the just-completed triton.compiler.compile()."""
        # 1. user_name
        name = None
        for chain in (("metadata", "name"), ("name",), ("src", "name")):
            obj = ck
            ok = True
            for a in chain:
                obj = getattr(obj, a, None)
                if obj is None:
                    ok = False
                    break
            if ok and isinstance(obj, str):
                name = obj
                break
        if name is None:
            return

        # 2. HSACO bytes (from CompiledKernel; fall back to cache scan)
        hsaco_bytes = None
        try:
            hsaco_bytes = ck.asm["hsaco"]
        except Exception:
            pass
        if hsaco_bytes is None:
            triton_home = Path(os.environ.get("TRITON_HOME") or Path.home())
            cache = triton_home / ".triton" / "cache"
            cands = sorted(
                cache.glob(f"*/{name}.hsaco"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if cands:
                hsaco_bytes = cands[0].read_bytes()
        if not isinstance(hsaco_bytes, (bytes, bytearray)):
            return
        sha = _sha256_bytes(bytes(hsaco_bytes))

        # 3. Persist HSACO copy (content-addressed filename)
        hsaco_path = ""
        if _hsaco_dir:
            dst = Path(_hsaco_dir) / f"{name}.{sha[:16]}.hsaco"
            try:
                if not dst.exists():
                    dst.write_bytes(bytes(hsaco_bytes))
                hsaco_path = str(dst)
            except OSError:
                hsaco_path = ""

        # 4. Locate the JITFunction (for param names + source file)
        src = getattr(ck, "src", None)
        jit_fn = None
        for obj in (src, ck):
            cand = getattr(obj, "fn", None)
            if cand is not None and hasattr(cand, "fn") and hasattr(cand, "params"):
                jit_fn = cand
                break

        # 5. Param names in declaration order
        param_names = []
        if jit_fn is not None:
            try:
                param_names = [getattr(p, "name", None) for p in jit_fn.params]
                param_names = [n for n in param_names if isinstance(n, str)]
            except Exception:
                param_names = []
        if not param_names and jit_fn is not None and hasattr(jit_fn, "fn"):
            try:
                param_names = list(inspect.signature(jit_fn.fn).parameters.keys())
            except Exception:
                param_names = []

        # 6. Source file (for the editable Python reproducer)
        source_file = ""
        if jit_fn is not None and hasattr(jit_fn, "fn"):
            try:
                source_file = inspect.getsourcefile(jit_fn.fn) or ""
            except Exception:
                source_file = ""

        # 7. Triton signature dict {param_name: type_str}
        signature = {}
        raw_sig = getattr(src, "signature", None)
        if isinstance(raw_sig, dict):
            for k, v in raw_sig.items():
                if isinstance(k, int) and 0 <= k < len(param_names):
                    key = param_names[k]
                else:
                    key = _safe_str(k)
                signature[key] = _safe_str(v)

        # 8. Constexpr values keyed by param name when possible
        constexprs_by_name = {}
        raw_constexprs = (
            getattr(src, "constexprs", None) or getattr(src, "constants", None)
        )
        if isinstance(raw_constexprs, dict):
            for k, v in raw_constexprs.items():
                if isinstance(k, tuple) and len(k) == 1 and isinstance(k[0], int):
                    idx = k[0]
                elif isinstance(k, int):
                    idx = k
                else:
                    idx = -1
                if 0 <= idx < len(param_names):
                    key = param_names[idx]
                else:
                    key = _safe_str(k)
                constexprs_by_name[key] = _safe_serializable(v)

        # 9. Launch attributes from CompiledKernel.metadata
        launch = {}
        meta = getattr(ck, "metadata", None)
        for attr in ("num_warps", "num_stages", "num_ctas", "waves_per_eu"):
            val = getattr(meta, attr, None)
            if val is not None:
                try:
                    launch[attr] = int(val)
                except Exception:
                    pass

        # 10. Snapshot the source file (insurance for torch.compile /
        # temp dirs).  If the @triton.jit function lives in a Python
        # *package* (its directory has __init__.py), copy the whole
        # package directory -- otherwise relative imports like
        # ``from .utils import ...`` in the snapshotted file will fail
        # when the reproducer re-imports it standalone.
        snapshot_path = ""
        if source_file and _source_dir and os.path.isfile(source_file):
            try:
                src_dir = os.path.dirname(source_file)
                base = os.path.basename(source_file)
                if os.path.isfile(os.path.join(src_dir, "__init__.py")):
                    for fname in os.listdir(src_dir):
                        if not fname.endswith(".py"):
                            continue
                        s = os.path.join(src_dir, fname)
                        d = os.path.join(_source_dir, fname)
                        if os.path.isfile(s) and not os.path.exists(d):
                            shutil.copy2(s, d)
                else:
                    dst = os.path.join(_source_dir, base)
                    if not os.path.exists(dst):
                        shutil.copy2(source_file, dst)
                snapshot_path = os.path.join(_source_dir, base)
            except OSError:
                snapshot_path = ""

        _append_row({
            "user_name": name,
            "hsaco_sha256": sha,
            "hsaco_path": hsaco_path,
            "constexpr_values": constexprs_by_name,
            "signature": signature,
            "param_names": param_names,
            "launch": launch,
            "source_file": source_file,
            "source_snapshot": snapshot_path,
        })


    def _install_compile_hook():
        try:
            import triton.compiler as tc
        except Exception:
            return
        if not hasattr(tc, "compile"):
            return
        if getattr(tc.compile, "_kerncap_compile_hook", False):
            return
        orig = tc.compile

        def wrapped(*args, **kwargs):
            ck = orig(*args, **kwargs)
            try:
                _harvest(ck)
            except Exception:
                pass
            return ck

        wrapped._kerncap_compile_hook = True
        tc.compile = wrapped


    # -----------------------------------------------------------------
    # Short-lived JITFunction.run observer (advisory tensor_layout).
    #
    # Fires at most once for the first launch matching the target
    # kernel name, records {arg_index, name, shape, stride, dtype,
    # storage_offset} for each torch.Tensor arg, attaches it to the
    # matching name_map row, then uninstalls itself.  Purely advisory:
    # the HSA-layer kernarg buffer remains the source of truth.
    # -----------------------------------------------------------------
    _observer_fired = [False]


    def _compute_layout(jit_fn, args, kwargs):
        try:
            import torch
        except Exception:
            return []
        try:
            param_names = [getattr(p, "name", None) for p in jit_fn.params]
            param_names = [n for n in param_names if isinstance(n, str)]
        except Exception:
            try:
                param_names = list(inspect.signature(jit_fn.fn).parameters.keys())
            except Exception:
                param_names = []

        layouts = []
        for i, name in enumerate(param_names):
            if i < len(args):
                val = args[i]
            elif name in kwargs:
                val = kwargs[name]
            else:
                continue
            if isinstance(val, torch.Tensor):
                try:
                    nbytes_storage = int(val.untyped_storage().nbytes())
                except Exception:
                    nbytes_storage = int(val.element_size() * val.numel())
                layouts.append({
                    "index": i,
                    "name": name,
                    "shape": [int(d) for d in val.shape],
                    "stride": [int(s) for s in val.stride()],
                    "storage_offset": int(val.storage_offset()),
                    "dtype": str(val.dtype),
                    "nbytes_storage": nbytes_storage,
                })
        return layouts


    def _attach_layout_to_latest_row(user_name, layouts):
        with _locked_rmw():
            rows = _load_rows()
            target = None
            for r in rows:
                if r.get("user_name") == user_name:
                    target = r
            if target is None:
                # Compile hook didn't fire (cache hit).  Synthesise a
                # minimal row so the reproducer generator has at least
                # the layout to work with.
                target = {
                    "user_name": user_name,
                    "tensor_layout": layouts,
                    "synthesized_from_observer": True,
                }
                rows.append(target)
            else:
                target["tensor_layout"] = layouts
            _atomic_write_rows(rows)


    def _install_run_observer():
        if not _target_kernel:
            return
        try:
            import triton.runtime.jit as _jit
        except Exception:
            return
        if not hasattr(_jit, "JITFunction"):
            return
        orig = _jit.JITFunction.run
        if getattr(orig, "_kerncap_observer_installed", False):
            return

        def _observed_run(self_jit, *args, **kwargs):
            captured_name = None
            layout = None
            try:
                if not _observer_fired[0]:
                    fn_name = getattr(
                        getattr(self_jit, "fn", None), "__name__", ""
                    )
                    if fn_name and (
                        _target_kernel in fn_name or fn_name in _target_kernel
                    ):
                        layout = _compute_layout(self_jit, args, kwargs)
                        captured_name = fn_name
                        _observer_fired[0] = True
                        _jit.JITFunction.run = orig
            except Exception:
                pass

            result = orig(self_jit, *args, **kwargs)

            if layout:
                try:
                    _attach_layout_to_latest_row(captured_name, layout)
                except Exception:
                    pass

            return result

        _observed_run._kerncap_observer_installed = True
        _jit.JITFunction.run = _observed_run


    _install_compile_hook()
    _install_run_observer()
''')


# ---------------------------------------------------------------------------
# sitecustomize.py template -- imported by every Python interpreter.
# Same trampoline pattern as triton_capture.py so this works through
# torch.compile's parallel-compile worker subprocesses.
# ---------------------------------------------------------------------------
_SITECUSTOMIZE = textwrap.dedent("""\
    import os as _os
    _hook = _os.environ.get("_KERNCAP_TRITON_HSA_HOOK")
    if _hook and _os.path.isfile(_hook):
        try:
            exec(compile(open(_hook).read(), _hook, "exec"))
        except Exception:
            pass
""")


def _maybe_clear_triton_caches() -> None:
    """Best-effort: clear Triton's on-disk JIT cache for the captured run.

    Inductor / Triton may skip ``triton.compiler.compile`` on a cache
    hit, in which case our compile shim never fires and ``name_map.json``
    is missing the row needed to map the dispatched HSACO back to a
    user-friendly name.  Clearing the cache forces a fresh compile.

    Honors ``KERNCAP_NO_CLEAR_TRITON_CACHE=1`` as an opt-out (e.g. for
    debugging).  The Inductor cache is *not* cleared here -- that is
    workload-specific and the user can purge ``/tmp/torchinductor_*``
    themselves if the hook is being bypassed by Inductor's own cache.
    """
    if os.environ.get("KERNCAP_NO_CLEAR_TRITON_CACHE", "") == "1":
        return
    triton_home = os.environ.get("TRITON_HOME") or os.path.expanduser("~")
    cache = os.path.join(triton_home, ".triton", "cache")
    if os.path.isdir(cache):
        try:
            shutil.rmtree(cache)
            logger.debug("Cleared Triton JIT cache at %s", cache)
        except OSError as e:
            logger.debug("Failed to clear Triton cache at %s: %s", cache, e)


def run_triton_capture_hsa(
    kernel_name: str,
    cmd: List[str],
    output_dir: str,
    dispatch: int = -1,
    timeout: int = 300,
) -> str:
    """Capture a Triton kernel via the HSA layer (no Python launch hook).

    Mirrors the signature of :func:`kerncap.triton_capture.run_triton_capture`
    so the ``--triton-backend`` switch in the CLI is a one-line dispatch.

    Side effects: launches *cmd* with ``LD_PRELOAD=libkerncap.so`` plus
    a ``sitecustomize.py`` hook that installs the compile shim and the
    short-lived run observer in every spawned interpreter.  The shim
    writes ``name_map.json`` rows and copies HSACOs into
    ``output_dir/triton_hsacos/`` and source files into
    ``output_dir/triton_sources/``; ``libkerncap.so`` consumes
    ``name_map.json`` at dispatch time to resolve the kernel
    content-hash to a user name.

    Parameters
    ----------
    kernel_name, cmd, output_dir, dispatch, timeout
        See :func:`kerncap.triton_capture.run_triton_capture`.

    Returns
    -------
    str
        ``output_dir`` (which now contains ``metadata.json``,
        ``kernarg_raw.bin``, ``name_map.json``, etc.).
    """
    from kerncap import _get_lib_path

    os.makedirs(output_dir, exist_ok=True)
    lib_path = _get_lib_path()

    _maybe_clear_triton_caches()

    site_dir = tempfile.mkdtemp(prefix="kerncap_site_hsa_")
    hook_fd, hook_path = tempfile.mkstemp(
        suffix=".py",
        prefix="kerncap_triton_hsa_hook_",
        dir=site_dir,
    )
    name_map_path = os.path.join(output_dir, "name_map.json")
    hsaco_dir = os.path.join(output_dir, "triton_hsacos")
    source_dir = os.path.join(output_dir, "triton_sources")

    try:
        with os.fdopen(hook_fd, "w") as f:
            f.write(_HOOK_INSTALLER)
        with open(os.path.join(site_dir, "sitecustomize.py"), "w") as f:
            f.write(_SITECUSTOMIZE)

        env = os.environ.copy()
        env.pop("HSA_TOOLS_LIB", None)
        env.pop("HSA_TOOLS_REPORT_LOAD_FAILURE", None)
        if "LD_PRELOAD" in env:
            env["LD_PRELOAD"] = lib_path + ":" + env["LD_PRELOAD"]
        else:
            env["LD_PRELOAD"] = lib_path

        env["KERNCAP_KERNEL"] = kernel_name
        env["KERNCAP_OUTPUT"] = output_dir
        env["KERNCAP_CAPTURE_CHILD"] = "1"
        env["KERNCAP_TRITON_NAME_MAP"] = name_map_path
        env["KERNCAP_TRITON_HSACO_DIR"] = hsaco_dir
        env["KERNCAP_TRITON_SOURCE_DIR"] = source_dir
        if dispatch >= 0:
            env["KERNCAP_DISPATCH"] = str(dispatch)

        env["_KERNCAP_TRITON_HSA_HOOK"] = hook_path
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{site_dir}:{existing_pp}" if existing_pp else site_dir

        logger.debug(
            "Triton HSA capture: lib=%s site=%s hook=%s name_map=%s",
            lib_path,
            site_dir,
            hook_path,
            name_map_path,
        )

        sentinel = os.path.join(output_dir, "capture_complete")
        try:
            proc = run_streaming(
                cmd,
                env=env,
                timeout=timeout,
                completion_sentinel=sentinel,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"Application did not complete within {timeout}s") from e

        dispatch_file = os.path.join(output_dir, "dispatch.json")
        meta_file = os.path.join(output_dir, "metadata.json")
        if not os.path.exists(dispatch_file) and not os.path.exists(meta_file):
            tail = 2000
            stdout_tail = proc.stdout[-tail:] if proc.stdout else ""
            stderr_tail = proc.stderr[-tail:] if proc.stderr else ""
            raise RuntimeError(
                f"Triton HSA capture did not produce metadata.json/dispatch.json "
                f"in {output_dir}.\n"
                f"stdout (last {tail} chars): {stdout_tail}\n"
                f"stderr (last {tail} chars): {stderr_tail}"
            )

        return output_dir
    finally:
        shutil.rmtree(site_dir, ignore_errors=True)
