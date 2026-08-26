"""Run a child process while streaming output and capturing a bounded tail.

Replaces ``subprocess.run(cmd, capture_output=True, text=True, ...)`` for
the long-running, chatty children kerncap launches (vLLM, llama.cpp,
``rocprofv3 -- <app>``, ...).

Two problems with ``capture_output=True`` for those workloads:

1. The user sees nothing while the child runs — model loads, autotune
   passes, and OOM tracebacks are all hidden until the child exits.  If
   the child hangs or is killed, the parent looks frozen forever and
   critical diagnostics never reach the terminal.
2. All of stdout/stderr is buffered into the parent's RSS.  A multi-GB
   chatty child easily makes the orchestrator itself OOM.

``run_streaming`` solves both by:

* tee-ing stdout/stderr to the parent's terminal in real time, and
* keeping only the last ``tail_bytes`` of each stream (default 64 KiB)
  in a bounded ring buffer so the existing
  ``"...stderr (last N chars): ..."`` error-reporting paths still work
  without unbounded memory growth.

The return value is shape-compatible with ``subprocess.CompletedProcess``
(``.returncode``, ``.stdout``, ``.stderr``, ``.args``), so call sites
that previously did ``proc = subprocess.run(...)`` can continue to read
``proc.stdout`` / ``proc.stderr`` for diagnostic tails.
"""

from __future__ import annotations

import collections
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Mapping, Optional, Sequence


_DEFAULT_TAIL_BYTES = 64 * 1024
_SENTINEL_POLL_INTERVAL_S = 1.0
_SENTINEL_TERM_GRACE_S = 5.0


def _resolve_sink(sink, fallback) -> object:
    """Return a binary-write sink (prefers ``.buffer`` on text streams)."""
    if sink is not None:
        return sink
    return getattr(fallback, "buffer", fallback)


def _pump(src, sink, tail: collections.deque) -> None:
    """Copy bytes from *src* to *sink* and append into *tail* (bounded)."""
    try:
        for chunk in iter(lambda: src.read(4096), b""):
            if not chunk:
                break
            try:
                sink.write(chunk)
                sink.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass
            tail.extend(chunk)
    finally:
        try:
            src.close()
        except OSError:
            pass


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Send SIGTERM to the child's process group, escalate to SIGKILL.

    The whole point is that vLLM (and friends) spawn N worker
    subprocesses; signalling only the leader leaves the workers as
    orphaned zombies and the leader stays blocked on their IPC.  We
    rely on ``start_new_session=True`` at Popen time so the child is
    its own session leader and ``os.killpg(pgid, ...)`` reaches every
    descendant.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None

    def _signal(sig: int) -> None:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass

    _signal(signal.SIGTERM)
    try:
        proc.wait(timeout=_SENTINEL_TERM_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal(signal.SIGKILL)
    try:
        proc.wait(timeout=_SENTINEL_TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        pass


def _watch_sentinel(
    proc: subprocess.Popen,
    sentinel: str,
    stop_event: threading.Event,
) -> None:
    """Poll for *sentinel* and terminate *proc* the moment it appears."""
    while not stop_event.is_set():
        if os.path.exists(sentinel):
            _terminate_process_group(proc)
            return
        if proc.poll() is not None:
            return
        stop_event.wait(_SENTINEL_POLL_INTERVAL_S)


def run_streaming(
    cmd: Sequence[str],
    *,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
    tail_bytes: int = _DEFAULT_TAIL_BYTES,
    stdout_sink=None,
    stderr_sink=None,
    completion_sentinel: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run *cmd*, stream its output live, return a ``CompletedProcess``.

    Drop-in replacement for
    ``subprocess.run(cmd, env=env, timeout=timeout, capture_output=True,
    text=True)`` — but the child's stdout/stderr are also written to the
    parent's terminal in real time, and only the last ``tail_bytes`` of
    each stream are retained for the returned ``CompletedProcess``.

    Override ``tail_bytes`` per-call, or globally via the
    ``KERNCAP_TAIL_BYTES`` environment variable.

    Parameters
    ----------
    cmd
        Command to execute.
    env
        Environment for the child (defaults to inheriting the parent's).
    timeout
        Maximum seconds to wait.  Raises ``subprocess.TimeoutExpired``
        on expiry, with partial tails attached as ``output``/``stderr``.
    tail_bytes
        Bytes of stdout/stderr to retain for diagnostics.
    stdout_sink, stderr_sink
        Override the live-stream destinations (binary writers).  Default
        is ``sys.stdout.buffer`` / ``sys.stderr.buffer``.  Useful for
        tests.
    completion_sentinel
        If set, a watchdog thread polls for this filesystem path and
        SIGTERM-then-SIGKILLs the child's whole process group the
        instant it appears.  Used by the kerncap capture pipeline so
        long-lived hosts (vLLM, llama.cpp, ...) don't have to
        gracefully complete a benchmark run after the artifacts we
        actually care about are already on disk.  ``returncode`` from
        a sentinel-driven termination is *not* treated as an error by
        callers in this package.

    Returns
    -------
    subprocess.CompletedProcess
        ``returncode`` and decoded ``stdout``/``stderr`` tails.
    """
    env_tail = os.environ.get("KERNCAP_TAIL_BYTES")
    if env_tail:
        try:
            tail_bytes = max(0, int(env_tail))
        except ValueError:
            pass

    out_sink = _resolve_sink(stdout_sink, sys.stdout)
    err_sink = _resolve_sink(stderr_sink, sys.stderr)

    out_tail: collections.deque = collections.deque(maxlen=tail_bytes)
    err_tail: collections.deque = collections.deque(maxlen=tail_bytes)

    # ``start_new_session=True`` puts the child (and everything it
    # spawns) in a fresh process group so the sentinel watchdog can
    # signal the whole tree, not just the leader.  Harmless when
    # ``completion_sentinel`` is None.
    proc = subprocess.Popen(
        list(cmd),
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
    )

    t_out = threading.Thread(target=_pump, args=(proc.stdout, out_sink, out_tail), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, err_sink, err_tail), daemon=True)
    t_out.start()
    t_err.start()

    sentinel_stop: Optional[threading.Event] = None
    t_sentinel: Optional[threading.Thread] = None
    if completion_sentinel:
        sentinel_stop = threading.Event()
        t_sentinel = threading.Thread(
            target=_watch_sentinel,
            args=(proc, completion_sentinel, sentinel_stop),
            daemon=True,
        )
        t_sentinel.start()

    try:
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc)
            t_out.join(timeout=2)
            t_err.join(timeout=2)
            raise subprocess.TimeoutExpired(
                cmd,
                timeout,
                output=bytes(out_tail).decode("utf-8", errors="replace"),
                stderr=bytes(err_tail).decode("utf-8", errors="replace"),
            )
    finally:
        if sentinel_stop is not None:
            sentinel_stop.set()
        if t_sentinel is not None:
            t_sentinel.join(timeout=2)

    t_out.join()
    t_err.join()

    # If the watchdog killed the child after the sentinel appeared,
    # ``returncode`` will be a negative signal value.  That's expected
    # for the kerncap capture pipeline; callers that pass
    # ``completion_sentinel`` interpret artifact existence as success.
    return subprocess.CompletedProcess(
        args=list(cmd),
        returncode=rc,
        stdout=bytes(out_tail).decode("utf-8", errors="replace"),
        stderr=bytes(err_tail).decode("utf-8", errors="replace"),
    )
