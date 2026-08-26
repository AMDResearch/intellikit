"""Unit tests for kerncap._subprocess — streaming child output with a bounded tail.

These drive **real** child processes (``/bin/sh``) rather than mocks. What is
under test here is process and thread behaviour — bounded ring buffers filled
from pump threads, process-group signalling, a polling watchdog — and a mocked
``Popen`` would mostly assert the behaviour of the mock.

The children are all sub-second. The two that exercise the watchdog's polling
loop cost about a second each, which is the price of testing the real thing.
"""

import collections
import io
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from kerncap._subprocess import (
    _DEFAULT_TAIL_BYTES,
    _pump,
    _resolve_sink,
    _terminate_process_group,
    _watch_sentinel,
    run_streaming,
)


def sh(script: str) -> list:
    """A shell command list, so tests read as the shell they are."""
    return ["/bin/sh", "-c", script]


@pytest.fixture
def sinks():
    """Binary stdout/stderr sinks standing in for the parent's terminal."""
    return io.BytesIO(), io.BytesIO()


# --------------------------------------------------------------------------
# _resolve_sink
# --------------------------------------------------------------------------


class TestResolveSink:
    def test_explicit_sink_wins(self):
        explicit = io.BytesIO()
        assert _resolve_sink(explicit, sys.stdout) is explicit

    def test_prefers_the_binary_buffer_of_a_text_stream(self):
        class TextStream:
            buffer = io.BytesIO()

        stream = TextStream()
        assert _resolve_sink(None, stream) is stream.buffer

    def test_falls_back_to_the_stream_itself(self):
        """Under pytest capture the replacement stdout may have no .buffer."""

        class NoBuffer:
            pass

        stream = NoBuffer()
        assert _resolve_sink(None, stream) is stream


# --------------------------------------------------------------------------
# _pump
# --------------------------------------------------------------------------


class TestPump:
    def test_copies_to_sink_and_tail(self):
        src = io.BytesIO(b"hello world")
        sink = io.BytesIO()
        tail: collections.deque = collections.deque(maxlen=1024)

        _pump(src, sink, tail)

        assert sink.getvalue() == b"hello world"
        assert bytes(tail) == b"hello world"

    def test_tail_is_bounded_but_sink_is_not(self):
        """The ring buffer drops old bytes; the live stream still sees them all."""
        src = io.BytesIO(b"0123456789")
        sink = io.BytesIO()
        tail: collections.deque = collections.deque(maxlen=4)

        _pump(src, sink, tail)

        assert sink.getvalue() == b"0123456789"
        assert bytes(tail) == b"6789"

    def test_source_is_closed(self):
        src = io.BytesIO(b"x")
        _pump(src, io.BytesIO(), collections.deque(maxlen=8))
        assert src.closed

    def test_a_broken_sink_does_not_lose_the_tail(self):
        """If the terminal goes away the diagnostics must still be captured."""

        class BrokenSink:
            def write(self, _chunk):
                raise BrokenPipeError("gone")

            def flush(self):
                pass

        src = io.BytesIO(b"important diagnostics")
        tail: collections.deque = collections.deque(maxlen=1024)

        _pump(src, BrokenSink(), tail)

        assert bytes(tail) == b"important diagnostics"

    def test_a_closed_sink_does_not_raise(self):
        sink = io.BytesIO()
        sink.close()
        tail: collections.deque = collections.deque(maxlen=64)

        _pump(io.BytesIO(b"data"), sink, tail)

        assert bytes(tail) == b"data"


# --------------------------------------------------------------------------
# run_streaming — basics
# --------------------------------------------------------------------------


class TestRunStreamingBasics:
    def test_captures_stdout_stderr_and_returncode(self, sinks):
        out, err = sinks
        proc = run_streaming(
            sh("printf out; printf err >&2; exit 3"), stdout_sink=out, stderr_sink=err
        )

        assert proc.returncode == 3
        assert proc.stdout == "out"
        assert proc.stderr == "err"

    def test_is_shape_compatible_with_completedprocess(self, sinks):
        out, err = sinks
        cmd = sh("true")
        proc = run_streaming(cmd, stdout_sink=out, stderr_sink=err)

        assert isinstance(proc, subprocess.CompletedProcess)
        assert proc.args == cmd

    def test_streams_live_to_the_sinks(self, sinks):
        """The whole point: the user sees output as it happens, not at exit."""
        out, err = sinks
        run_streaming(sh("printf hello; printf oops >&2"), stdout_sink=out, stderr_sink=err)

        assert out.getvalue() == b"hello"
        assert err.getvalue() == b"oops"

    def test_undecodable_bytes_do_not_raise(self, sinks):
        """A chatty child may emit binary junk; it must not kill the run.

        Octal escapes, not ``\\x``: POSIX ``printf`` in dash does not accept
        hex escapes and would emit them literally.
        """
        out, err = sinks
        proc = run_streaming(sh(r"printf '\377\376'"), stdout_sink=out, stderr_sink=err)

        assert proc.returncode == 0
        assert out.getvalue() == b"\xff\xfe"
        # errors="replace" — the tail is decoded lossily rather than raising.
        assert proc.stdout == "��"

    def test_env_replaces_the_inherited_environment(self, sinks):
        out, err = sinks
        proc = run_streaming(
            sh('printf %s "$KERNCAP_TEST_VAR"'),
            env={"KERNCAP_TEST_VAR": "injected"},
            stdout_sink=out,
            stderr_sink=err,
        )

        assert proc.stdout == "injected"

    def test_without_env_the_child_inherits(self, sinks, monkeypatch):
        monkeypatch.setenv("KERNCAP_INHERITED", "yes")
        out, err = sinks
        proc = run_streaming(sh('printf %s "$KERNCAP_INHERITED"'), stdout_sink=out, stderr_sink=err)

        assert proc.stdout == "yes"

    def test_child_runs_in_its_own_session(self, sinks):
        """start_new_session is what makes process-group signalling reach workers."""
        out, err = sinks
        proc = run_streaming(sh("ps -o sid= -p $$"), stdout_sink=out, stderr_sink=err)

        assert proc.stdout.strip()
        assert int(proc.stdout.strip()) != os.getsid(0)


# --------------------------------------------------------------------------
# run_streaming — tail bounding
# --------------------------------------------------------------------------


class TestTailBounding:
    def test_tail_bytes_truncates_to_the_last_n(self, sinks):
        out, err = sinks
        proc = run_streaming(
            sh("printf 0123456789ABCDEFGHIJ"), tail_bytes=5, stdout_sink=out, stderr_sink=err
        )

        assert proc.stdout == "FGHIJ"
        # The live stream is unbounded — only the retained tail is capped.
        assert out.getvalue() == b"0123456789ABCDEFGHIJ"

    def test_stdout_and_stderr_are_bounded_independently(self, sinks):
        out, err = sinks
        proc = run_streaming(
            sh("printf aaaaaa; printf bbbbbb >&2"),
            tail_bytes=2,
            stdout_sink=out,
            stderr_sink=err,
        )

        assert proc.stdout == "aa"
        assert proc.stderr == "bb"

    def test_env_var_overrides_the_argument(self, sinks, monkeypatch):
        monkeypatch.setenv("KERNCAP_TAIL_BYTES", "3")
        out, err = sinks
        proc = run_streaming(
            sh("printf 0123456789"), tail_bytes=1024, stdout_sink=out, stderr_sink=err
        )

        assert proc.stdout == "789"

    def test_unparseable_env_var_is_ignored(self, sinks, monkeypatch):
        """A typo in the env var must not crash a long profiling run."""
        monkeypatch.setenv("KERNCAP_TAIL_BYTES", "not-a-number")
        out, err = sinks
        proc = run_streaming(
            sh("printf 0123456789"), tail_bytes=4, stdout_sink=out, stderr_sink=err
        )

        assert proc.stdout == "6789"

    def test_empty_env_var_is_ignored(self, sinks, monkeypatch):
        monkeypatch.setenv("KERNCAP_TAIL_BYTES", "")
        out, err = sinks
        proc = run_streaming(
            sh("printf 0123456789"), tail_bytes=4, stdout_sink=out, stderr_sink=err
        )

        assert proc.stdout == "6789"

    def test_negative_env_var_clamps_to_zero(self, sinks, monkeypatch):
        """max(0, ...) — a negative maxlen would be a ValueError from deque."""
        monkeypatch.setenv("KERNCAP_TAIL_BYTES", "-5")
        out, err = sinks
        proc = run_streaming(sh("printf hello"), stdout_sink=out, stderr_sink=err)

        assert proc.stdout == ""
        assert out.getvalue() == b"hello"

    def test_default_tail_is_64_kib(self):
        assert _DEFAULT_TAIL_BYTES == 64 * 1024


# --------------------------------------------------------------------------
# run_streaming — timeout
# --------------------------------------------------------------------------


class TestTimeout:
    def test_timeout_raises_with_partial_output_attached(self, sinks):
        """The diagnostics gathered before the timeout must survive it."""
        out, err = sinks
        with pytest.raises(subprocess.TimeoutExpired) as exc:
            run_streaming(
                sh("printf partial; printf trouble >&2; sleep 30"),
                timeout=1.5,
                stdout_sink=out,
                stderr_sink=err,
            )

        assert "partial" in (exc.value.output or "")
        assert "trouble" in (exc.value.stderr or "")

    def test_timeout_kills_the_child(self, sinks):
        out, err = sinks
        start = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            run_streaming(sh("sleep 30"), timeout=1.0, stdout_sink=out, stderr_sink=err)

        # Must not have waited for the 30s child.
        assert time.monotonic() - start < 15

    def test_no_timeout_waits_for_completion(self, sinks):
        out, err = sinks
        proc = run_streaming(sh("sleep 0.2; printf done"), stdout_sink=out, stderr_sink=err)
        assert proc.stdout == "done"


# --------------------------------------------------------------------------
# _terminate_process_group
# --------------------------------------------------------------------------


class TestTerminateProcessGroup:
    def test_kills_the_whole_group(self):
        """A child that spawns a grandchild: signalling the leader is not enough."""
        proc = subprocess.Popen(
            sh("sleep 30 & sleep 30"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        pgid = os.getpgid(proc.pid)

        _terminate_process_group(proc)

        assert proc.poll() is not None
        # The group is gone; signalling it again must fail.
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)
        proc.stdout.close()
        proc.stderr.close()

    def test_escalates_to_sigkill_when_sigterm_is_ignored(self):
        """A child trapping SIGTERM must still be killed."""
        proc = subprocess.Popen(
            sh("trap '' TERM; sleep 30"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        time.sleep(0.3)  # let the trap install

        _terminate_process_group(proc)

        assert proc.poll() is not None
        assert proc.returncode == -signal.SIGKILL
        proc.stdout.close()
        proc.stderr.close()

    def test_falls_back_to_signalling_the_process_directly(self, monkeypatch):
        """If the pgid cannot be read, the leader is still signalled."""
        proc = subprocess.Popen(
            sh("sleep 30"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        monkeypatch.setattr(os, "getpgid", lambda _pid: (_ for _ in ()).throw(ProcessLookupError()))

        _terminate_process_group(proc)

        assert proc.poll() is not None
        proc.stdout.close()
        proc.stderr.close()

    def test_already_dead_process_is_a_noop(self):
        proc = subprocess.Popen(
            sh("true"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        )
        proc.wait()

        _terminate_process_group(proc)  # must not raise

        proc.stdout.close()
        proc.stderr.close()


# --------------------------------------------------------------------------
# _watch_sentinel / completion_sentinel
# --------------------------------------------------------------------------


class TestWatchSentinel:
    def test_returns_when_the_process_exits_on_its_own(self, tmp_path):
        proc = subprocess.Popen(
            sh("true"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        )
        proc.wait()
        stop = threading.Event()

        _watch_sentinel(proc, str(tmp_path / "never"), stop)  # must return, not hang

        proc.stdout.close()
        proc.stderr.close()

    def test_stop_event_ends_the_watch(self, tmp_path):
        proc = subprocess.Popen(
            sh("sleep 30"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stop = threading.Event()
        stop.set()

        _watch_sentinel(proc, str(tmp_path / "never"), stop)

        assert proc.poll() is None  # untouched
        _terminate_process_group(proc)
        proc.stdout.close()
        proc.stderr.close()


class TestCompletionSentinel:
    def test_preexisting_sentinel_terminates_immediately(self, tmp_path, sinks):
        """The artifacts are already on disk, so the host need not finish."""
        sentinel = tmp_path / "done.marker"
        sentinel.write_text("")
        out, err = sinks

        start = time.monotonic()
        proc = run_streaming(
            sh("sleep 30"),
            completion_sentinel=str(sentinel),
            stdout_sink=out,
            stderr_sink=err,
        )

        assert time.monotonic() - start < 15
        assert proc.returncode < 0  # killed by signal, not a clean exit

    def test_sentinel_written_by_the_child_terminates_the_run(self, tmp_path, sinks):
        """The real shape: capture completes mid-run and drops the marker."""
        sentinel = tmp_path / "done.marker"
        out, err = sinks

        start = time.monotonic()
        proc = run_streaming(
            sh(f"printf captured; touch {sentinel}; sleep 30"),
            completion_sentinel=str(sentinel),
            stdout_sink=out,
            stderr_sink=err,
        )

        assert time.monotonic() - start < 20
        assert proc.stdout == "captured"
        assert proc.returncode < 0

    def test_no_sentinel_leaves_a_short_child_alone(self, sinks):
        out, err = sinks
        proc = run_streaming(sh("printf ok"), stdout_sink=out, stderr_sink=err)
        assert proc.returncode == 0
        assert proc.stdout == "ok"

    def test_sentinel_that_never_appears_lets_the_child_finish(self, tmp_path, sinks):
        out, err = sinks
        proc = run_streaming(
            sh("printf finished"),
            completion_sentinel=str(tmp_path / "never"),
            stdout_sink=out,
            stderr_sink=err,
        )

        assert proc.returncode == 0
        assert proc.stdout == "finished"
