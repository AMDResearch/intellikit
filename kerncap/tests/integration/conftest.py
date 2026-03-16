"""Shared fixtures for integration tests."""

import io
import os
import shutil
import subprocess
import sys
import threading

import pytest


def run_streaming(args, timeout=None):
    """Run a subprocess, stream stdout/stderr in real time, and return a CompletedProcess.

    Use this for long-running Docker (or other) commands so you can see progress
    while the test runs. Output is both streamed and captured for assertions.
    """
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    out_buf = io.StringIO()
    err_buf = io.StringIO()

    def read_stream(stream, buf, dest):
        for line in iter(stream.readline, ""):
            if not line:
                break
            buf.write(line)
            dest.write(line)
            dest.flush()

    t_out = threading.Thread(target=read_stream, args=(proc.stdout, out_buf, sys.stdout))
    t_err = threading.Thread(target=read_stream, args=(proc.stderr, err_buf, sys.stderr))
    t_out.daemon = True
    t_err.daemon = True
    t_out.start()
    t_err.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    t_out.join(timeout=1)
    t_err.join(timeout=1)
    return subprocess.CompletedProcess(
        proc.args, proc.returncode, out_buf.getvalue(), err_buf.getvalue()
    )


def _has_amd_gpu() -> bool:
    """Check if an AMD GPU is accessible via /dev/kfd."""
    return os.path.exists("/dev/kfd")


def _has_docker() -> bool:
    """Check if docker CLI is available."""
    return shutil.which("docker") is not None


def _has_rocprof() -> bool:
    """Check if rocprofv3 is available on PATH."""
    return shutil.which("rocprofv3") is not None


skip_no_gpu = pytest.mark.skipif(
    not _has_amd_gpu(), reason="No AMD GPU detected (/dev/kfd missing)"
)
skip_no_docker = pytest.mark.skipif(
    not _has_docker(), reason="Docker not available"
)
skip_no_rocprof = pytest.mark.skipif(
    not _has_rocprof(), reason="rocprofv3 not available on PATH"
)


@pytest.fixture
def tmp_output(tmp_path):
    """Provide a temporary output directory for test artifacts."""
    out = tmp_path / "output"
    out.mkdir()
    return out
