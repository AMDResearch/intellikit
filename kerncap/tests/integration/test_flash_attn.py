"""Docker-based integration test: Flash Attention 2 Triton backend.

Validates kerncap's full pipeline against the Flash Attention 2 Triton
kernel running inside a rocm/pytorch container. Install and usage follow
ROCm docs: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html

Container: rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.7.1
Requires: Docker, AMD GPU, network access (clone + build in container).

NOTE: rocprofv3's ring_buffer teardown crashes with ``munmap failed: Invalid
Argument`` when its stdout/stderr are pipes (Python ``capture_output=True``).
Two mitigations are applied here:
  1. ``profiler.py`` redirects rocprofv3 I/O to temp files instead of pipes.
  2. Docker is launched with ``--init`` so ``tini`` is PID 1 (proper signal
     forwarding and zombie reaping, matching interactive-session behaviour).

Triton is already included in the container; we only install flash-attn from
source with FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE. First run can take
15–20+ min while flash-attn builds.
"""

import os
import textwrap

import pytest

from tests.integration.conftest import run_streaming, skip_no_docker, skip_no_gpu

CONTAINER_IMAGE = "rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.7.1"

# Script to run INSIDE the container. Benchmark code lives in scripts/bench_fa.py
# and is copied into the test tmp_path so we avoid nested triple-quoted strings.
INNER_SCRIPT = textwrap.dedent("""\
    #!/usr/bin/env python3
    import subprocess
    import sys
    import os

    def run(cmd, **kwargs):
        print(f">>> {cmd}", flush=True)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
        print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, flush=True)
        return result

    def run_passthrough(cmd, **kwargs):
        \"\"\"Run a command with stdout/stderr going directly to the parent's FDs.

        Avoids creating pipes around rocprofv3, which triggers a ring_buffer
        teardown crash (munmap EINVAL) when its FDs are pipe-backed.
        \"\"\"
        print(f">>> {cmd}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        return subprocess.run(cmd, shell=True, **kwargs)

    # Install system dependencies for kerncap[full] (requires libdwarf for kernelDB)
    run("apt-get update -qq")
    run("apt-get install -y libdwarf-dev")

    # Install Flash Attention 2 with Triton backend (ROCm docs: clone + setup.py;
    # Triton is already in the container, do not reinstall it.)
    # https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html
    run("pip install ninja")
    run("git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention")
    result = run("cd /tmp/flash-attention && FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install",
                 env={**os.environ, "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE"})
    if result.returncode != 0:
        print("flash-attn install failed", file=sys.stderr)
        sys.exit(1)

    # Install kerncap from mounted source
    run("pip install /workspace/kerncap[full]")

    # Warmup: run once WITHOUT profiling so Triton JIT-compiles and caches
    # its kernels, preventing compiler child processes during the profiled run.
    fa_env = {**os.environ, "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE"}
    result = run("python /workspace/test/bench_fa.py", env=fa_env)
    if result.returncode != 0:
        print("Warmup run failed", file=sys.stderr)
        sys.exit(1)

    # Profile with warm Triton cache.
    # Use run_passthrough so kerncap (and therefore rocprofv3) does NOT get
    # pipe-backed stdout/stderr -- profiler.py already redirects rocprofv3 I/O
    # to temp files, but this also avoids an outer pipe layer from run_test.py.
    result = run_passthrough("kerncap profile -- python /workspace/test/bench_fa.py", env=fa_env)
    if result.returncode != 0:
        print("Profile failed", file=sys.stderr)
        sys.exit(1)

    print("Flash Attention Triton integration test: PASS")
""")


@skip_no_docker
@skip_no_gpu
@pytest.mark.docker
class TestFlashAttentionTriton:
    """Docker-based integration test for Flash Attention Triton backend."""

    def test_profile_flash_attention(self, tmp_path):
        """Profile Flash Attention in a ROCm PyTorch container."""
        # Write the inner test script and copy the benchmark script (avoids nested
        # triple-quoted strings; benchmark is real Python in scripts/bench_fa.py)
        script_path = tmp_path / "run_test.py"
        script_path.write_text(INNER_SCRIPT)

        this_dir = os.path.dirname(os.path.abspath(__file__))
        bench_src = os.path.join(this_dir, "scripts", "bench_fa.py")
        with open(bench_src) as f:
            (tmp_path / "bench_fa.py").write_text(f.read())

        kerncap_src = os.path.dirname(os.path.dirname(this_dir))

        # Run in Docker; stream output so you can see progress (e.g. pip, build)
        result = run_streaming(
            [
                "docker",
                "run",
                "--rm",
                "--init",
                "--device=/dev/kfd",
                "--device=/dev/dri",
                "--security-opt",
                "seccomp=unconfined",
                "--ipc=host",
                "--network=host",
                "--group-add",
                "video",
                "--cap-add=SYS_PTRACE",
                "--privileged=true",
                "--shm-size=128GB",
                "-v",
                f"{kerncap_src}:/workspace/kerncap:ro",
                "-v",
                f"{str(tmp_path)}:/workspace/test",
                CONTAINER_IMAGE,
                "python3",
                "/workspace/test/run_test.py",
            ],
            timeout=1200,  # flash-attn builds from source in-container; first run can be 15–20+ min
        )

        assert result.returncode == 0, (
            f"Container test failed (exit {result.returncode}):\n{result.stderr}"
        )
        assert "PASS" in result.stdout
