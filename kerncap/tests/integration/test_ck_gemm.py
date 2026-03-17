"""Docker-based integration test: Composable Kernel GEMM XDL FP16.

Validates kerncap's full pipeline against a CK GEMM example running
inside a rocm/composable_kernel container.

Container: rocm/composable_kernel:ck_pytorch
Requires: Docker, AMD GPU
"""

import os
import textwrap

import pytest

from tests.integration.conftest import run_streaming, skip_no_docker, skip_no_gpu


CONTAINER_IMAGE = "rocm/composable_kernel:ck_pytorch"

# Script to run INSIDE the container
INNER_SCRIPT = textwrap.dedent("""\
    #!/usr/bin/env python3
    import subprocess
    import sys
    import os
    import glob

    def run(cmd, **kwargs):
        print(f">>> {cmd}", flush=True)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
        print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, flush=True)
        return result

    # Install system dependencies for kerncap[full] (requires libdwarf for kernelDB)
    run("apt-get update -qq")
    run("apt-get install -y libdwarf-dev")

    # Install kerncap from mounted source
    result = run("pip install /workspace/kerncap[full]")
    if result.returncode != 0:
        print("kerncap install failed", file=sys.stderr)
        sys.exit(1)

    # Find the CK GEMM example binary
    ck_binary = None
    search_paths = [
        "/opt/rocm/bin/example_gemm_xdl_fp16",
        "/workspace/composable_kernel/build/bin/example_gemm_xdl_fp16",
    ]
    # Also check glob patterns
    for pattern in ["/opt/rocm/bin/example_gemm*fp16*",
                    "/workspace/*/build/bin/example_gemm*fp16*"]:
        search_paths.extend(glob.glob(pattern))

    for path in search_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            ck_binary = path
            break

    if not ck_binary:
        print("CK GEMM example binary not found — skipping", file=sys.stderr)
        print("CK GEMM integration test: SKIP (binary not found)")
        sys.exit(0)

    print(f"Found CK binary: {ck_binary}")

    # Profile the GEMM example
    result = run(f"kerncap profile -- {ck_binary} 1 1 1")
    if result.returncode != 0:
        print("Profile failed", file=sys.stderr)
        sys.exit(1)

    print("CK GEMM XDL integration test: PASS")
""")


@skip_no_docker
@skip_no_gpu
@pytest.mark.docker
class TestCKGemm:
    """Docker-based integration test for Composable Kernel GEMM XDL."""

    def test_profile_ck_gemm(self, tmp_path):
        """Profile CK GEMM XDL FP16 in a CK container."""
        script_path = tmp_path / "run_test.py"
        script_path.write_text(INNER_SCRIPT)

        kerncap_src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Stream output so you can see progress inside the container
        result = run_streaming(
            [
                "docker",
                "run",
                "--rm",
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
            timeout=600,
        )

        assert result.returncode == 0, (
            f"Container test failed (exit {result.returncode}):\n{result.stderr}"
        )
        assert "PASS" in result.stdout or "SKIP" in result.stdout
