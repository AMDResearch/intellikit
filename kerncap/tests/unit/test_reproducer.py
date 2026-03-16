"""Unit tests for kerncap.reproducer — VA-faithful replay reproducer."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from kerncap.reproducer import (
    generate_hsaco_reproducer,
    generate_triton_reproducer,
)
from kerncap.source_finder import KernelSource

FIXTURES = Path(__file__).parent / "fixtures"


def _make_capture_dir(tmp_path, metadata_file=None, metadata_dict=None):
    """Create a mock capture directory with dispatch.json format.

    Uses either a fixture file (metadata_file) or a dict (metadata_dict).
    Creates minimal memory_regions.json and memory blobs.
    """
    cap = tmp_path / "capture"
    cap.mkdir(exist_ok=True)
    mem_dir = cap / "memory"
    mem_dir.mkdir(exist_ok=True)

    if metadata_dict:
        meta = metadata_dict
    elif metadata_file:
        meta = json.loads((FIXTURES / metadata_file).read_text())
    else:
        meta = {
            "mangled_name": "_Z11test_kernelPfS_S_i",
            "demangled_name": "void test_kernel(float*, float*, float*, int)",
            "isa_name": "amdgcn-amd-amdhsa--gfx90a",
            "grid": [256, 1, 1],
            "block": [256, 1, 1],
            "kernarg_size": 32,
        }

    (cap / "dispatch.json").write_text(json.dumps(meta))

    # Minimal kernarg.bin
    kernarg_size = meta.get("kernarg_size", 32)
    (cap / "kernarg.bin").write_bytes(b"\x00" * kernarg_size)

    # Minimal HSACO (empty placeholder)
    (cap / "kernel.hsaco").write_bytes(b"\x00" * 64)

    # Minimal memory regions
    regions = {
        "regions": [
            {"base": 140730000000000, "size": 4096,
             "is_pool": True, "is_vmem": False,
             "contains_kernarg": False, "handle": 0, "access": 0}
        ]
    }
    (cap / "memory_regions.json").write_text(json.dumps(regions))
    (mem_dir / f"region_{140730000000000:x}.bin").write_bytes(b"\x00" * 4096)

    return str(cap)


@pytest.fixture
def capture_dir(tmp_path):
    """Create a mock capture directory with new VA-faithful format."""
    return _make_capture_dir(tmp_path)


class TestHsacoReproducer:
    """Tests for HSACO reproducer generation (VA-faithful replay)."""

    def test_generates_files(self, capture_dir, tmp_path):
        """Check that Makefile and capture/ are generated."""
        output = str(tmp_path / "repro")

        generate_hsaco_reproducer(capture_dir, output)

        assert os.path.exists(os.path.join(output, "Makefile"))
        assert os.path.exists(os.path.join(output, "capture", "dispatch.json"))
        assert os.path.exists(os.path.join(output, "capture", "kernarg.bin"))
        assert os.path.exists(os.path.join(output, "capture", "memory_regions.json"))

    def test_makefile_uses_replay(self, capture_dir, tmp_path):
        """Check that the generated Makefile invokes kerncap-replay."""
        output = str(tmp_path / "repro")

        generate_hsaco_reproducer(capture_dir, output)

        content = Path(os.path.join(output, "Makefile")).read_text()
        assert "kerncap-replay" in content
        assert "REPLAY" in content
        assert "CAPTURE_DIR" in content

    def test_copies_original_source_files(self, capture_dir, tmp_path):
        """Original source files should be copied to kernel_variant.cpp."""
        src_root = tmp_path / "src"
        src_root.mkdir()
        main_kernel = src_root / "kernel.hip"
        main_kernel.write_text("__global__ void k() {}\n")

        ks = KernelSource(
            language="hip",
            kernel_name="k",
            main_file=str(main_kernel),
            source_files=[],
            compile_command="hipcc -c kernel.hip -o kernel.o",
            compile_dir=str(src_root),
        )

        output = str(tmp_path / "repro")

        generate_hsaco_reproducer(
            capture_dir, output,
            kernel_source=ks,
        )

        assert os.path.exists(os.path.join(output, "kernel_variant.cpp"))
        assert os.path.exists(os.path.join(output, "vfs.yaml"))

    def test_makefile_has_recompile_when_source(self, capture_dir, tmp_path):
        """Makefile should have recompile target when source is provided."""
        src_root = tmp_path / "src"
        src_root.mkdir()
        main_kernel = src_root / "kernel.hip"
        main_kernel.write_text("__global__ void k() {}\n")
        
        ks = KernelSource(
            language="hip",
            kernel_name="k",
            main_file=str(main_kernel),
            source_files=[],
            compile_command="hipcc -c kernel.hip -o kernel.o",
            compile_dir=str(src_root),
        )
        
        output = str(tmp_path / "repro")

        generate_hsaco_reproducer(
            capture_dir, output,
            kernel_source=ks,
        )

        content = Path(os.path.join(output, "Makefile")).read_text()
        assert "recompile" in content
        assert "-ivfsoverlay" in content
        assert "hipcc" in content
        assert " -c " not in content
        assert "--cuda-device-only" in content
        assert "--no-gpu-bundle-output" in content
        assert "optimized.hsaco" in content
        assert "clang-offload-bundler" not in content

    def test_copies_dependency_headers_to_deps(self, capture_dir, tmp_path):
        """Dependency headers from source_files should be copied to deps/."""
        src_root = tmp_path / "src"
        src_root.mkdir()
        main_kernel = src_root / "mmq.cu"
        main_kernel.write_text('#include "vecdotq.cuh"\n__global__ void k() {}\n')
        dep1 = src_root / "vecdotq.cuh"
        dep1.write_text("inline __device__ int vec_dot() { return 0; }\n")
        dep2 = src_root / "common.cuh"
        dep2.write_text("#define COMMON 1\n")

        ks = KernelSource(
            language="hip",
            kernel_name="k",
            main_file=str(main_kernel),
            source_files=[str(main_kernel), str(dep1), str(dep2)],
            compile_command="hipcc -c mmq.cu -o mmq.o",
            compile_dir=str(src_root),
        )

        output = str(tmp_path / "repro")
        generate_hsaco_reproducer(capture_dir, output, kernel_source=ks)

        assert os.path.exists(os.path.join(output, "kernel_variant.cpp"))
        assert os.path.exists(os.path.join(output, "deps", "vecdotq.cuh"))
        assert os.path.exists(os.path.join(output, "deps", "common.cuh"))

        vfs = json.loads(Path(os.path.join(output, "vfs.yaml")).read_text())
        assert vfs["version"] == 0
        all_vfs_names = []
        for root in vfs["roots"]:
            for entry in root["contents"]:
                all_vfs_names.append(entry["name"])
        assert "mmq.cu" in all_vfs_names
        assert "vecdotq.cuh" in all_vfs_names
        assert "common.cuh" in all_vfs_names

    def test_deps_from_multiple_directories(self, capture_dir, tmp_path):
        """Dependencies from different directories produce multiple VFS roots."""
        src_root = tmp_path / "src"
        cuda_dir = src_root / "ggml-cuda"
        common_dir = src_root / "ggml-common"
        cuda_dir.mkdir(parents=True)
        common_dir.mkdir(parents=True)

        main_kernel = cuda_dir / "mmq.cu"
        main_kernel.write_text("__global__ void k() {}\n")
        dep_same = cuda_dir / "vecdotq.cuh"
        dep_same.write_text("// vecdotq\n")
        dep_other = common_dir / "types.cuh"
        dep_other.write_text("// types\n")

        ks = KernelSource(
            language="hip",
            kernel_name="k",
            main_file=str(main_kernel),
            source_files=[str(main_kernel), str(dep_same), str(dep_other)],
            compile_command="hipcc -c mmq.cu -o mmq.o",
            compile_dir=str(cuda_dir),
        )

        output = str(tmp_path / "repro")
        generate_hsaco_reproducer(capture_dir, output, kernel_source=ks)

        vfs = json.loads(Path(os.path.join(output, "vfs.yaml")).read_text())
        root_dirs = [r["name"] for r in vfs["roots"]]
        assert len(root_dirs) == 2
        assert str(cuda_dir.resolve()) in root_dirs
        assert str(common_dir.resolve()) in root_dirs

    def test_no_deps_dir_when_no_dependencies(self, capture_dir, tmp_path):
        """No deps/ directory should be created when there are no dependencies."""
        src_root = tmp_path / "src"
        src_root.mkdir()
        main_kernel = src_root / "kernel.hip"
        main_kernel.write_text("__global__ void k() {}\n")

        ks = KernelSource(
            language="hip",
            kernel_name="k",
            main_file=str(main_kernel),
            source_files=[str(main_kernel)],
            compile_command="hipcc -c kernel.hip -o kernel.o",
            compile_dir=str(src_root),
        )

        output = str(tmp_path / "repro")
        generate_hsaco_reproducer(capture_dir, output, kernel_source=ks)

        assert os.path.exists(os.path.join(output, "kernel_variant.cpp"))
        assert not os.path.exists(os.path.join(output, "deps"))

    def test_dep_basename_collision(self, capture_dir, tmp_path):
        """Same-named deps from different dirs should not overwrite each other."""
        src_root = tmp_path / "src"
        dir_a = src_root / "a"
        dir_b = src_root / "b"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)

        main_kernel = dir_a / "kernel.cu"
        main_kernel.write_text("__global__ void k() {}\n")
        dep_a = dir_a / "common.cuh"
        dep_a.write_text("// from a\n")
        dep_b = dir_b / "common.cuh"
        dep_b.write_text("// from b\n")

        ks = KernelSource(
            language="hip",
            kernel_name="k",
            main_file=str(main_kernel),
            source_files=[str(main_kernel), str(dep_a), str(dep_b)],
            compile_command="hipcc -c kernel.cu -o kernel.o",
            compile_dir=str(dir_a),
        )

        output = str(tmp_path / "repro")
        generate_hsaco_reproducer(capture_dir, output, kernel_source=ks)

        deps_dir = os.path.join(output, "deps")
        dep_files = os.listdir(deps_dir)
        assert len(dep_files) == 2
        assert "common.cuh" in dep_files
        assert "b__common.cuh" in dep_files

    def test_gpu_arch_from_isa_name(self, tmp_path):
        """GPU arch should be derived from isa_name field."""
        meta = {
            "demangled_name": "void k()",
            "isa_name": "amdgcn-amd-amdhsa--gfx942",
            "grid": [1, 1, 1],
            "block": [1, 1, 1],
            "kernarg_size": 0,
        }
        cap = _make_capture_dir(tmp_path, metadata_dict=meta)
        output = str(tmp_path / "repro")

        generate_hsaco_reproducer(cap, output)

        content = Path(os.path.join(output, "Makefile")).read_text()
        assert "gfx942" in content


class TestTritonReproducer:
    """Tests for Triton reproducer generation."""

    def _make_triton_capture(self, tmp_path):
        """Create a capture dir with legacy metadata.json for Triton."""
        cap = tmp_path / "capture"
        cap.mkdir()
        meta = {
            "kernel_name": "vector_add_kernel",
            "grid": {"x": 4, "y": 1, "z": 1},
            "block": {"x": 1024, "y": 1, "z": 1},
            "args": [],
        }
        (cap / "metadata.json").write_text(json.dumps(meta))
        return str(cap)

    def test_generates_files(self, tmp_path):
        """Check that reproducer.py is generated."""
        cap = self._make_triton_capture(tmp_path)
        output = str(tmp_path / "repro")

        ks = KernelSource(
            language="triton",
            kernel_name="vector_add_kernel",
            main_file=str(FIXTURES / "sample_triton_kernel.py"),
            source_files=[str(FIXTURES / "sample_triton_kernel.py")],
            kernel_function="vector_add_kernel",
        )

        generate_triton_reproducer(cap, ks, output)

        assert os.path.exists(os.path.join(output, "reproducer.py"))
        assert os.path.exists(os.path.join(output, "sample_triton_kernel.py"))

    def test_reproducer_imports_kernel(self, tmp_path):
        """Check that the generated script imports the kernel."""
        cap = self._make_triton_capture(tmp_path)
        output = str(tmp_path / "repro")

        ks = KernelSource(
            language="triton",
            kernel_name="vector_add_kernel",
            main_file=str(FIXTURES / "sample_triton_kernel.py"),
            source_files=[str(FIXTURES / "sample_triton_kernel.py")],
            kernel_function="vector_add_kernel",
        )

        generate_triton_reproducer(cap, ks, output)

        content = Path(os.path.join(output, "reproducer.py")).read_text()
        assert "from sample_triton_kernel import vector_add_kernel" in content
        assert "_kernel[grid]" in content
        assert "_kernel = vector_add_kernel" in content
