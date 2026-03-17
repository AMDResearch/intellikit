"""Unit tests for kerncap.source_finder — kernel source location."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kerncap.source_finder import (
    detect_language,
    find_kernel_source,
    _extract_base_name,
    _has_kernel_definition,
    _find_definition_file,
    _detect_link_libraries,
    _defines_from_source_scan,
    _detect_compile_defines,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestDetectLanguage:
    """Tests for language auto-detection."""

    def test_detect_triton(self):
        """Detect Triton when @triton.jit kernel matches."""
        lang = detect_language("vector_add_kernel", str(FIXTURES))
        assert lang == "triton"

    def test_detect_hip(self):
        """Detect HIP when __global__ functions exist."""
        lang = detect_language("vector_add", str(FIXTURES))
        # May detect triton first if triton file matches; test with HIP-only dir
        assert lang in ("hip", "triton")

    def test_detect_unknown(self, tmp_path):
        """Return unknown for empty directory."""
        lang = detect_language("nonexistent", str(tmp_path))
        assert lang == "unknown"


class TestExtractBaseName:
    """Tests for demangled name parsing."""

    def test_simple_name(self):
        assert _extract_base_name("vector_add") == "vector_add"

    def test_void_prefix(self):
        assert _extract_base_name("void vector_add(float*, float*, int)") == "vector_add"

    def test_template(self):
        assert (
            _extract_base_name("void matmul_kernel<float>(float*, float*, float*, int)")
            == "matmul_kernel"
        )

    def test_namespaced(self):
        assert _extract_base_name("void ck::GridwiseGemm<float>::Run()") == "Run"

    def test_empty(self):
        assert _extract_base_name("") == ""

    def test_unbalanced_template_no_closing(self):
        assert _extract_base_name("mul_mat_vec_q<(ggml_type)39") == "mul_mat_vec_q"

    def test_unbalanced_template_truncated_profiler(self):
        assert (
            _extract_base_name("void mul_mat_vec_q<(ggml_type)39, 1, true, false")
            == "mul_mat_vec_q"
        )

    def test_unbalanced_template_with_void_prefix(self):
        assert (
            _extract_base_name("void mul_mat_q<(ggml_type)39, 128, true>(char const*, int")
            == "mul_mat_q"
        )

    def test_balanced_template_still_works(self):
        assert _extract_base_name("mul_mat_vec_q<(ggml_type)39, 1, true, false>") == "mul_mat_vec_q"


class TestFindTritonKernel:
    """Tests for Triton kernel source finding."""

    def test_find_triton_kernel(self):
        """Find the sample Triton kernel by name."""
        result = find_kernel_source(
            "vector_add_kernel",
            str(FIXTURES),
            language="triton",
        )
        assert result is not None
        assert result.language == "triton"
        assert result.kernel_function == "vector_add_kernel"
        assert "sample_triton_kernel.py" in result.main_file

    def test_triton_substring_match(self):
        """Find kernel with substring of name."""
        result = find_kernel_source(
            "vector_add",
            str(FIXTURES),
            language="triton",
        )
        assert result is not None
        assert result.kernel_function == "vector_add_kernel"

    def test_triton_not_found(self):
        """Return None when kernel doesn't exist."""
        result = find_kernel_source(
            "nonexistent_kernel_xyz",
            str(FIXTURES),
            language="triton",
        )
        assert result is None


class TestFindHipKernel:
    """Tests for HIP kernel source finding."""

    def test_find_hip_kernel(self):
        """Find the sample HIP kernel by name."""
        result = find_kernel_source(
            "vector_add",
            str(FIXTURES),
            language="hip",
        )
        assert result is not None
        assert result.language == "hip"
        assert result.kernel_function == "vector_add"
        assert "sample_hip_kernel.hip" in result.main_file

    def test_hip_not_found(self, tmp_path):
        """Return None in empty directory."""
        result = find_kernel_source(
            "vector_add",
            str(tmp_path),
            language="hip",
        )
        assert result is None


MULTI_TU = FIXTURES / "multi_tu"


class TestMultiTranslationUnit:
    """Tests for kernels defined in a separate translation unit.

    Mirrors the llama.cpp pattern where a driver file (driver.cu)
    references a kernel by name but the __global__ definition lives
    in a sibling file (kernels.cu).
    """

    def test_finds_definition_file_not_driver(self):
        """Source finder should pick kernels.cu (definition), not driver.cu."""
        result = find_kernel_source(
            "mul_mat_vec_q",
            str(MULTI_TU),
            language="hip",
        )
        assert result is not None
        assert result.kernel_function == "mul_mat_vec_q"
        assert "kernels.cu" in result.main_file
        assert "driver.cu" not in result.main_file

    def test_includes_traced_from_definition(self):
        """#include deps should be traced from the definition file."""
        result = find_kernel_source(
            "mul_mat_vec_q",
            str(MULTI_TU),
            language="hip",
        )
        assert result is not None
        basenames = [os.path.basename(f) for f in result.source_files]
        assert "kernels.cu" in basenames
        assert "helpers.cuh" in basenames

    def test_has_kernel_definition_positive(self):
        files = [str(MULTI_TU / "kernels.cu")]
        assert _has_kernel_definition("mul_mat_vec_q", files) is True

    def test_has_kernel_definition_negative(self):
        files = [str(MULTI_TU / "driver.cu")]
        assert _has_kernel_definition("mul_mat_vec_q", files) is False

    def test_find_definition_file(self):
        search = [
            str(MULTI_TU / "driver.cu"),
            str(MULTI_TU / "kernels.cu"),
        ]
        result = _find_definition_file("mul_mat_vec_q", search, set())
        assert result is not None
        assert "kernels.cu" in result

    def test_find_definition_file_excludes(self):
        """Already-collected files are skipped."""
        search = [
            str(MULTI_TU / "driver.cu"),
            str(MULTI_TU / "kernels.cu"),
        ]
        exclude = {os.path.abspath(str(MULTI_TU / "kernels.cu"))}
        result = _find_definition_file("mul_mat_vec_q", search, exclude)
        assert result is None

    def test_post_verification_adds_missing_definition(self, tmp_path):
        """When fallback picks the wrong main_file, post-verification
        corrects it by finding the actual definition file."""
        driver = tmp_path / "driver.cu"
        driver.write_text(
            "__global__ void unrelated_kernel(int x) {}\n"
            "void host_launch_my_kernel();\n"
            "void call_my_kernel() { host_launch_my_kernel(); }\n"
        )
        impl = tmp_path / "impl.cu"
        impl.write_text(
            "__global__ void my_kernel(float* a, int n) {\n"
            "    int i = threadIdx.x; if (i < n) a[i] = 0;\n"
            "}\n"
        )
        result = find_kernel_source(
            "my_kernel",
            str(tmp_path),
            language="hip",
        )
        assert result is not None
        assert "impl.cu" in result.main_file


class TestDirectTranslationUnit:
    """Tests for kernels defined in a .cu file that IS the translation unit.

    Mirrors the llama.cpp mmvq.cu pattern where the __global__ kernel
    lives directly in a .cu file listed in compile_commands.json,
    rather than in a header #included by a separate TU.
    """

    def test_cu_file_is_own_tu(self, tmp_path):
        """When a .cu file has a compile_commands.json entry, it should
        be recognised as its own translation unit with a compile command."""
        kernel = tmp_path / "mmvq.cu"
        kernel.write_text(
            "__global__ void mul_mat_vec_q(float* a, int n) {\n"
            "    int i = threadIdx.x; if (i < n) a[i] *= 2.0f;\n"
            "}\n"
        )
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        cc = build_dir / "compile_commands.json"
        cc.write_text(
            json.dumps(
                [
                    {
                        "directory": str(build_dir),
                        "command": f"/opt/rocm/llvm/bin/clang -DFOO -x hip {kernel} -c -o mmvq.o",
                        "file": str(kernel),
                    }
                ]
            )
        )

        result = find_kernel_source(
            "mul_mat_vec_q",
            str(tmp_path),
            language="hip",
        )
        assert result is not None
        assert result.kernel_function == "mul_mat_vec_q"
        assert "mmvq.cu" in result.main_file
        assert result.compile_command != ""
        assert result.translation_unit != ""
        assert "mmvq.cu" in result.translation_unit

    def test_cu_file_without_compile_commands(self, tmp_path):
        """Without compile_commands.json, a .cu file has no compile command."""
        kernel = tmp_path / "mmvq.cu"
        kernel.write_text(
            "__global__ void mul_mat_vec_q(float* a, int n) {\n"
            "    int i = threadIdx.x; if (i < n) a[i] *= 2.0f;\n"
            "}\n"
        )

        result = find_kernel_source(
            "mul_mat_vec_q",
            str(tmp_path),
            language="hip",
        )
        assert result is not None
        assert result.kernel_function == "mul_mat_vec_q"
        assert result.compile_command == ""

    def test_header_still_uses_include_search(self, tmp_path):
        """A .cuh header should NOT match Phase 0; the include search
        should find the .cu file that #includes it instead."""
        header = tmp_path / "mmq.cuh"
        header.write_text(
            "__global__ void mul_mat_q(float* a, int n) {\n"
            "    int i = threadIdx.x; if (i < n) a[i] *= 2.0f;\n"
            "}\n"
        )
        tu = tmp_path / "mmq_instance.cu"
        tu.write_text('#include "mmq.cuh"\n')

        build_dir = tmp_path / "build"
        build_dir.mkdir()
        cc = build_dir / "compile_commands.json"
        cc.write_text(
            json.dumps(
                [
                    {
                        "directory": str(build_dir),
                        "command": f"/opt/rocm/llvm/bin/clang -x hip {tu} -c -o mmq.o",
                        "file": str(tu),
                    }
                ]
            )
        )

        result = find_kernel_source(
            "mul_mat_q",
            str(tmp_path),
            language="hip",
        )
        assert result is not None
        assert result.compile_command != ""
        assert "mmq_instance.cu" in result.translation_unit


def _make_ldd_output(project_lib_path: str) -> str:
    """Build a fake ``ldd`` stdout with one project-local lib and common system libs."""
    return (
        f"\tlibggml.so.0 => {project_lib_path} (0x00007f0000000000)\n"
        "\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007efff0000000)\n"
        "\tlibm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007efff1000000)\n"
        "\tlibpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007efff2000000)\n"
        "\tlibdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007efff3000000)\n"
        "\tlibrt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007efff4000000)\n"
        "\tlibstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007efff5000000)\n"
    )


@pytest.fixture()
def fake_ldd_project(tmp_path):
    """Fixture that creates a fake project tree and patches ``ldd`` output.

    Yields a dict with:
      - ``binary``: path to a dummy (empty) executable file
      - ``source``: path to the fake project source directory
      - ``lib_file``: path to the fake project-local shared library
    """
    source = tmp_path / "myproject"
    source.mkdir()
    build = source / "build"
    build.mkdir()

    binary = build / "my-app"
    binary.write_bytes(b"")

    lib_dir = build / "lib"
    lib_dir.mkdir()
    lib_file = lib_dir / "libggml.so.0"
    lib_file.write_bytes(b"")

    ldd_out = _make_ldd_output(str(lib_file))

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ldd_out

    with patch("subprocess.run", return_value=mock_result):
        yield {
            "binary": str(binary),
            "source": str(source),
            "lib_file": str(lib_file),
        }


class TestLinkLibraryDetection:
    """Tests for shared library auto-detection via ldd."""

    def test_no_binary_returns_empty(self):
        paths, libs = _detect_link_libraries(None, "/tmp")
        assert paths == []
        assert libs == []

    def test_nonexistent_binary_returns_empty(self, tmp_path):
        paths, libs = _detect_link_libraries(
            str(tmp_path / "nonexistent"),
            str(tmp_path),
        )
        assert paths == []
        assert libs == []

    def test_detects_project_local_libs(self, fake_ldd_project):
        """Project-local libs appear in the returned paths and names."""
        paths, libs = _detect_link_libraries(
            fake_ldd_project["binary"],
            fake_ldd_project["source"],
        )
        assert len(paths) > 0
        assert any("ggml" in lib for lib in libs)

    def test_filters_system_libs(self, fake_ldd_project):
        """System libraries (libc, libm, etc.) must not appear in the result."""
        _, libs = _detect_link_libraries(
            fake_ldd_project["binary"],
            fake_ldd_project["source"],
        )
        for lib in libs:
            assert lib not in ("c", "m", "pthread", "dl", "rt", "stdc++")

    def test_ldd_failure_returns_empty(self, tmp_path):
        """Non-zero ldd exit code produces empty results."""
        binary = tmp_path / "app"
        binary.write_bytes(b"")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            paths, libs = _detect_link_libraries(str(binary), str(tmp_path))
        assert paths == []
        assert libs == []

    def test_ldd_exception_returns_empty(self, tmp_path):
        """OSError / timeout from ldd produces empty results gracefully."""
        binary = tmp_path / "app"
        binary.write_bytes(b"")

        with patch("subprocess.run", side_effect=OSError("ldd not found")):
            paths, libs = _detect_link_libraries(str(binary), str(tmp_path))
        assert paths == []
        assert libs == []

    def test_lib_path_returned(self, fake_ldd_project):
        """The directory containing the project-local lib is included in paths."""
        lib_dir = str(Path(fake_ldd_project["lib_file"]).parent)
        paths, _ = _detect_link_libraries(
            fake_ldd_project["binary"],
            fake_ldd_project["source"],
        )
        assert any(os.path.abspath(p) == os.path.abspath(lib_dir) for p in paths)


class TestCompileDefines:
    """Tests for compile-define detection."""

    def test_source_scan_finds_ifdef(self, tmp_path):
        """Detect defines from #ifdef guards with HIP substring."""
        src = tmp_path / "common.cuh"
        src.write_text("#ifdef GGML_USE_HIP\nint x;\n#endif\n")
        result = _defines_from_source_scan([str(src)])
        assert "GGML_USE_HIP" in result

    def test_source_scan_finds_if_defined(self, tmp_path):
        """Detect defines from #if defined() guards."""
        src = tmp_path / "common.cuh"
        src.write_text("#if defined(USE_ROCM)\nint x;\n#endif\n")
        result = _defines_from_source_scan([str(src)])
        assert "USE_ROCM" in result

    def test_source_scan_finds_amd_substring(self, tmp_path):
        """Detect defines containing AMD substring."""
        src = tmp_path / "foo.h"
        src.write_text("#if defined(__HIP_PLATFORM_AMD__)\nint x;\n#endif\n")
        result = _defines_from_source_scan([str(src)])
        assert "__HIP_PLATFORM_AMD__" in result

    def test_source_scan_ignores_non_hip_guards(self, tmp_path):
        """Non-HIP/ROCM/AMD guards should be ignored."""
        src = tmp_path / "foo.h"
        src.write_text("#ifdef DEBUG\nint x;\n#endif\n")
        result = _defines_from_source_scan([str(src)])
        assert len(result) == 0

    def test_source_scan_multiple_files(self, tmp_path):
        """Detect defines across multiple source files."""
        f1 = tmp_path / "a.h"
        f1.write_text("#ifdef GGML_USE_HIP\nint x;\n#endif\n")
        f2 = tmp_path / "b.h"
        f2.write_text("#if defined(USE_ROCM)\nint y;\n#endif\n")
        result = _defines_from_source_scan([str(f1), str(f2)])
        assert "GGML_USE_HIP" in result
        assert "USE_ROCM" in result

    def test_compile_commands_json(self, tmp_path):
        """Strategy 1: extract defines from compile_commands.json."""
        src = tmp_path / "kernel.hip"
        src.write_text("__global__ void foo() {}\n")
        cc = tmp_path / "compile_commands.json"
        cc.write_text(
            json.dumps(
                [
                    {
                        "directory": str(tmp_path),
                        "command": f"hipcc -DGGML_USE_HIP -DFOO=1 -o kernel.o {src}",
                        "file": str(src),
                    }
                ]
            )
        )
        result = _detect_compile_defines(
            [str(src)],
            binary_path=str(tmp_path / "libfoo.so"),
        )
        assert "GGML_USE_HIP" in result
        assert "FOO=1" in result

    def test_extra_defines_merged(self, tmp_path):
        """User-supplied extra_defines are merged in."""
        src = tmp_path / "foo.h"
        src.write_text("#ifdef GGML_USE_HIP\nint x;\n#endif\n")
        result = _detect_compile_defines(
            [str(src)],
            extra_defines=["MY_CUSTOM_DEF"],
        )
        assert "GGML_USE_HIP" in result
        assert "MY_CUSTOM_DEF" in result

    def test_empty_sources(self):
        """No sources produces no defines."""
        result = _detect_compile_defines([])
        assert result == []
