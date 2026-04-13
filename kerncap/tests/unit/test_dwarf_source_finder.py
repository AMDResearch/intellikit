"""Unit tests for DWARF-based source discovery in kerncap.source_finder."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from kerncap.source_finder import (
    _extract_dwarf_source_files,
    _find_hip_kernel_via_dwarf,
    _find_tu_by_symbol,
    _parse_llvm_dwarfdump_output,
    _parse_readelf_output,
    _pick_main_file,
    find_kernel_source,
)

SAMPLE_DWARFDUMP_OUTPUT = """\
debug_line[0x00000000]
Line table prologue:
    total_length: 0x000001e8
          format: DWARF32
         version: 5
    address_size: 8
   seg_select_size: 0
 prologue_length: 0x00000067
 min_inst_length: 1
max_ops_per_inst: 1
 default_is_stmt: 1
       line_base: -5
      line_range: 14
     opcode_base: 13
standard_opcode_lengths[DW_LNS_copy] = 0
standard_opcode_lengths[DW_LNS_advance_pc] = 1
standard_opcode_lengths[DW_LNS_advance_line] = 1
standard_opcode_lengths[DW_LNS_set_file] = 1
standard_opcode_lengths[DW_LNS_set_column] = 1
standard_opcode_lengths[DW_LNS_negate_stmt] = 0
standard_opcode_lengths[DW_LNS_set_basic_block] = 0
standard_opcode_lengths[DW_LNS_const_add_pc] = 0
standard_opcode_lengths[DW_LNS_fixed_advance_pc] = 1
standard_opcode_lengths[DW_LNS_set_prologue_end] = 0
standard_opcode_lengths[DW_LNS_set_epilogue_begin] = 0
standard_opcode_lengths[DW_LNS_set_isa] = 1
include_directories[  0] = "/home/user/project/build"
include_directories[  1] = "/home/user/project/src"
include_directories[  2] = "/home/user/project/include"
include_directories[  3] = "/opt/rocm/include"
file_names[  0]:
           name: "main.cpp"
      dir_index: 1
   md5_checksum: 0123456789abcdef0123456789abcdef
file_names[  1]:
           name: "helper.h"
      dir_index: 2
   md5_checksum: fedcba9876543210fedcba9876543210
file_names[  2]:
           name: "hip_runtime.h"
      dir_index: 3
   md5_checksum: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
"""

SAMPLE_READELF_OUTPUT = """\
Contents of the .debug_line section:

CU: /home/user/project/src/main.cpp:
File name                            Line number    Starting address    View    Stmt
/home/user/project/src/main.cpp               1            0x1000               x
/home/user/project/include/helper.h          10            0x1020               x
/opt/rocm/include/hip_runtime.h              20            0x1040               x
/home/user/project/src/main.cpp              30            0x1060               x
"""


class TestParseLlvmDwarfdumpOutput:
    """Tests for DWARF line table prologue parsing."""

    def test_parse_basic_output(self):
        files = _parse_llvm_dwarfdump_output(SAMPLE_DWARFDUMP_OUTPUT)
        assert len(files) == 3
        assert "/home/user/project/src/main.cpp" in files
        assert "/home/user/project/include/helper.h" in files
        assert "/opt/rocm/include/hip_runtime.h" in files

    def test_empty_output_returns_empty(self):
        assert _parse_llvm_dwarfdump_output("") == []

    def test_no_file_names_returns_empty(self):
        output = 'include_directories[  0] = "/path"\n'
        assert _parse_llvm_dwarfdump_output(output) == []

    def test_absolute_name_ignores_dir_index(self):
        output = (
            'include_directories[  0] = "/should/not/use"\n'
            "file_names[  0]:\n"
            '           name: "/absolute/path/file.cpp"\n'
            "      dir_index: 0\n"
        )
        files = _parse_llvm_dwarfdump_output(output)
        assert len(files) == 1
        assert files[0] == "/absolute/path/file.cpp"

    def test_deduplication(self):
        output = (
            'include_directories[  0] = "/path"\n'
            "file_names[  0]:\n"
            '           name: "file.cpp"\n'
            "      dir_index: 0\n"
            "file_names[  1]:\n"
            '           name: "file.cpp"\n'
            "      dir_index: 0\n"
        )
        files = _parse_llvm_dwarfdump_output(output)
        assert len(files) == 1

    def test_missing_dir_index_uses_name_as_is(self):
        output = 'file_names[  0]:\n           name: "orphan.cpp"\n      dir_index: 99\n'
        files = _parse_llvm_dwarfdump_output(output)
        assert len(files) == 1
        assert files[0] == "orphan.cpp"

    def test_dwarf_v4_format(self):
        """DWARF v4 uses 1-based indices and mod_time/length fields."""
        output = (
            'include_directories[  1] = "/home/user/src"\n'
            'include_directories[  2] = "/home/user/include"\n'
            "file_names[  1]:\n"
            '           name: "kernel.cu"\n'
            "      dir_index: 1\n"
            "           mod_time: 0x00000000\n"
            "             length: 0x00000000\n"
            "file_names[  2]:\n"
            '           name: "types.h"\n'
            "      dir_index: 2\n"
            "           mod_time: 0x00000000\n"
            "             length: 0x00000000\n"
        )
        files = _parse_llvm_dwarfdump_output(output)
        assert len(files) == 2
        assert "/home/user/src/kernel.cu" in files
        assert "/home/user/include/types.h" in files


class TestParseReadelfOutput:
    """Tests for readelf --debug-dump=decodedline parsing."""

    def test_parse_basic_output(self):
        files = _parse_readelf_output(SAMPLE_READELF_OUTPUT)
        assert "/home/user/project/src/main.cpp" in files
        assert "/home/user/project/include/helper.h" in files
        assert "/opt/rocm/include/hip_runtime.h" in files

    def test_empty_output_returns_empty(self):
        assert _parse_readelf_output("") == []

    def test_deduplication(self):
        files = _parse_readelf_output(SAMPLE_READELF_OUTPUT)
        main_count = sum(1 for f in files if f.endswith("main.cpp"))
        assert main_count == 1

    def test_cu_header_extracted(self):
        output = "CU: /path/to/file.cpp:\n"
        files = _parse_readelf_output(output)
        assert "/path/to/file.cpp" in files


class TestPickMainFile:
    """Tests for main_file selection from DWARF-discovered user files."""

    def test_prefer_source_with_matching_name(self):
        files = [
            "/project/src/utils.cpp",
            "/project/src/my_kernel.cu",
            "/project/src/common.h",
        ]
        assert _pick_main_file(files, "my_kernel") == "/project/src/my_kernel.cu"

    def test_prefer_source_over_header(self):
        files = [
            "/project/include/kernel.h",
            "/project/src/driver.cpp",
        ]
        result = _pick_main_file(files, "something_else")
        assert result == "/project/src/driver.cpp"

    def test_prefer_source_over_matching_header(self):
        """A non-matching source is preferred over a name-matching header."""
        files = [
            "/project/src/driver.cpp",
            "/project/include/my_kernel.cuh",
        ]
        result = _pick_main_file(files, "my_kernel")
        assert result == "/project/src/driver.cpp"

    def test_fallback_to_name_matching_header(self):
        files = [
            "/project/include/my_kernel.cuh",
        ]
        result = _pick_main_file(files, "my_kernel")
        assert result == "/project/include/my_kernel.cuh"

    def test_fallback_to_first_file(self):
        files = [
            "/project/include/types.h",
            "/project/include/macros.h",
        ]
        result = _pick_main_file(files, "nonexistent")
        assert result == files[0]

    def test_hip_extension_recognized(self):
        files = [
            "/project/src/kernel.hip",
            "/project/include/common.h",
        ]
        result = _pick_main_file(files, "anything")
        assert result == "/project/src/kernel.hip"


class TestSourceDirFiltering:
    """Tests for source_dir-based filtering in DWARF discovery."""

    def test_filters_framework_headers(self):
        dwarf_files = [
            "/home/user/project/src/main.cpp",
            "/home/user/project/include/helper.h",
            "/opt/rocm/include/hip_runtime.h",
            "/home/user/kokkos/core/Kokkos_Parallel.hpp",
        ]
        source_dir = "/home/user/project"
        abs_source_dir = os.path.abspath(source_dir)

        user_files = [
            f for f in dwarf_files if os.path.abspath(f).startswith(abs_source_dir + os.sep)
        ]

        assert len(user_files) == 2
        assert "/home/user/project/src/main.cpp" in user_files
        assert "/home/user/project/include/helper.h" in user_files
        assert "/opt/rocm/include/hip_runtime.h" not in user_files
        assert "/home/user/kokkos/core/Kokkos_Parallel.hpp" not in user_files

    def test_partial_path_not_matched(self):
        """source_dir '/home/user/project' must not match '/home/user/project_extra/'."""
        dwarf_files = [
            "/home/user/project_extra/src/file.cpp",
        ]
        source_dir = "/home/user/project"
        abs_source_dir = os.path.abspath(source_dir)

        user_files = [
            f for f in dwarf_files if os.path.abspath(f).startswith(abs_source_dir + os.sep)
        ]
        assert len(user_files) == 0


class TestFindTuBySymbol:
    """Tests for TU discovery via nm symbol lookup."""

    def test_finds_matching_tu(self, tmp_path):
        obj = tmp_path / "kernel.o"
        obj.write_bytes(b"")

        cc = tmp_path / "compile_commands.json"
        cc.write_text(
            json.dumps(
                [
                    {
                        "directory": str(tmp_path),
                        "command": f"/opt/rocm/llvm/bin/clang -x hip -c kernel.cu -o {obj}",
                        "file": str(tmp_path / "kernel.cu"),
                    }
                ]
            )
        )

        nm_proc = MagicMock()
        nm_proc.returncode = 0
        nm_proc.stdout = "0000000000001000 T _Z10my_kernelPfPfi\n"

        with patch("subprocess.run", return_value=nm_proc):
            result = _find_tu_by_symbol("_Z10my_kernelPfPfi", str(cc))

        assert result is not None
        tu_path, cmd, comp_dir, found_obj = result
        assert "kernel.cu" in tu_path
        assert comp_dir == str(tmp_path)
        assert found_obj == str(obj)

    def test_returns_none_when_symbol_not_found(self, tmp_path):
        obj = tmp_path / "kernel.o"
        obj.write_bytes(b"")

        cc = tmp_path / "compile_commands.json"
        cc.write_text(
            json.dumps(
                [
                    {
                        "directory": str(tmp_path),
                        "command": f"/opt/rocm/llvm/bin/clang -x hip -c kernel.cu -o {obj}",
                        "file": str(tmp_path / "kernel.cu"),
                    }
                ]
            )
        )

        nm_proc = MagicMock()
        nm_proc.returncode = 0
        nm_proc.stdout = "                 U hipMalloc\n"

        with patch("subprocess.run", return_value=nm_proc):
            result = _find_tu_by_symbol("_Z10nonexistentPfPfi", str(cc))

        assert result is None

    def test_returns_none_when_no_compile_commands(self, tmp_path):
        result = _find_tu_by_symbol("_Z10my_kernelPfPfi", str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_handles_nm_failure(self, tmp_path):
        obj = tmp_path / "kernel.o"
        obj.write_bytes(b"")

        cc = tmp_path / "compile_commands.json"
        cc.write_text(
            json.dumps(
                [
                    {
                        "directory": str(tmp_path),
                        "command": f"/opt/rocm/llvm/bin/clang -x hip -c kernel.cu -o {obj}",
                        "file": str(tmp_path / "kernel.cu"),
                    }
                ]
            )
        )

        nm_proc = MagicMock()
        nm_proc.returncode = 1
        nm_proc.stdout = ""

        with patch("subprocess.run", return_value=nm_proc):
            result = _find_tu_by_symbol("_Z10my_kernelPfPfi", str(cc))

        assert result is None

    def test_handles_arguments_array(self, tmp_path):
        """compile_commands.json may use 'arguments' instead of 'command'."""
        obj = tmp_path / "kernel.o"
        obj.write_bytes(b"")

        cc = tmp_path / "compile_commands.json"
        cc.write_text(
            json.dumps(
                [
                    {
                        "directory": str(tmp_path),
                        "arguments": [
                            "/opt/rocm/llvm/bin/clang",
                            "-x",
                            "hip",
                            "-c",
                            "kernel.cu",
                            "-o",
                            str(obj),
                        ],
                        "file": str(tmp_path / "kernel.cu"),
                    }
                ]
            )
        )

        nm_proc = MagicMock()
        nm_proc.returncode = 0
        nm_proc.stdout = "0000000000001000 T _Z9my_kernelPf\n"

        with patch("subprocess.run", return_value=nm_proc):
            result = _find_tu_by_symbol("_Z9my_kernelPf", str(cc))

        assert result is not None
        _, cmd, _, _ = result
        assert cmd != ""


class TestDwarfFallback:
    """Tests for DWARF fallback behavior."""

    def test_fallback_no_debug_info(self, tmp_path):
        """_extract_dwarf_source_files returns None when no debug info."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""

        with patch("subprocess.run", return_value=mock_proc):
            result = _extract_dwarf_source_files(str(tmp_path / "kernel.o"))

        assert result is None

    def test_fallback_no_compile_commands(self, tmp_path):
        """Without compile_commands.json, DWARF is skipped, grep path used."""
        kernel = tmp_path / "kernel.hip"
        kernel.write_text(
            "__global__ void my_kernel(float* a, int n) {\n    int i = threadIdx.x;\n}\n"
        )

        result = find_kernel_source(
            "my_kernel",
            str(tmp_path),
            language="hip",
            mangled_name="_Z9my_kernelPfi",
        )
        assert result is not None
        assert result.kernel_function == "my_kernel"
        assert "kernel.hip" in result.main_file

    def test_fallback_tools_not_found(self, tmp_path):
        """When all DWARF tools raise FileNotFoundError, returns None."""
        with patch("subprocess.run", side_effect=FileNotFoundError("tool not found")):
            result = _extract_dwarf_source_files(str(tmp_path / "kernel.o"))

        assert result is None

    def test_fallback_subprocess_timeout(self, tmp_path):
        """When DWARF tools time out, returns None."""
        import subprocess

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="llvm-dwarfdump", timeout=30),
        ):
            result = _extract_dwarf_source_files(str(tmp_path / "kernel.o"))

        assert result is None


class TestEndToEndDwarfPath:
    """End-to-end test for the full DWARF discovery chain."""

    def test_full_dwarf_chain(self, tmp_path):
        """Mock the full chain: compile_commands + nm + dwarfdump -> KernelSource."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        kernel_file = src_dir / "my_kernel.hip"
        kernel_file.write_text("__global__ void my_kernel(float* a) {}\n")
        helper_file = src_dir / "helper.h"
        helper_file.write_text("inline void helper() {}\n")

        build_dir = tmp_path / "build"
        build_dir.mkdir()
        obj_file = build_dir / "my_kernel.o"
        obj_file.write_bytes(b"")

        cc_path = build_dir / "compile_commands.json"
        cc_path.write_text(
            json.dumps(
                [
                    {
                        "directory": str(build_dir),
                        "command": (
                            f"/opt/rocm/llvm/bin/clang -DUSE_HIP "
                            f"-I{src_dir} -x hip {kernel_file} "
                            f"-c -o {obj_file}"
                        ),
                        "file": str(kernel_file),
                    }
                ]
            )
        )

        nm_proc = MagicMock()
        nm_proc.returncode = 0
        nm_proc.stdout = "0000000000001000 T _Z9my_kernelPf\n"

        dwarfdump_output = (
            "Line table prologue:\n"
            f'include_directories[  0] = "{build_dir}"\n'
            f'include_directories[  1] = "{src_dir}"\n'
            'include_directories[  2] = "/opt/rocm/include"\n'
            "file_names[  0]:\n"
            '           name: "my_kernel.hip"\n'
            "      dir_index: 1\n"
            "file_names[  1]:\n"
            '           name: "helper.h"\n'
            "      dir_index: 1\n"
            "file_names[  2]:\n"
            '           name: "hip_runtime.h"\n'
            "      dir_index: 2\n"
        )
        dwarfdump_proc = MagicMock()
        dwarfdump_proc.returncode = 0
        dwarfdump_proc.stdout = dwarfdump_output

        def mock_run(cmd, **kwargs):
            if cmd[0] == "nm":
                return nm_proc
            if "dwarfdump" in cmd[0]:
                return dwarfdump_proc
            return MagicMock(returncode=1, stdout="")

        with patch("subprocess.run", side_effect=mock_run):
            result = _find_hip_kernel_via_dwarf(
                kernel_name="my_kernel",
                mangled_name="_Z9my_kernelPf",
                source_dir=str(src_dir),
                compile_commands_path=str(cc_path),
                extra_defines=["EXTRA_DEF"],
            )

        assert result is not None
        assert result.language == "hip"
        assert result.kernel_function == "my_kernel"
        assert "my_kernel.hip" in result.main_file
        assert len(result.source_files) == 2
        assert result.translation_unit != ""
        assert result.compile_command != ""

        define_names = [d.split("=")[0] for d in result.compile_defines]
        assert "USE_HIP" in define_names
        assert "EXTRA_DEF" in define_names

    def test_dwarf_path_filters_system_headers(self, tmp_path):
        """System/framework headers from DWARF should not appear in user_files."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        kernel_file = src_dir / "app.cpp"
        kernel_file.write_text("void app() {}\n")

        build_dir = tmp_path / "build"
        build_dir.mkdir()
        obj_file = build_dir / "app.o"
        obj_file.write_bytes(b"")

        cc_path = build_dir / "compile_commands.json"
        cc_path.write_text(
            json.dumps(
                [
                    {
                        "directory": str(build_dir),
                        "command": f"clang -x hip {kernel_file} -c -o {obj_file}",
                        "file": str(kernel_file),
                    }
                ]
            )
        )

        nm_proc = MagicMock()
        nm_proc.returncode = 0
        nm_proc.stdout = "0000000000001000 T _Z3appv\n"

        dwarfdump_output = (
            "Line table prologue:\n"
            f'include_directories[  0] = "{src_dir}"\n'
            'include_directories[  1] = "/opt/rocm/include"\n'
            'include_directories[  2] = "/home/other/kokkos/core/src"\n'
            "file_names[  0]:\n"
            '           name: "app.cpp"\n'
            "      dir_index: 0\n"
            "file_names[  1]:\n"
            '           name: "hip_runtime.h"\n'
            "      dir_index: 1\n"
            "file_names[  2]:\n"
            '           name: "Kokkos_Parallel.hpp"\n'
            "      dir_index: 2\n"
        )
        dwarfdump_proc = MagicMock()
        dwarfdump_proc.returncode = 0
        dwarfdump_proc.stdout = dwarfdump_output

        def mock_run(cmd, **kwargs):
            if cmd[0] == "nm":
                return nm_proc
            if "dwarfdump" in cmd[0]:
                return dwarfdump_proc
            return MagicMock(returncode=1, stdout="")

        with patch("subprocess.run", side_effect=mock_run):
            result = _find_hip_kernel_via_dwarf(
                kernel_name="app",
                mangled_name="_Z3appv",
                source_dir=str(src_dir),
                compile_commands_path=str(cc_path),
            )

        assert result is not None
        assert len(result.source_files) == 1
        assert "app.cpp" in result.source_files[0]
