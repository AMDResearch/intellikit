# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import pytest

from rocm_mcp import HipCompiler


def test_hip_compiler_minimal(tmp_path: Path) -> None:
    """Test HIP compilation only with source."""
    source_file = "tests/compile/vectoradd_hip.cpp"
    output_file = tmp_path / "vectoradd_hip.out"
    compiler = HipCompiler()
    result = compiler.compile(source_file=source_file, output_file=output_file)
    assert result.success
    assert result.errors is None


def test_hip_compiler_no_source(tmp_path: Path) -> None:
    """Test HIP compilation with no source."""
    output_file = tmp_path / "vectoradd_hip.out"
    compiler = HipCompiler()
    with pytest.raises(ValueError, match="provided"):
        compiler.compile(source_file=None, output_file=output_file)


def test_hip_compiler_unknown_source(tmp_path: Path) -> None:
    """Test HIP compilation with unknown source."""
    source_file = "tests/compile/doesntexist.cpp"
    output_file = tmp_path / "vectoradd_hip.out"
    compiler = HipCompiler()
    with pytest.raises(FileNotFoundError, match="not exist"):
        compiler.compile(source_file=source_file, output_file=output_file)


def test_hip_compiler_no_output() -> None:
    """Test HIP compilation with missing output file."""
    source_file = "tests/compile/vectoradd_hip.cpp"
    compiler = HipCompiler()
    with pytest.raises(ValueError, match="provided"):
        compiler.compile(source_file=source_file, output_file=None)


def test_hip_compiler_flags(tmp_path: Path) -> None:
    """Test HIP compilation with flags."""
    source_file = "tests/compile/vectoradd_hip.cpp"
    output_file = tmp_path / "vectoradd_hip.out"
    compiler = HipCompiler()
    result = compiler.compile(
        source_file=source_file,
        output_file=output_file,
        extra_flags=["-O2", "-DDEBUG"],
    )
    assert result.success
    assert result.errors is None


def test_hip_compiler_include_dirs(tmp_path: Path) -> None:
    """Test HIP compilation with include dirs."""
    source_file = "tests/compile/vectoradd_hip.cpp"
    output_file = tmp_path / "vectoradd_hip.out"
    compiler = HipCompiler()
    result = compiler.compile(
        source_file=source_file,
        output_file=output_file,
        include_dirs=["/usr/include", "/usr/local/include"],
    )
    assert result.success
    assert result.errors is None


def test_hip_compiler_libraries(tmp_path: Path) -> None:
    """Test HIP compilation with libraries."""
    source_file = "tests/compile/vectoradd_hip.cpp"
    output_file = tmp_path / "vectoradd_hip.out"
    compiler = HipCompiler()
    result = compiler.compile(
        source_file=source_file,
        output_file=output_file,
        libraries=["m"],
        library_dirs=["/usr/lib", "/usr/local/lib"],
    )
    assert result.success
    assert result.errors is None
