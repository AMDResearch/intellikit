# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Linex tests: inline HIP kernel, compile (with or without -g), profile with Linex.
- With -g: source_lines and instructions populated; SourceLine and InstructionData structure.
- Without -g: instructions (ISA + cycles) still populated; source_lines may be empty.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from linex import Linex

VECTOR_ADD_HIP = """
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vector_add(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 65536;
    size_t bytes = N * sizeof(float);
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }
    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(vector_add, dim3((N + 255) / 256), dim3(256), 0, 0, d_a, d_b, d_c, N);
    hipDeviceSynchronize();
    free(h_a);
    free(h_b);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    return 0;
}
"""


def _compile_hip(kernel_code: str, name: str, tmp_dir: Path, debug: bool = True) -> Path:
    """Write HIP source, compile with hipcc; debug=True adds -g for source-line mapping."""
    src = tmp_dir / f"{name}.hip"
    bin_path = tmp_dir / name
    src.write_text(kernel_code)
    cmd = ["hipcc", str(src), "-o", str(bin_path), "-O2"]
    if debug:
        cmd.append("-g")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_dir, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"hipcc failed for {name}:\n{r.stderr}")
    return bin_path


@pytest.mark.parametrize("kernel_filter", ["vector_add", None], ids=["filter", "no_filter"])
def test_profile_vector_add_source_lines(kernel_filter):
    """Profile inline vector_add; instructions always populated; source_lines when built with -g."""
    with tempfile.TemporaryDirectory(prefix="linex_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path, debug=True)
        profiler = Linex()
        profiler.profile(
            command=str(bin_path),
            output_dir=str(tmp_path / "sqtt_out"),
            kernel_filter=kernel_filter,
        )
    assert len(profiler.instructions) >= 1
    if len(profiler.source_lines) >= 1:
        user_lines = [
            s
            for s in profiler.source_lines
            if "vector_add" in s.file or "vector_add" in s.source_location
        ]
        if kernel_filter:
            assert len(user_lines) >= 1, (
                f"Expected source lines for vector_add: {[s.source_location for s in profiler.source_lines]}"
            )


def test_profile_source_line_attributes():
    """SourceLine has file, line_number, source_location, cycles, instructions, stall_percent."""
    with tempfile.TemporaryDirectory(prefix="linex_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path)
        profiler = Linex()
        profiler.profile(
            command=str(bin_path),
            output_dir=str(tmp_path / "sqtt_out"),
            kernel_filter="vector_add",
        )
    assert len(profiler.source_lines) >= 1
    line = profiler.source_lines[0]
    assert hasattr(line, "file")
    assert hasattr(line, "line_number")
    assert hasattr(line, "source_location")
    assert hasattr(line, "execution_count")
    assert hasattr(line, "total_cycles")
    assert hasattr(line, "stall_cycles")
    assert hasattr(line, "idle_cycles")
    assert hasattr(line, "instructions")
    assert isinstance(line.instructions, list)
    assert hasattr(line, "stall_percent")
    assert isinstance(line.stall_percent, (int, float))


def test_profile_instruction_data_attributes():
    """InstructionData has isa, source_location, cycles, file, line, stall_percent."""
    with tempfile.TemporaryDirectory(prefix="linex_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path, debug=True)
        profiler = Linex()
        profiler.profile(
            command=str(bin_path),
            output_dir=str(tmp_path / "sqtt_out"),
            kernel_filter="vector_add",
        )
    assert len(profiler.instructions) >= 1
    inst = profiler.instructions[0]
    assert hasattr(inst, "isa")
    assert hasattr(inst, "instruction_index")
    assert hasattr(inst, "source_location")
    assert hasattr(inst, "execution_count")
    assert hasattr(inst, "latency_cycles")
    assert hasattr(inst, "stall_cycles")
    assert hasattr(inst, "idle_cycles")
    assert hasattr(inst, "file")
    assert hasattr(inst, "line")
    assert hasattr(inst, "stall_percent")


def test_profile_without_debug_symbols_assembly_only():
    """Without -g, instructions (ISA + cycles) are still populated; source_lines may be empty."""
    with tempfile.TemporaryDirectory(prefix="linex_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path, debug=False)
        profiler = Linex()
        profiler.profile(
            command=str(bin_path),
            output_dir=str(tmp_path / "sqtt_out"),
            kernel_filter="vector_add",
        )
    assert len(profiler.instructions) >= 1
    inst = profiler.instructions[0]
    assert inst.isa
    assert hasattr(inst, "latency_cycles")
    assert hasattr(inst, "stall_cycles")


def test_profile_with_output_dir():
    """profile(output_dir=path) writes ui_output_* and loads trace."""
    with tempfile.TemporaryDirectory(prefix="linex_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path)
        out_dir = tmp_path / "sqtt_out"
        out_dir.mkdir()
        profiler = Linex()
        profiler.profile(
            command=str(bin_path),
            output_dir=str(out_dir),
            kernel_filter="vector_add",
        )
        assert len(profiler.source_lines) >= 1
        ui_dirs = list(out_dir.glob("ui_output_*"))
        assert len(ui_dirs) >= 1


@pytest.mark.parametrize("force_cu_mask", [True, False], ids=["cu_mask", "no_cu_mask"])
def test_profile_force_cu_mask(force_cu_mask):
    """profile(force_cu_mask=...) runs; with mask often gives more deterministic SQTT."""
    with tempfile.TemporaryDirectory(prefix="linex_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_path = _compile_hip(VECTOR_ADD_HIP, "vector_add", tmp_path)
        profiler = Linex()
        profiler.profile(
            command=str(bin_path),
            output_dir=str(tmp_path / "sqtt_out"),
            kernel_filter="vector_add",
            force_cu_mask=force_cu_mask,
        )
    assert len(profiler.instructions) >= 1 or len(profiler.source_lines) >= 1
