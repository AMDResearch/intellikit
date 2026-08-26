# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the Snapshot dataclass."""

import numpy as np

from accordo.snapshot import Snapshot


def _snap(**over):
    base = dict(
        arrays=[np.zeros(4, dtype=np.float32), np.ones(2, dtype=np.int32)],
        execution_time_ms=12.345,
        binary=["./my_app"],
        working_directory="/tmp/proj",
    )
    base.update(over)
    return Snapshot(**base)


class TestConstruction:
    def test_optional_fields_default_to_none(self):
        s = _snap()
        assert s.grid_size is None
        assert s.block_size is None
        assert s.dispatch_arrays is None

    def test_fields_round_trip(self):
        s = _snap()
        assert len(s.arrays) == 2
        assert s.execution_time_ms == 12.345
        assert s.binary == ["./my_app"]
        assert s.working_directory == "/tmp/proj"


class TestRepr:
    def test_includes_binary_array_count_and_time(self):
        r = repr(_snap())
        assert "./my_app" in r
        assert "arrays=2" in r
        assert "12.35" in r  # formatted to 2dp

    def test_joins_multi_token_binary(self):
        assert "python run.py" in repr(_snap(binary=["python", "run.py"]))

    def test_handles_no_arrays(self):
        assert "arrays=0" in repr(_snap(arrays=[]))


class TestSummary:
    def test_core_lines_present(self):
        out = _snap().summary()
        assert "Snapshot Summary:" in out
        assert "Binary: ./my_app" in out
        assert "Working Directory: /tmp/proj" in out
        assert "Execution Time: 12.35ms" in out
        assert "Number of Arrays: 2" in out

    def test_lists_every_array_with_shape_and_dtype(self):
        out = _snap().summary()
        assert "Array 0: shape=(4,), dtype=float32" in out
        assert "Array 1: shape=(2,), dtype=int32" in out

    def test_optional_sections_absent_by_default(self):
        out = _snap().summary()
        assert "Grid Size" not in out
        assert "Block Size" not in out
        assert "Number of Dispatches" not in out

    def test_grid_size_rendered(self):
        out = _snap(grid_size={"x": 8, "y": 2, "z": 1}).summary()
        assert "Grid Size: x=8, y=2, z=1" in out

    def test_block_size_rendered(self):
        out = _snap(block_size={"x": 256, "y": 1, "z": 1}).summary()
        assert "Block Size: x=256, y=1, z=1" in out

    def test_missing_dimension_keys_render_as_none(self):
        out = _snap(grid_size={"x": 4}).summary()
        assert "Grid Size: x=4, y=None, z=None" in out

    def test_dispatch_count_rendered(self):
        out = _snap(dispatch_arrays=[[np.zeros(1)], [np.zeros(1)], [np.zeros(1)]]).summary()
        assert "Number of Dispatches: 3" in out

    def test_empty_dispatch_list_still_reported(self):
        assert "Number of Dispatches: 0" in _snap(dispatch_arrays=[]).summary()

    def test_no_arrays_produces_no_array_lines(self):
        out = _snap(arrays=[]).summary()
        assert "Number of Arrays: 0" in out
        assert "Array 0" not in out
