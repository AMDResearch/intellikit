# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the ELF surgery patcher."""

import pytest

from proboscis.patcher import (
    PROBOSCIS_HIDDEN_ARG,
    HiddenAbiPlan,
    PatchConfig,
    align_up,
    compute_explicit_args_length,
    mutate_kernel_metadata,
    plan_hidden_abi,
)


class TestAlignUp:
    def test_already_aligned(self):
        assert align_up(16, 8) == 16

    def test_needs_alignment(self):
        assert align_up(17, 8) == 24

    def test_zero(self):
        assert align_up(0, 8) == 0

    def test_alignment_1(self):
        assert align_up(17, 1) == 17

    def test_invalid_alignment(self):
        with pytest.raises(ValueError):
            align_up(10, 0)


class TestComputeExplicitArgsLength:
    def test_simple_args(self):
        args = [
            {".offset": 0, ".size": 8, ".value_kind": "by_value"},
            {".offset": 8, ".size": 8, ".value_kind": "by_value"},
            {".offset": 16, ".size": 4, ".value_kind": "by_value"},
        ]
        assert compute_explicit_args_length(args) == 20

    def test_excludes_hidden_args(self):
        args = [
            {".offset": 0, ".size": 8, ".value_kind": "by_value"},
            {".offset": 8, ".size": 8, ".value_kind": "hidden_global_offset_x"},
        ]
        assert compute_explicit_args_length(args) == 8

    def test_empty(self):
        assert compute_explicit_args_length([]) == 0


class TestPlanHiddenAbi:
    def test_basic_plan(self):
        kernel = {
            ".name": "vec_add",
            ".symbol": "vec_add.kd",
            ".kernarg_segment_size": 32,
            ".args": [
                {".offset": 0, ".size": 8, ".value_kind": "global_buffer"},
                {".offset": 8, ".size": 8, ".value_kind": "global_buffer"},
                {".offset": 16, ".size": 8, ".value_kind": "global_buffer"},
                {".offset": 24, ".size": 4, ".value_kind": "by_value"},
            ],
        }
        plan = plan_hidden_abi(kernel)
        assert plan.kernel_name == "vec_add"
        assert plan.source_kernarg_size == 32
        assert plan.insertion_offset == 32  # aligned after existing args
        assert plan.new_kernarg_size == 40  # 32 + 8 (pointer)
        assert plan.pointer_size == 8

    def test_with_hidden_args(self):
        kernel = {
            ".name": "kern",
            ".kernarg_segment_size": 48,
            ".args": [
                {".offset": 0, ".size": 8, ".value_kind": "global_buffer"},
                {".offset": 8, ".size": 8, ".value_kind": "hidden_global_offset_x"},
                {".offset": 16, ".size": 8, ".value_kind": "hidden_global_offset_y"},
                {".offset": 24, ".size": 8, ".value_kind": "hidden_global_offset_z"},
            ],
        }
        plan = plan_hidden_abi(kernel)
        assert plan.insertion_offset >= 48  # after all args including hidden
        assert plan.new_kernarg_size == plan.insertion_offset + 8


class TestMutateKernelMetadata:
    def test_adds_hidden_arg(self):
        kernel = {
            ".name": "vec_add",
            ".kernarg_segment_size": 32,
            ".args": [
                {".offset": 0, ".size": 8, ".value_kind": "global_buffer"},
                {".offset": 8, ".size": 4, ".value_kind": "by_value"},
            ],
        }
        plan = plan_hidden_abi(kernel)
        mutated = mutate_kernel_metadata(kernel, plan)

        assert mutated[".kernarg_segment_size"] == plan.new_kernarg_size
        args = mutated[".args"]
        probe_args = [a for a in args if a.get(".name") == PROBOSCIS_HIDDEN_ARG]
        assert len(probe_args) == 1
        assert probe_args[0][".offset"] == plan.insertion_offset
        assert probe_args[0][".size"] == 8

    def test_idempotent(self):
        kernel = {
            ".name": "vec_add",
            ".kernarg_segment_size": 32,
            ".args": [
                {".offset": 0, ".size": 8, ".value_kind": "global_buffer"},
            ],
        }
        plan = plan_hidden_abi(kernel)
        mutated = mutate_kernel_metadata(kernel, plan)
        # Mutate again — should not add a second hidden arg
        plan2 = plan_hidden_abi(mutated)
        mutated2 = mutate_kernel_metadata(mutated, plan2)
        probe_args = [a for a in mutated2[".args"] if a.get(".name") == PROBOSCIS_HIDDEN_ARG]
        assert len(probe_args) == 1
