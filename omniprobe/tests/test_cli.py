# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the Omniprobe CLI argument parsing."""

from omniprobe_ik.cli.main import create_parser


class TestCLIParser:
    """Test CLI argument parser structure."""

    def test_memory_subcommand(self):
        parser = create_parser()
        args = parser.parse_args(["memory", "./app"])
        assert args.command == "memory"
        assert args.target == "./app"
        assert args.kernel is None
        assert args.dispatches == "all"
        assert args.output is None

    def test_basic_blocks_subcommand(self):
        parser = create_parser()
        args = parser.parse_args(["basic-blocks", "./app"])
        assert args.command == "basic-blocks"
        assert args.target == "./app"

    def test_kernel_filter(self):
        parser = create_parser()
        args = parser.parse_args(["memory", "--kernel", "matmul.*", "./app"])
        assert args.kernel == "matmul.*"

    def test_dispatches(self):
        parser = create_parser()
        args = parser.parse_args(["memory", "--dispatches", "random", "./app"])
        assert args.dispatches == "random"

    def test_output_flag(self):
        parser = create_parser()
        args = parser.parse_args(["memory", "-o", "out.json", "./app"])
        assert args.output == "out.json"

    def test_flags_subcommand(self):
        parser = create_parser()
        args = parser.parse_args(["flags", "basic-blocks", "--offload-arch", "gfx942"])
        assert args.command == "flags"
        assert args.analysis == "basic-blocks"
        assert args.offload_arch == "gfx942"
