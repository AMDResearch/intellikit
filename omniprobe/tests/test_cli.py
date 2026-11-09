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
        """Omniprobe dbb53ed's -d grammar passes through unchanged (validated by Omniprobe)."""
        parser = create_parser()
        for spec in ("all", "1", "2,5", "1-10", "10-", "1,3,4:3:100", "1::100"):
            args = parser.parse_args(["memory", "--dispatches", spec, "./app"])
            assert args.dispatches == spec

    def test_malformed_dispatches_exit_2(self, monkeypatch, capsys):
        """Omniprobe's DispatchSpecError (a ValueError) is a usage error: exit 2, message shown."""
        import omniprobe_ik.cli.main as cli

        def bad(args):
            raise ValueError("invalid dispatch spec '2,x': expected a number")

        monkeypatch.setattr(cli, "memory_command", bad)
        monkeypatch.setattr(cli.sys, "argv", ["omniprobe-ik", "memory", "-d", "2,x", "./app"])
        assert cli.main() == 2
        assert "invalid dispatch spec '2,x'" in capsys.readouterr().err

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

    def test_application_arguments_follow_the_target(self):
        from omniprobe_ik.cli.main import _command

        parser = create_parser()
        args = parser.parse_args(
            ["memory", "--kernel", "my_kernel", "--output", "x.json", "./app", "input.dat"]
        )
        assert args.kernel == "my_kernel" and args.output == "x.json"
        assert _command(args) == "./app input.dat"

    def test_double_dash_passes_options_to_the_application(self):
        from omniprobe_ik.cli.main import _command

        args = create_parser().parse_args(["memory", "-k", "k", "--", "./app", "--size", "1024"])
        assert args.kernel == "k"
        assert _command(args) == "./app --size 1024"

    def test_quoted_command_still_works(self):
        from omniprobe_ik.cli.main import _command

        args = create_parser().parse_args(["basic-blocks", "./app input.dat"])
        assert _command(args) == "./app input.dat"
