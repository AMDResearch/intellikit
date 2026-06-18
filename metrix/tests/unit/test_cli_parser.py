"""Unit tests for Metrix CLI argument parsing."""

import pytest

from metrix.cli.main import create_parser


class TestKernelIterationCLI:
    def test_kernel_iteration_and_kernel(self):
        parser = create_parser()
        args = parser.parse_args(
            ["profile", "--kernel-iteration", "5", "--kernel", "foo", "./my_app"]
        )
        assert args.command == "profile"
        assert args.kernel_iteration == 5
        assert args.kernel == "foo"
        assert args.target == "./my_app"

    def test_kernel_iteration_range(self):
        parser = create_parser()
        args = parser.parse_args(
            ["profile", "--kernel-iteration-range", "[10,10]", "-k", "bar.*", "./app"]
        )
        assert args.kernel_iteration_range == "[10,10]"
        assert args.kernel == "bar.*"

    def test_kernel_iteration_mutually_exclusive_with_range(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "profile",
                    "--kernel-iteration",
                    "1",
                    "--kernel-iteration-range",
                    "[2,2]",
                    "./app",
                ]
            )
