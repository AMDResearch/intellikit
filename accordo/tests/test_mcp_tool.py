# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the Accordo MCP tool body.

``test_mcp_server.py`` covers ``main()`` and transport wiring.  This covers
``run_validate_kernel_correctness`` — the logic the MCP tool actually delegates
to — with the validator substituted so no GPU is required.
"""

from unittest.mock import MagicMock, patch

import pytest

from accordo.mcp import server as mcp_server


def _result(is_valid=True, n=3, summary="3/3 arrays matched"):
    r = MagicMock()
    r.is_valid = is_valid
    r.num_arrays_validated = n
    r.summary.return_value = summary
    return r


def _patched(result=None, side_effect=None):
    validator = MagicMock()
    validator.capture_snapshot.side_effect = ["REF_SNAP", "OPT_SNAP"]
    validator.compare_snapshots.return_value = result if result is not None else _result()
    cls = MagicMock(return_value=validator)
    if side_effect is not None:
        cls.side_effect = side_effect
    return cls, validator


class TestRunValidateKernelCorrectness:
    def test_returns_the_three_documented_keys(self):
        cls, _ = _patched()
        with patch.object(mcp_server, "Accordo", cls):
            out = mcp_server.run_validate_kernel_correctness("k", ["./ref"], ["./opt"])
        assert set(out) == {"is_valid", "num_arrays_validated", "summary"}

    def test_values_come_from_the_comparison_result(self):
        cls, _ = _patched(result=_result(is_valid=False, n=5, summary="2/5 mismatched"))
        with patch.object(mcp_server, "Accordo", cls):
            out = mcp_server.run_validate_kernel_correctness("k", ["./ref"], ["./opt"])
        assert out == {
            "is_valid": False,
            "num_arrays_validated": 5,
            "summary": "2/5 mismatched",
        }

    def test_validator_constructed_with_reference_command(self):
        cls, _ = _patched()
        with patch.object(mcp_server, "Accordo", cls):
            mcp_server.run_validate_kernel_correctness("reduce", ["./ref", "--n", "8"], ["./opt"])
        kwargs = cls.call_args.kwargs
        assert kwargs["binary"] == ["./ref", "--n", "8"]
        assert kwargs["kernel_name"] == "reduce"
        assert kwargs["kernel_args"] is None
        assert kwargs["working_directory"] == "."

    def test_working_directory_forwarded(self):
        cls, _ = _patched()
        with patch.object(mcp_server, "Accordo", cls):
            mcp_server.run_validate_kernel_correctness(
                "k", ["./ref"], ["./opt"], working_directory="/build"
            )
        assert cls.call_args.kwargs["working_directory"] == "/build"

    def test_reference_then_optimized_are_both_captured(self):
        cls, validator = _patched()
        with patch.object(mcp_server, "Accordo", cls):
            mcp_server.run_validate_kernel_correctness("k", ["./ref"], ["./opt"])
        captured = [c.kwargs["binary"] for c in validator.capture_snapshot.call_args_list]
        assert captured == [["./ref"], ["./opt"]]

    def test_snapshots_passed_to_compare_in_order(self):
        cls, validator = _patched()
        with patch.object(mcp_server, "Accordo", cls):
            mcp_server.run_validate_kernel_correctness("k", ["./ref"], ["./opt"])
        assert validator.compare_snapshots.call_args.args == ("REF_SNAP", "OPT_SNAP")

    def test_tolerance_defaults(self):
        cls, validator = _patched()
        with patch.object(mcp_server, "Accordo", cls):
            mcp_server.run_validate_kernel_correctness("k", ["./ref"], ["./opt"])
        kwargs = validator.compare_snapshots.call_args.kwargs
        assert kwargs["tolerance"] is None
        assert kwargs["atol"] == 1e-08
        assert kwargs["rtol"] == 1e-05
        assert kwargs["equal_nan"] is False

    def test_tolerance_overrides_forwarded(self):
        cls, validator = _patched()
        with patch.object(mcp_server, "Accordo", cls):
            mcp_server.run_validate_kernel_correctness(
                "k", ["./ref"], ["./opt"], tolerance=1e-3, atol=1e-9, rtol=1e-4, equal_nan=True
            )
        kwargs = validator.compare_snapshots.call_args.kwargs
        assert kwargs["tolerance"] == 1e-3
        assert kwargs["atol"] == 1e-9
        assert kwargs["rtol"] == 1e-4
        assert kwargs["equal_nan"] is True

    def test_validator_errors_propagate(self):
        from accordo.exceptions import AccordoError

        cls, _ = _patched(side_effect=AccordoError("kernel never dispatched"))
        with patch.object(mcp_server, "Accordo", cls):
            with pytest.raises(AccordoError, match="kernel never dispatched"):
                mcp_server.run_validate_kernel_correctness("k", ["./ref"], ["./opt"])


class TestToolWrapper:
    def _tool(self):
        return getattr(
            mcp_server.validate_kernel_correctness, "fn", mcp_server.validate_kernel_correctness
        )

    def test_tool_delegates_to_the_plain_function(self):
        with patch.object(
            mcp_server, "run_validate_kernel_correctness", return_value={"is_valid": True}
        ) as run:
            out = self._tool()("k", ["./ref"], ["./opt"])
        assert out == {"is_valid": True}
        assert run.call_args.kwargs["kernel_name"] == "k"

    def test_tool_forwards_every_tolerance_argument(self):
        with patch.object(mcp_server, "run_validate_kernel_correctness", return_value={}) as run:
            self._tool()(
                "k",
                ["./ref"],
                ["./opt"],
                tolerance=1e-2,
                atol=1e-7,
                rtol=1e-3,
                equal_nan=True,
                working_directory="/w",
            )
        kwargs = run.call_args.kwargs
        assert kwargs["tolerance"] == 1e-2
        assert kwargs["atol"] == 1e-7
        assert kwargs["rtol"] == 1e-3
        assert kwargs["equal_nan"] is True
        assert kwargs["working_directory"] == "/w"

    def test_server_name(self):
        assert mcp_server.mcp.name == "IntelliKit Accordo"
