# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the Metrix CLI.

These exercise the CLI end-to-end without a GPU: the backend is replaced by a
fake that returns real ``Statistics`` dataclasses, so a renamed field breaks
these tests rather than silently passing the way a ``MagicMock`` would.
"""

from __future__ import annotations

import csv
import json
from unittest.mock import patch

import pytest

from metrix.backends import Statistics
from metrix.cli.info_cmd import info_command, show_metric_info, show_profile_info
from metrix.cli.list_cmd import (
    list_command,
    list_devices,
    list_metrics,
    list_profiles,
)
from metrix.cli.main import create_parser, main
from metrix.cli.profile_cmd import (
    _write_csv_output,
    _write_json_output,
    _write_text_output,
    profile_command,
)
from metrix.metrics import METRIC_CATALOG, METRIC_PROFILES
from .conftest import DEFAULT_FAKE_METRIC, FakeBackend, fake_stats

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

KNOWN_METRIC = DEFAULT_FAKE_METRIC
_stats = fake_stats


class Args:
    """argparse.Namespace stand-in with the profile defaults filled in."""

    def __init__(self, **kw):
        defaults = {
            "target": "./app",
            "profile": None,
            "metrics": None,
            "time_only": False,
            "kernel": None,
            "num_replays": 1,
            "aggregate": False,
            "top": None,
            "output": None,
            "no_counters": False,
        }
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


@pytest.fixture
def patched_backend():
    """Patch arch detection + backend lookup, yielding the FakeBackend."""
    backend = FakeBackend()
    with (
        patch("metrix.cli.profile_cmd.detect_or_default", return_value="gfx942"),
        patch("metrix.cli.profile_cmd.get_backend", return_value=backend),
    ):
        yield backend


# --------------------------------------------------------------------------
# profile_cmd — metric selection
# --------------------------------------------------------------------------


def test_profile_time_only_collects_no_metrics(patched_backend):
    assert profile_command(Args(time_only=True)) == 0
    assert patched_backend.profile_calls[0]["metrics"] == []


def test_profile_explicit_metrics_are_split_and_stripped(patched_backend):
    patched_backend._available = [KNOWN_METRIC, "memory.l2_hit_rate"]
    profile_command(Args(metrics=f" {KNOWN_METRIC} , memory.l2_hit_rate "))
    assert patched_backend.profile_calls[0]["metrics"] == [
        KNOWN_METRIC,
        "memory.l2_hit_rate",
    ]


def test_profile_default_uses_all_backend_metrics(patched_backend):
    profile_command(Args())
    assert patched_backend.profile_calls[0]["metrics"] == patched_backend.get_available_metrics()


def test_profile_named_profile_uses_catalog_metrics(patched_backend):
    """A profile collects the metrics it declares that this backend can compute.

    Metric coverage varies by architecture, so the profile is narrowed to the
    intersection rather than passed through verbatim.
    """
    # 'quick' declares hbm_bandwidth_utilization and l2_hit_rate; the fake
    # backend offers only the former, so only the former may be collected.
    assert METRIC_PROFILES["quick"]["metrics"] == [KNOWN_METRIC, "memory.l2_hit_rate"]
    assert patched_backend.get_available_metrics() == [KNOWN_METRIC]

    profile_command(Args(profile="quick"))
    assert patched_backend.profile_calls[0]["metrics"] == [KNOWN_METRIC]


def test_profile_unknown_profile_returns_1(patched_backend):
    assert profile_command(Args(profile="does-not-exist")) == 1
    assert patched_backend.profile_calls == []


def test_profile_with_no_available_metrics_returns_1(patched_backend):
    """A profile whose every metric is missing must fail, not silently time the run."""
    patched_backend._available = []
    assert profile_command(Args(profile="quick")) == 1
    assert patched_backend.profile_calls == []


def test_profile_keeps_unsupported_metrics_so_their_reason_is_reported(patched_backend, caplog):
    """An unsupported metric must surface its reason, not a bare 'not available'.

    get_available_metrics() omits metrics flagged unsupported, so narrowing a
    profile to that list alone would replace the actionable reason with a
    generic message.
    """
    reason = "counter is broken on this part"
    patched_backend._unsupported_metrics = {"memory.l2_hit_rate": reason}

    profile_command(Args(profile="quick"))

    assert reason in caplog.text
    # ... and it is still excluded from what actually gets collected.
    assert patched_backend.profile_calls[0]["metrics"] == [KNOWN_METRIC]


def test_explicit_unavailable_metric_returns_1_without_traceback(patched_backend):
    """--metrics with a real catalog metric this arch lacks must fail cleanly."""
    assert profile_command(Args(metrics="memory.l2_hit_rate")) == 1
    assert patched_backend.profile_calls == []


# --------------------------------------------------------------------------
# profile_cmd — unsupported-metric handling (the two branches differ)
# --------------------------------------------------------------------------


def test_explicitly_requested_unsupported_metric_is_an_error():
    backend = FakeBackend(unsupported={KNOWN_METRIC: "no such counter"})
    with (
        patch("metrix.cli.profile_cmd.detect_or_default", return_value="gfx942"),
        patch("metrix.cli.profile_cmd.get_backend", return_value=backend),
    ):
        # --metrics is an explicit request, so this must fail rather than skip.
        assert profile_command(Args(metrics=KNOWN_METRIC)) == 1
        assert backend.profile_calls == []


def test_unsupported_metric_from_profile_is_filtered_not_fatal():
    backend = FakeBackend(
        unsupported={KNOWN_METRIC: "no such counter"},
        available=[KNOWN_METRIC, "memory.l2_hit_rate"],
    )
    with (
        patch("metrix.cli.profile_cmd.detect_or_default", return_value="gfx942"),
        patch("metrix.cli.profile_cmd.get_backend", return_value=backend),
    ):
        assert profile_command(Args()) == 0
        assert KNOWN_METRIC not in backend.profile_calls[0]["metrics"]


# --------------------------------------------------------------------------
# profile_cmd — kernels, filtering, top-K
# --------------------------------------------------------------------------


def test_no_dispatches_returns_1():
    backend = FakeBackend(dispatch_keys=[])
    with (
        patch("metrix.cli.profile_cmd.detect_or_default", return_value="gfx942"),
        patch("metrix.cli.profile_cmd.get_backend", return_value=backend),
    ):
        assert profile_command(Args()) == 1


def test_no_dispatches_with_filter_returns_1():
    backend = FakeBackend(dispatch_keys=[])
    with (
        patch("metrix.cli.profile_cmd.detect_or_default", return_value="gfx942"),
        patch("metrix.cli.profile_cmd.get_backend", return_value=backend),
    ):
        assert profile_command(Args(kernel="nomatch.*")) == 1


def test_kernel_filter_is_passed_through(patched_backend):
    profile_command(Args(kernel="gemm.*"))
    assert patched_backend.profile_calls[0]["kernel_filter"] == "gemm.*"


def test_absent_kernel_filter_is_none(patched_backend):
    profile_command(Args())
    assert patched_backend.profile_calls[0]["kernel_filter"] is None


def test_top_k_keeps_only_the_slowest(capsys):
    # Durations are 100/150/200us, so top=1 must keep only the 200us one.
    # Keys deliberately contain no ':' so they print verbatim.
    backend = FakeBackend(dispatch_keys=["alpha", "beta", "gamma"])
    with (
        patch("metrix.cli.profile_cmd.detect_or_default", return_value="gfx942"),
        patch("metrix.cli.profile_cmd.get_backend", return_value=backend),
    ):
        assert profile_command(Args(top=1)) == 0
    out = capsys.readouterr().out
    assert "gamma" in out
    assert "alpha" not in out
    assert "beta" not in out


def test_replays_and_aggregate_are_forwarded(patched_backend):
    profile_command(Args(num_replays=5, aggregate=True))
    call = patched_backend.profile_calls[0]
    assert call["num_replays"] == 5
    assert call["aggregate_by_kernel"] is True


def test_backend_profile_exception_returns_1(patched_backend):
    with patch.object(patched_backend, "profile", side_effect=RuntimeError("boom")):
        assert profile_command(Args()) == 1


def test_metric_computation_failure_is_warned_not_fatal(patched_backend):
    with patch.object(
        patched_backend, "compute_metric_stats", side_effect=ValueError("bad counter")
    ):
        # A single metric failing must not sink the whole run.
        assert profile_command(Args()) == 0


# --------------------------------------------------------------------------
# profile_cmd — output formats
# --------------------------------------------------------------------------


def test_output_json_roundtrips(tmp_path, patched_backend):
    out = tmp_path / "r.json"
    assert profile_command(Args(output=str(out))) == 0
    data = json.loads(out.read_text())
    key = "dispatch_1:gemm_kernel"
    assert data[key]["duration_us"]["avg"] == 100.0
    assert data[key]["metrics"][KNOWN_METRIC]["unit"] == "%"


def test_output_csv_has_header_and_one_row_per_dispatch(tmp_path, patched_backend):
    out = tmp_path / "r.csv"
    assert profile_command(Args(output=str(out))) == 0
    rows = list(csv.reader(out.read_text().splitlines()))
    assert rows[0][0] == "dispatch_key"
    assert len(rows) == 2


def test_unknown_extension_falls_back_to_text(tmp_path, patched_backend):
    out = tmp_path / "r.weird"
    assert profile_command(Args(output=str(out))) == 0
    assert "gemm_kernel" in out.read_text()


def test_json_writer_handles_missing_duration(tmp_path):
    results = {"k": {"duration_us": None, "metrics": {KNOWN_METRIC: _stats()}}}
    out = tmp_path / "r.json"
    _write_json_output(out, results, [KNOWN_METRIC])
    assert json.loads(out.read_text())["k"]["duration_us"] is None


def test_csv_writer_zero_fills_missing_values(tmp_path):
    results = {"k": {"duration_us": None, "metrics": {}}}
    out = tmp_path / "r.csv"
    _write_csv_output(out, results, [KNOWN_METRIC], aggregated=False)
    row = list(csv.reader(out.read_text().splitlines()))[1]
    assert row[1:4] == ["0", "0", "0"]


def test_text_writer_restores_stdout(tmp_path, capsys):
    results = {"k": {"duration_us": _stats(unit="us"), "metrics": {}}}
    _write_text_output(tmp_path / "r.txt", results, [], aggregated=True)
    # If sys.stdout were left pointing at the buffer this would not be captured.
    print("still working")
    assert "still working" in capsys.readouterr().out


def test_aggregated_output_prints_kernel_not_dispatch(capsys, patched_backend):
    profile_command(Args(aggregate=True))
    assert "Kernel: dispatch_1:gemm_kernel" in capsys.readouterr().out


def test_unaggregated_output_splits_dispatch_id(capsys, patched_backend):
    profile_command(Args())
    assert "Dispatch #1: gemm_kernel" in capsys.readouterr().out


def test_dispatch_key_without_colon_prints_as_kernel(capsys):
    backend = FakeBackend(dispatch_keys=["bare_name"])
    with (
        patch("metrix.cli.profile_cmd.detect_or_default", return_value="gfx942"),
        patch("metrix.cli.profile_cmd.get_backend", return_value=backend),
    ):
        profile_command(Args())
    assert "Kernel: bare_name" in capsys.readouterr().out


# --------------------------------------------------------------------------
# list_cmd
# --------------------------------------------------------------------------


def test_list_metrics_prints_every_catalog_entry(capsys):
    list_metrics()
    out = capsys.readouterr().out
    assert f"Total: {len(METRIC_CATALOG)} metrics" in out


def test_list_metrics_by_category_is_a_subset(capsys):
    category = METRIC_CATALOG[KNOWN_METRIC]["category"].value
    list_metrics(category)
    out = capsys.readouterr().out
    assert f"Category: {category}" in out


def test_list_profiles_prints_each_profile(capsys):
    list_profiles()
    out = capsys.readouterr().out
    for name in METRIC_PROFILES:
        assert name.upper() in out


def test_list_devices_discovers_backends_from_disk(capsys):
    list_devices()
    out = capsys.readouterr().out
    # gfx942 has a backend module, so discovery must surface it.
    assert "gfx942" in out
    assert "AMD Instinct MI300" in out


def test_list_devices_handles_no_backend_modules(capsys, tmp_path):
    class _EmptyDir:
        def glob(self, _pat):
            return iter(())

        def __truediv__(self, _other):
            return self

    with patch("pathlib.Path.glob", return_value=iter(())):
        list_devices()
    assert "no backend modules found" in capsys.readouterr().out


@pytest.mark.parametrize(
    "item_type,needle",
    [
        ("metrics", "AVAILABLE METRICS"),
        ("profiles", "AVAILABLE PROFILES"),
        ("counters", "not yet implemented"),
        ("devices", "SUPPORTED DEVICES"),
    ],
)
def test_list_command_dispatches(item_type, needle, capsys):
    args = Args(item_type=item_type, category=None)
    assert list_command(args) == 0
    assert needle in capsys.readouterr().out


# --------------------------------------------------------------------------
# info_cmd
# --------------------------------------------------------------------------


def test_show_metric_info_prints_counters_from_backend(capsys):
    with patch("metrix.cli.info_cmd.get_backend", return_value=FakeBackend()):
        show_metric_info(KNOWN_METRIC, "gfx942")
    out = capsys.readouterr().out
    assert "TCC_HIT_sum" in out
    assert "gfx942" in out


def test_show_metric_info_unknown_metric_returns_1(capsys):
    assert show_metric_info("memory.not_a_metric") == 1
    assert "Unknown metric" in capsys.readouterr().out


def test_show_metric_info_unimplemented_on_arch_is_handled(capsys):
    backend = FakeBackend()
    with (
        patch("metrix.cli.info_cmd.get_backend", return_value=backend),
        patch.object(backend, "get_metric_counters", side_effect=ValueError),
    ):
        show_metric_info(KNOWN_METRIC, "gfx1030")
    assert "not implemented" in capsys.readouterr().out


def test_show_profile_info_lists_metrics(capsys):
    name = next(iter(METRIC_PROFILES))
    show_profile_info(name)
    out = capsys.readouterr().out
    assert name.upper() in out
    assert f"metrix profile --profile {name}" in out


def test_show_profile_info_unknown_returns_1(capsys):
    assert show_profile_info("nope") == 1
    assert "Unknown profile" in capsys.readouterr().out


def test_info_command_metric_autodetects_arch(capsys):
    with (
        patch("metrix.cli.info_cmd.detect_or_default", return_value="gfx942") as det,
        patch("metrix.cli.info_cmd.get_backend", return_value=FakeBackend()),
    ):
        args = Args(info_type="metric", name=KNOWN_METRIC)
        assert info_command(args) == 0
        det.assert_called_once()


def test_info_command_metric_honours_explicit_arch():
    with (
        patch("metrix.cli.info_cmd.detect_or_default") as det,
        patch("metrix.cli.info_cmd.get_backend", return_value=FakeBackend()),
    ):
        args = Args(info_type="metric", name=KNOWN_METRIC, arch="gfx950")
        info_command(args)
        det.assert_not_called()


def test_info_command_profile(capsys):
    name = next(iter(METRIC_PROFILES))
    assert info_command(Args(info_type="profile", name=name)) == 0
    assert name.upper() in capsys.readouterr().out


def test_info_command_counter_not_implemented(capsys):
    assert info_command(Args(info_type="counter", name="TCC_HIT")) == 0
    assert "not yet implemented" in capsys.readouterr().out


# --------------------------------------------------------------------------
# main / parser
# --------------------------------------------------------------------------


def test_parser_builds_and_exposes_subcommands():
    parser = create_parser()
    args = parser.parse_args(["profile", "./app"])
    assert args.command == "profile"
    assert args.target == "./app"


def test_bare_invocation_prints_help(capsys):
    with patch("sys.argv", ["metrix"]):
        assert main() == 0
    assert "usage:" in capsys.readouterr().out.lower()


def test_implicit_profile_subcommand_is_inserted():
    # `metrix ./app` must behave as `metrix profile ./app`.
    with (
        patch("sys.argv", ["metrix", "./app"]),
        patch("metrix.cli.main.profile_command", return_value=0) as cmd,
    ):
        assert main() == 0
    assert cmd.call_args[0][0].target == "./app"


def test_explicit_profile_subcommand_is_not_double_inserted():
    with (
        patch("sys.argv", ["metrix", "profile", "./app"]),
        patch("metrix.cli.main.profile_command", return_value=0) as cmd,
    ):
        main()
    assert cmd.call_args[0][0].target == "./app"


def test_main_routes_list():
    with (
        patch("sys.argv", ["metrix", "list", "metrics"]),
        patch("metrix.cli.main.list_command", return_value=0) as cmd,
    ):
        assert main() == 0
    cmd.assert_called_once()


def test_main_routes_info():
    with (
        patch("sys.argv", ["metrix", "info", "metric", KNOWN_METRIC]),
        patch("metrix.cli.main.info_command", return_value=0) as cmd,
    ):
        assert main() == 0
    cmd.assert_called_once()


def test_main_keyboard_interrupt_returns_130(capsys):
    with (
        patch("sys.argv", ["metrix", "profile", "./app"]),
        patch("metrix.cli.main.profile_command", side_effect=KeyboardInterrupt),
    ):
        assert main() == 130
    assert "Interrupted" in capsys.readouterr().out


def test_main_exception_returns_1(capsys):
    with (
        patch("sys.argv", ["metrix", "profile", "./app"]),
        patch("metrix.cli.main.profile_command", side_effect=RuntimeError("kaboom")),
    ):
        assert main() == 1
    assert "kaboom" in capsys.readouterr().err


def test_main_debug_log_prints_traceback(capsys):
    with (
        patch("sys.argv", ["metrix", "profile", "./app", "--log", "debug"]),
        patch("metrix.cli.main.profile_command", side_effect=RuntimeError("kaboom")),
    ):
        assert main() == 1
    assert "Traceback" in capsys.readouterr().err


def test_version_flag_exits_zero(capsys):
    with patch("sys.argv", ["metrix", "--version"]), pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
