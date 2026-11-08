# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for check_installation, `omniprobe-ik check`, and the "not installed" messages."""

import json
import stat
import sys
from pathlib import Path

import pytest

import omniprobe_ik.api as api
from omniprobe_ik import INSTALL_DOC, Omniprobe, check_installation, instrumentation_flags
from omniprobe_ik.cli.main import main


def _fake_install(root: Path) -> Path:
    """An Omniprobe install tree: bin/omniprobe, lib/plugins/*.so, api/__init__.py."""
    prefix = root / "omniprobe"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib" / "plugins").mkdir(parents=True)
    (prefix / "api").mkdir()
    (prefix / "api" / "__init__.py").write_text("")
    for name in api.ANALYSIS_PLUGINS.values():
        (prefix / "lib" / "plugins" / name).write_text("")
    exe = prefix / "bin" / "omniprobe"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


@pytest.fixture
def env(monkeypatch):
    """Isolate from the real environment: no omniprobe, no ROCM_PATH, fixed arch and ROCms."""
    monkeypatch.delenv("OMNIPROBE_PLUGIN_DIR", raising=False)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.setattr(api, "detect_offload_arch", lambda: "gfx90a")
    monkeypatch.setattr(api, "omniprobe_rocm", lambda exe=None: Path("/opt/rocm-7.2.0"))
    monkeypatch.setattr(api, "compiler_rocm", lambda: Path("/opt/rocm-7.2.0"))
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)  # no upstream API installed
    return monkeypatch


def _only_on_path(monkeypatch, *dirs):
    monkeypatch.setenv("PATH", ":".join(str(d) for d in dirs) or "/nonexistent")


class TestCheckInstallation:
    def test_install_on_path_is_ok(self, env, tmp_path):
        exe = _fake_install(tmp_path)
        _only_on_path(env, exe.parent)
        report = check_installation()
        assert report["ok"], report["problems"]
        assert report["executable"] == str(exe)
        assert report["executable_runs"] is True
        assert report["plugin_dir"] == str(exe.parent.parent / "lib" / "plugins")
        assert report["plugins"] == {"memory": True, "basic_blocks": True}
        assert report["python_api"] == str(exe.parent.parent / "api")
        assert report["gpu_arch"] == "gfx90a"
        assert report["warnings"] == []

    def test_absent_names_path_and_install_page(self, env):
        _only_on_path(env)
        report = check_installation()
        assert not report["ok"]
        text = " ".join(report["problems"])
        assert "<prefix>/omniprobe/bin" in text and "PATH" in text
        assert INSTALL_DOC in text
        assert "PYTHONPATH" not in text

    def test_rocm_mismatch_is_a_warning(self, env, tmp_path):
        exe = _fake_install(tmp_path)
        _only_on_path(env, exe.parent)
        env.setattr(api, "compiler_rocm", lambda: Path("/opt/rocm-7.1.0"))
        report = check_installation()
        assert report["ok"]
        assert any(
            "ROCm mismatch" in w and "/opt/rocm-7.1.0" in w and "/opt/rocm-7.2.0" in w
            for w in report["warnings"]
        )

    def test_rocm_path_elsewhere_is_a_warning(self, env, tmp_path):
        exe = _fake_install(tmp_path)
        _only_on_path(env, exe.parent)
        env.setenv("ROCM_PATH", str(tmp_path))
        report = check_installation()
        assert any("ROCM_PATH" in w for w in report["warnings"])

    def test_missing_hipcc_and_arch_are_problems(self, env, tmp_path):
        exe = _fake_install(tmp_path)
        _only_on_path(env, exe.parent)
        env.setattr(api, "compiler_rocm", lambda: None)
        env.setattr(api, "detect_offload_arch", lambda: None)
        report = check_installation()
        assert not report["ok"]
        assert any("hipcc" in p for p in report["problems"])
        assert any("rocminfo" in p for p in report["problems"])

    def test_launcher_that_cannot_start_is_a_problem(self, env, tmp_path):
        exe = _fake_install(tmp_path)
        exe.write_text(
            "#!/bin/sh\necho \"ModuleNotFoundError: No module named 'pyfiglet'\" >&2\nexit 1\n"
        )
        _only_on_path(env, exe.parent)
        report = check_installation()
        assert report["executable_runs"] is False
        assert any("pyfiglet" in p and "requirements.txt" in p for p in report["problems"])

    def test_missing_plugin_is_a_problem(self, env, tmp_path):
        exe = _fake_install(tmp_path)
        (exe.parent.parent / "lib" / "plugins" / api.ANALYSIS_PLUGINS["basic_blocks"]).unlink()
        _only_on_path(env, exe.parent)
        report = check_installation()
        assert report["plugins"] == {"memory": True, "basic_blocks": False}
        assert not report["ok"]


class TestCheckCommand:
    def test_exit_codes_and_json(self, env, tmp_path, capsys, monkeypatch):
        exe = _fake_install(tmp_path)
        _only_on_path(env, exe.parent)
        monkeypatch.setattr("sys.argv", ["omniprobe-ik", "check", "--json"])
        assert main() == 0
        assert json.loads(capsys.readouterr().out)["ok"] is True

        _only_on_path(env)
        monkeypatch.setattr("sys.argv", ["omniprobe-ik", "check"])
        assert main() == 1
        out = capsys.readouterr().out
        assert "NOT FOUND" in out and INSTALL_DOC in out


class TestNotInstalledMessages:
    """Every "not found" message points at PATH and the install page, not PYTHONPATH."""

    def _assert_points_to_install(self, text):
        assert "<prefix>/omniprobe/bin" in text and INSTALL_DOC in text
        assert "PYTHONPATH" not in text

    def test_flags(self, env):
        _only_on_path(env)
        with pytest.raises(FileNotFoundError) as e:
            instrumentation_flags("memory")
        self._assert_points_to_install(str(e.value))

    def test_constructor(self, env):
        _only_on_path(env)
        env.setitem(sys.modules, "omniprobe", None)  # upstream not importable
        env.setitem(sys.modules, "omniprobe.api", None)
        with pytest.raises(ImportError) as e:
            Omniprobe()
        self._assert_points_to_install(str(e.value))

    def test_raw_runner(self, env):
        from omniprobe_ik import run_analyzer

        _only_on_path(env)
        with pytest.raises(FileNotFoundError) as e:
            run_analyzer("Heatmap", "./app")
        self._assert_points_to_install(str(e.value))
