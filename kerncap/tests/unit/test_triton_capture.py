"""Unit tests for the two Triton capture backends' injection contracts.

Neither backend can be verified by its return value — both just hand back
``output_dir``. What actually determines whether a capture works is the
environment they build for the child: ``LD_PRELOAD``, ``PYTHONPATH``, and the
``KERNCAP_*`` variables the compile shim and ``libkerncap.so`` read. If one of
those silently stops being injected, capture produces nothing and the failure
surfaces far away from the cause.

So that environment is what these tests assert. ``run_streaming`` is stubbed at
its call site and records the env it was handed; no GPU, no ROCm, no Triton.
"""

import os
import subprocess
from unittest.mock import patch

import pytest

from kerncap.triton_capture import run_triton_capture
from kerncap.triton_capture_hsa import _maybe_clear_triton_caches, run_triton_capture_hsa


class Recorder:
    """A ``run_streaming`` stub that records its call and fakes the artifacts."""

    def __init__(self, artifacts=("metadata.json",), returncode=0, raises=None):
        self.artifacts = artifacts
        self.returncode = returncode
        self.raises = raises
        self.cmd = None
        self.env = None
        self.kwargs = None

    def __call__(self, cmd, **kwargs):
        self.cmd = list(cmd)
        self.env = dict(kwargs.get("env") or {})
        self.kwargs = kwargs
        if self.raises is not None:
            raise self.raises
        out = self.env.get("KERNCAP_OUTPUT")
        for name in self.artifacts:
            with open(os.path.join(out, name), "w") as fh:
                fh.write("{}")
        return subprocess.CompletedProcess(
            args=list(cmd), returncode=self.returncode, stdout="app out", stderr="app err"
        )


# --------------------------------------------------------------------------
# run_triton_capture — the legacy python backend
# --------------------------------------------------------------------------


class TestRunTritonCapture:
    def test_returns_the_output_dir_and_creates_it(self, tmp_path):
        out = tmp_path / "cap"
        rec = Recorder()
        with patch("kerncap.triton_capture.run_streaming", rec):
            assert run_triton_capture("k", ["python", "b.py"], str(out)) == str(out)
        assert out.is_dir()

    def test_command_is_passed_through_unwrapped(self, tmp_path):
        """No runpy wrapper — anything that eventually runs Triton is hooked."""
        rec = Recorder()
        with patch("kerncap.triton_capture.run_streaming", rec):
            run_triton_capture("k", ["vllm", "serve", "model"], str(tmp_path / "cap"))

        assert rec.cmd == ["vllm", "serve", "model"]

    def test_injects_the_kernel_and_output_variables(self, tmp_path):
        out = tmp_path / "cap"
        rec = Recorder()
        with patch("kerncap.triton_capture.run_streaming", rec):
            run_triton_capture("my_kernel", ["python", "b.py"], str(out))

        assert rec.env["KERNCAP_KERNEL"] == "my_kernel"
        assert rec.env["KERNCAP_OUTPUT"] == str(out)

    def test_pythonpath_points_at_the_sitecustomize_dir(self, tmp_path):
        """This is the mechanism: every interpreter imports sitecustomize."""
        rec = Recorder()
        with patch("kerncap.triton_capture.run_streaming", rec):
            run_triton_capture("k", ["python", "b.py"], str(tmp_path / "cap"))

        site_dir = rec.env["PYTHONPATH"].split(os.pathsep)[0]
        assert "kerncap_site_" in site_dir
        assert rec.env["_KERNCAP_TRITON_HOOK"].startswith(site_dir)

    def test_existing_pythonpath_is_preserved(self, tmp_path, monkeypatch):
        """Clobbering the user's PYTHONPATH would break their imports."""
        monkeypatch.setenv("PYTHONPATH", "/user/libs")
        rec = Recorder()
        with patch("kerncap.triton_capture.run_streaming", rec):
            run_triton_capture("k", ["python", "b.py"], str(tmp_path / "cap"))

        parts = rec.env["PYTHONPATH"].split(os.pathsep)
        assert len(parts) == 2
        assert parts[1] == "/user/libs"

    def test_sitecustomize_and_hook_are_written(self, tmp_path):
        """Captured while the call is in flight — the dir is removed after."""
        contents = {}
        rec = Recorder()

        def _capture_files(cmd, **kwargs):
            site_dir = kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0]
            hook = kwargs["env"]["_KERNCAP_TRITON_HOOK"]
            contents["site"] = open(os.path.join(site_dir, "sitecustomize.py")).read()
            contents["hook"] = open(hook).read()
            return rec(cmd, **kwargs)

        with patch("kerncap.triton_capture.run_streaming", _capture_files):
            run_triton_capture("k", ["python", "b.py"], str(tmp_path / "cap"))

        assert contents["site"].strip()
        assert contents["hook"].strip()

    def test_dispatch_is_only_set_when_non_negative(self, tmp_path):
        """-1 means 'first match' and must not pin a dispatch index."""
        rec = Recorder()
        with patch("kerncap.triton_capture.run_streaming", rec):
            run_triton_capture("k", ["python", "b.py"], str(tmp_path / "a"), dispatch=-1)
        assert "KERNCAP_DISPATCH" not in rec.env

        rec2 = Recorder()
        with patch("kerncap.triton_capture.run_streaming", rec2):
            run_triton_capture("k", ["python", "b.py"], str(tmp_path / "b"), dispatch=0)
        assert rec2.env["KERNCAP_DISPATCH"] == "0"

    def test_timeout_is_forwarded_and_translated(self, tmp_path):
        rec = Recorder()
        with patch("kerncap.triton_capture.run_streaming", rec):
            run_triton_capture("k", ["python", "b.py"], str(tmp_path / "cap"), timeout=45)
        assert rec.kwargs["timeout"] == 45

        boom = Recorder(raises=subprocess.TimeoutExpired("python", 45))
        with patch("kerncap.triton_capture.run_streaming", boom):
            with pytest.raises(TimeoutError, match="did not complete within 45s"):
                run_triton_capture("k", ["python", "b.py"], str(tmp_path / "c2"), timeout=45)

    def test_missing_metadata_raises_with_output_tails(self, tmp_path):
        """The child's own output is the only clue why the hook never fired."""
        rec = Recorder(artifacts=())
        with patch("kerncap.triton_capture.run_streaming", rec):
            with pytest.raises(RuntimeError) as exc:
                run_triton_capture("k", ["python", "b.py"], str(tmp_path / "cap"))

        assert "did not produce metadata.json" in str(exc.value)
        assert "app out" in str(exc.value)
        assert "app err" in str(exc.value)

    def test_temp_site_dir_is_always_cleaned_up(self, tmp_path):
        """Including on the failure path — the finally block."""
        seen = {}

        def _run(cmd, **kwargs):
            seen["site"] = kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0]
            raise RuntimeError("boom")

        with patch("kerncap.triton_capture.run_streaming", _run):
            with pytest.raises(RuntimeError):
                run_triton_capture("k", ["python", "b.py"], str(tmp_path / "cap"))

        assert not os.path.exists(seen["site"])


# --------------------------------------------------------------------------
# _maybe_clear_triton_caches
# --------------------------------------------------------------------------


class TestMaybeClearTritonCaches:
    def test_clears_the_cache_under_triton_home(self, tmp_path, monkeypatch):
        """A cache hit means the compile shim never fires and name_map is empty."""
        monkeypatch.setenv("TRITON_HOME", str(tmp_path))
        monkeypatch.delenv("KERNCAP_NO_CLEAR_TRITON_CACHE", raising=False)
        cache = tmp_path / ".triton" / "cache"
        cache.mkdir(parents=True)
        (cache / "entry").write_text("x")

        _maybe_clear_triton_caches()

        assert not cache.exists()

    def test_opt_out_leaves_the_cache_alone(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRITON_HOME", str(tmp_path))
        monkeypatch.setenv("KERNCAP_NO_CLEAR_TRITON_CACHE", "1")
        cache = tmp_path / ".triton" / "cache"
        cache.mkdir(parents=True)

        _maybe_clear_triton_caches()

        assert cache.exists()

    def test_only_the_exact_opt_out_value_counts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRITON_HOME", str(tmp_path))
        monkeypatch.setenv("KERNCAP_NO_CLEAR_TRITON_CACHE", "yes")
        cache = tmp_path / ".triton" / "cache"
        cache.mkdir(parents=True)

        _maybe_clear_triton_caches()

        assert not cache.exists()

    def test_falls_back_to_the_home_directory(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TRITON_HOME", raising=False)
        monkeypatch.delenv("KERNCAP_NO_CLEAR_TRITON_CACHE", raising=False)
        monkeypatch.setattr(os.path, "expanduser", lambda _p: str(tmp_path))
        cache = tmp_path / ".triton" / "cache"
        cache.mkdir(parents=True)

        _maybe_clear_triton_caches()

        assert not cache.exists()

    def test_absent_cache_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRITON_HOME", str(tmp_path))
        monkeypatch.delenv("KERNCAP_NO_CLEAR_TRITON_CACHE", raising=False)

        _maybe_clear_triton_caches()  # must not raise

    def test_removal_failure_is_swallowed(self, tmp_path, monkeypatch):
        """A read-only cache must not abort the capture."""
        monkeypatch.setenv("TRITON_HOME", str(tmp_path))
        monkeypatch.delenv("KERNCAP_NO_CLEAR_TRITON_CACHE", raising=False)
        (tmp_path / ".triton" / "cache").mkdir(parents=True)

        with patch("shutil.rmtree", side_effect=OSError("read-only")):
            _maybe_clear_triton_caches()  # must not raise


# --------------------------------------------------------------------------
# run_triton_capture_hsa — the default backend
# --------------------------------------------------------------------------


@pytest.fixture
def lib(tmp_path):
    """A stand-in libkerncap.so path, patched into _get_lib_path."""
    path = tmp_path / "libkerncap.so"
    path.write_bytes(b"\x7fELF")
    return str(path)


@pytest.fixture(autouse=True)
def no_cache_clearing(monkeypatch):
    """Never touch the developer's real ~/.triton/cache from these tests."""
    monkeypatch.setenv("KERNCAP_NO_CLEAR_TRITON_CACHE", "1")


def run_hsa(rec, lib, out, **kwargs):
    with (
        patch("kerncap._get_lib_path", return_value=lib),
        patch("kerncap.triton_capture_hsa.run_streaming", rec),
    ):
        return run_triton_capture_hsa("k", ["python", "b.py"], str(out), **kwargs)


class TestRunTritonCaptureHsa:
    def test_returns_the_output_dir(self, tmp_path, lib):
        out = tmp_path / "cap"
        assert run_hsa(Recorder(), lib, out) == str(out)

    def test_ld_preload_is_set_to_the_library(self, tmp_path, lib):
        """Without LD_PRELOAD nothing is intercepted at all."""
        rec = Recorder()
        run_hsa(rec, lib, tmp_path / "cap")

        assert rec.env["LD_PRELOAD"] == lib

    def test_existing_ld_preload_is_prepended_not_replaced(self, tmp_path, lib, monkeypatch):
        monkeypatch.setenv("LD_PRELOAD", "/other/lib.so")
        rec = Recorder()
        run_hsa(rec, lib, tmp_path / "cap")

        assert rec.env["LD_PRELOAD"] == f"{lib}:/other/lib.so"

    @pytest.mark.parametrize("var", ["HSA_TOOLS_LIB", "HSA_TOOLS_REPORT_LOAD_FAILURE"])
    def test_inherited_hsa_tools_variables_are_stripped(self, tmp_path, lib, monkeypatch, var):
        """A leftover HSA_TOOLS_LIB would load a second, conflicting tool."""
        monkeypatch.setenv(var, "/some/other/tool.so")
        rec = Recorder()
        run_hsa(rec, lib, tmp_path / "cap")

        assert var not in rec.env

    def test_injects_the_full_kerncap_contract(self, tmp_path, lib):
        out = tmp_path / "cap"
        rec = Recorder()
        run_hsa(rec, lib, out)

        assert rec.env["KERNCAP_KERNEL"] == "k"
        assert rec.env["KERNCAP_OUTPUT"] == str(out)
        assert rec.env["KERNCAP_CAPTURE_CHILD"] == "1"
        assert rec.env["KERNCAP_TRITON_NAME_MAP"] == str(out / "name_map.json")
        assert rec.env["KERNCAP_TRITON_HSACO_DIR"] == str(out / "triton_hsacos")
        assert rec.env["KERNCAP_TRITON_SOURCE_DIR"] == str(out / "triton_sources")

    def test_artifact_directories_are_created_up_front(self, tmp_path, lib):
        """The shim writes into these from inside the child; they must exist."""
        out = tmp_path / "cap"
        run_hsa(Recorder(), lib, out)

        assert (out / "triton_hsacos").is_dir()
        assert (out / "triton_sources").is_dir()

    def test_pythonpath_carries_the_hsa_hook(self, tmp_path, lib):
        rec = Recorder()
        run_hsa(rec, lib, tmp_path / "cap")

        site_dir = rec.env["PYTHONPATH"].split(os.pathsep)[0]
        assert "kerncap_site_hsa_" in site_dir
        assert rec.env["_KERNCAP_TRITON_HSA_HOOK"].startswith(site_dir)

    def test_existing_pythonpath_is_preserved(self, tmp_path, lib, monkeypatch):
        monkeypatch.setenv("PYTHONPATH", "/user/libs")
        rec = Recorder()
        run_hsa(rec, lib, tmp_path / "cap")

        assert rec.env["PYTHONPATH"].split(os.pathsep)[1] == "/user/libs"

    def test_completion_sentinel_is_passed(self, tmp_path, lib):
        """This is what lets a long-lived host be killed once artifacts land."""
        out = tmp_path / "cap"
        rec = Recorder()
        run_hsa(rec, lib, out)

        assert rec.kwargs["completion_sentinel"] == str(out / "capture_complete")

    def test_dispatch_is_only_set_when_non_negative(self, tmp_path, lib):
        rec = Recorder()
        run_hsa(rec, lib, tmp_path / "a", dispatch=-1)
        assert "KERNCAP_DISPATCH" not in rec.env

        rec2 = Recorder()
        run_hsa(rec2, lib, tmp_path / "b", dispatch=3)
        assert rec2.env["KERNCAP_DISPATCH"] == "3"

    @pytest.mark.parametrize("artifact", ["dispatch.json", "metadata.json"])
    def test_either_artifact_satisfies_the_check(self, tmp_path, lib, artifact):
        """The HSA path may write either; requiring both would be wrong."""
        out = tmp_path / "cap"
        assert run_hsa(Recorder(artifacts=(artifact,)), lib, out) == str(out)

    def test_neither_artifact_raises_with_output_tails(self, tmp_path, lib):
        with pytest.raises(RuntimeError) as exc:
            run_hsa(Recorder(artifacts=()), lib, tmp_path / "cap")

        assert "did not produce metadata.json/dispatch.json" in str(exc.value)
        assert "app out" in str(exc.value)
        assert "app err" in str(exc.value)

    def test_a_sentinel_killed_child_is_not_an_error(self, tmp_path, lib):
        """The watchdog kills with a signal; artifacts on disk mean success."""
        out = tmp_path / "cap"
        rec = Recorder(returncode=-15)

        assert run_hsa(rec, lib, out) == str(out)

    def test_timeout_is_forwarded_and_translated(self, tmp_path, lib):
        rec = Recorder()
        run_hsa(rec, lib, tmp_path / "cap", timeout=60)
        assert rec.kwargs["timeout"] == 60

        boom = Recorder(raises=subprocess.TimeoutExpired("python", 60))
        with pytest.raises(TimeoutError, match="did not complete within 60s"):
            run_hsa(boom, lib, tmp_path / "c2", timeout=60)

    def test_temp_site_dir_is_always_cleaned_up(self, tmp_path, lib):
        seen = {}

        def _run(cmd, **kwargs):
            seen["site"] = kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0]
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            run_hsa(_run, lib, tmp_path / "cap")

        assert not os.path.exists(seen["site"])

    def test_a_missing_library_propagates(self, tmp_path):
        """No libkerncap.so means the HSA backend cannot work at all."""
        with (
            patch("kerncap._get_lib_path", side_effect=FileNotFoundError("libkerncap.so")),
            patch("kerncap.triton_capture_hsa.run_streaming", Recorder()) as _,
        ):
            with pytest.raises(FileNotFoundError):
                run_triton_capture_hsa("k", ["python", "b.py"], str(tmp_path / "cap"))
