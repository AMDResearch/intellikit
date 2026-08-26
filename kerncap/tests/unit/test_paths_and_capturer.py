"""Unit tests for kerncap's artifact-path resolution and the capture wrapper.

``_get_lib_path`` and ``_get_replay_path`` decide whether kerncap can find the
things it was built with. They are the first thing to fail on a broken install,
and their error messages are what a user has to debug from, so the search order
and the "searched:" report are pinned here.

Both search real filesystem locations, so ``pathlib.Path.is_file`` is stubbed
to place a hit at a chosen candidate rather than writing into site-packages.
"""

import os
import subprocess
from unittest.mock import patch

import pytest

import kerncap
from kerncap import _get_lib_path, _get_replay_path
from kerncap.capturer import run_capture


def only_this_file_exists(target: str):
    """A ``Path.is_file`` replacement that reports a hit only for *target*."""

    def _is_file(self):
        return str(self) == target

    return _is_file


def nothing_exists(self):
    return False


# --------------------------------------------------------------------------
# _get_replay_path
# --------------------------------------------------------------------------


class TestGetReplayPath:
    def test_finds_the_binary_in_the_package_bin_dir(self):
        import pathlib

        pkg_dir = pathlib.Path(kerncap.__file__).resolve().parent
        target = str(pkg_dir / "bin" / "kerncap-replay")

        with patch.object(pathlib.Path, "is_file", only_this_file_exists(target)):
            assert _get_replay_path() == target

    def test_falls_back_to_path(self):
        """A system install with no packaged binary still works."""
        import pathlib

        with (
            patch.object(pathlib.Path, "is_file", nothing_exists),
            patch("shutil.which", return_value="/usr/local/bin/kerncap-replay"),
        ):
            assert _get_replay_path() == "/usr/local/bin/kerncap-replay"

    def test_package_copy_wins_over_path(self):
        import pathlib

        pkg_dir = pathlib.Path(kerncap.__file__).resolve().parent
        target = str(pkg_dir / "bin" / "kerncap-replay")

        with (
            patch.object(pathlib.Path, "is_file", only_this_file_exists(target)),
            patch("shutil.which", return_value="/usr/local/bin/kerncap-replay"),
        ):
            assert _get_replay_path() == target

    def test_reports_everywhere_it_looked(self):
        """The message is the user's only debugging handle on a bad install."""
        import pathlib

        with (
            patch.object(pathlib.Path, "is_file", nothing_exists),
            patch("shutil.which", return_value=None),
        ):
            with pytest.raises(FileNotFoundError) as exc:
                _get_replay_path()

        message = str(exc.value)
        assert "Could not locate kerncap-replay" in message
        assert "Searched:" in message
        assert "bin/kerncap-replay" in message


# --------------------------------------------------------------------------
# _get_lib_path
# --------------------------------------------------------------------------


class TestGetLibPath:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        lib = tmp_path / "custom_libkerncap.so"
        lib.write_bytes(b"\x7fELF")
        monkeypatch.setenv("KERNCAP_LIB_PATH", str(lib))

        assert _get_lib_path() == str(lib)

    def test_env_override_pointing_at_nothing_is_ignored(self, tmp_path, monkeypatch):
        """A stale env var must not shadow a working packaged library."""
        import pathlib

        monkeypatch.setenv("KERNCAP_LIB_PATH", str(tmp_path / "gone.so"))
        pkg_dir = pathlib.Path(kerncap.__file__).resolve().parent
        target = str(pkg_dir / "lib" / "libkerncap.so")

        with patch.object(pathlib.Path, "is_file", only_this_file_exists(target)):
            assert _get_lib_path() == target

    def test_finds_the_library_beside_the_package(self, monkeypatch):
        import pathlib

        monkeypatch.delenv("KERNCAP_LIB_PATH", raising=False)
        pkg_dir = pathlib.Path(kerncap.__file__).resolve().parent
        target = str(pkg_dir / "libkerncap.so")

        with patch.object(pathlib.Path, "is_file", only_this_file_exists(target)):
            assert _get_lib_path() == target

    def test_reports_everywhere_it_looked(self, monkeypatch):
        import pathlib

        monkeypatch.delenv("KERNCAP_LIB_PATH", raising=False)

        with patch.object(pathlib.Path, "is_file", nothing_exists):
            with pytest.raises(FileNotFoundError) as exc:
                _get_lib_path()

        message = str(exc.value)
        assert "Could not locate libkerncap.so" in message
        assert "KERNCAP_LIB_PATH" in message
        assert "Searched:" in message


# --------------------------------------------------------------------------
# Kerncap.replay — capture directory resolution
# --------------------------------------------------------------------------


class TestReplayCaptureDir:
    def test_prefers_the_capture_subdirectory(self, tmp_path):
        (tmp_path / "capture").mkdir()
        kc = kerncap.Kerncap()

        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run,
        ):
            kc.replay(str(tmp_path))

        assert run.call_args.args[0][1] == str(tmp_path / "capture")

    def test_falls_back_to_the_directory_itself(self, tmp_path):
        """Lets a user point straight at a capture/ dir."""
        kc = kerncap.Kerncap()

        with (
            patch("kerncap._get_replay_path", return_value="/bin/replay"),
            patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run,
        ):
            kc.replay(str(tmp_path))

        assert run.call_args.args[0][1] == str(tmp_path)


# --------------------------------------------------------------------------
# capturer.run_capture — failure paths
# --------------------------------------------------------------------------


def fake_streaming(artifacts=(), returncode=0, raises=None):
    """Stand in for ``run_streaming``, materialising capture artifacts."""

    def _run(cmd, **kwargs):
        if raises is not None:
            raise raises
        out = kwargs["env"]["KERNCAP_OUTPUT"]
        for name in artifacts:
            with open(os.path.join(out, name), "w") as fh:
                fh.write("{}")
        return subprocess.CompletedProcess(
            list(cmd), returncode, stdout="app stdout", stderr="app stderr"
        )

    return _run


@pytest.fixture
def lib(tmp_path):
    path = tmp_path / "libkerncap.so"
    path.write_bytes(b"\x7fELF")
    return str(path)


class TestRunCaptureFailures:
    def test_timeout_is_translated(self, tmp_path, lib):
        with (
            patch("kerncap._get_lib_path", return_value=lib),
            patch(
                "kerncap.capturer.run_streaming",
                fake_streaming(raises=subprocess.TimeoutExpired("app", 30)),
            ),
        ):
            with pytest.raises(TimeoutError, match="did not complete within 30s"):
                run_capture("k", ["./app"], str(tmp_path / "cap"), timeout=30)

    def test_no_artifacts_raises_with_both_streams(self, tmp_path, lib):
        """The child's output is the only clue why interception did not fire."""
        with (
            patch("kerncap._get_lib_path", return_value=lib),
            patch("kerncap.capturer.run_streaming", fake_streaming(artifacts=())),
        ):
            with pytest.raises(RuntimeError) as exc:
                run_capture("k", ["./app"], str(tmp_path / "cap"))

        message = str(exc.value)
        assert "Capture did not produce output" in message
        assert "app stdout" in message
        assert "app stderr" in message

    @pytest.mark.parametrize("artifact", ["dispatch.json", "metadata.json"])
    def test_either_artifact_satisfies_the_check(self, tmp_path, lib, artifact):
        out = tmp_path / "cap"
        with (
            patch("kerncap._get_lib_path", return_value=lib),
            patch("kerncap.capturer.run_streaming", fake_streaming(artifacts=(artifact,))),
        ):
            assert run_capture("k", ["./app"], str(out)) == str(out)

    def test_empty_streams_are_reported_as_na(self, tmp_path, lib):
        """A silent child still produces a readable message."""

        def _run(cmd, **kwargs):
            return subprocess.CompletedProcess(list(cmd), 0, stdout="", stderr="")

        with (
            patch("kerncap._get_lib_path", return_value=lib),
            patch("kerncap.capturer.run_streaming", _run),
        ):
            with pytest.raises(RuntimeError, match="N/A"):
                run_capture("k", ["./app"], str(tmp_path / "cap"))
