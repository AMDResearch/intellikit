"""Unit tests for kerncap.hsaco_extractor — the roc-obj-extract fallback path.

This module is the fallback used when libkerncap.so did not capture the
.hsaco blob at runtime.  Every external tool it reaches for
(``roc-obj-extract``, ``llvm-nm``/``nm``) is stubbed at its call site, so
these tests need neither ROCm nor a GPU.

``roc-obj-extract`` writes into a ``TemporaryDirectory`` created inside the
function under test, so the stub reads the destination out of the argv it is
handed and materialises real files there.  That keeps the directory-scanning
and file-copying logic under test rather than mocked away.
"""

import os
import subprocess
from unittest.mock import patch

import pytest

from kerncap.hsaco_extractor import _find_matching_code_object, extract_hsaco_from_binary


def which_stub(**available):
    """Build a ``shutil.which`` replacement from name -> path/None."""

    def _which(name):
        return available.get(name)

    return _which


def writes_co_files(*names, returncode=0):
    """A ``subprocess.run`` stub that materialises *names* in the -o directory."""

    def _run(argv, **_kwargs):
        outdir = argv[argv.index("-o") + 1]
        for name in names:
            with open(os.path.join(outdir, name), "wb") as fh:
                fh.write(b"\x7fELF fake code object")
        return subprocess.CompletedProcess(argv, returncode, stdout="", stderr="")

    return _run


@pytest.fixture
def binary(tmp_path):
    """A real file standing in for the compiled ELF."""
    path = tmp_path / "my_app"
    path.write_bytes(b"\x7fELF")
    return str(path)


# --------------------------------------------------------------------------
# extract_hsaco_from_binary — guards
# --------------------------------------------------------------------------


class TestGuards:
    def test_missing_roc_obj_extract_returns_false(self, binary, tmp_path):
        """No tool on PATH is a normal outcome, not an exception."""
        with (
            patch("shutil.which", which_stub()),
            patch("subprocess.run") as run,
        ):
            ok = extract_hsaco_from_binary(binary, "kern", "gfx90a", str(tmp_path / "out.hsaco"))

        assert ok is False
        run.assert_not_called()

    def test_missing_binary_returns_false(self, tmp_path):
        with (
            patch("shutil.which", which_stub(**{"roc-obj-extract": "/usr/bin/roc-obj-extract"})),
            patch("subprocess.run") as run,
        ):
            ok = extract_hsaco_from_binary(
                str(tmp_path / "nope"), "kern", "gfx90a", str(tmp_path / "out.hsaco")
            )

        assert ok is False
        run.assert_not_called()

    def test_directory_is_not_a_valid_binary(self, tmp_path):
        """``os.path.isfile`` must reject a directory, not just a missing path."""
        with patch("shutil.which", which_stub(**{"roc-obj-extract": "/x"})):
            ok = extract_hsaco_from_binary(
                str(tmp_path), "kern", "gfx90a", str(tmp_path / "out.hsaco")
            )
        assert ok is False


# --------------------------------------------------------------------------
# extract_hsaco_from_binary — roc-obj-extract failures
# --------------------------------------------------------------------------


class TestExtractFailures:
    @pytest.mark.parametrize(
        "exc",
        [
            subprocess.TimeoutExpired("roc-obj-extract", 60),
            FileNotFoundError("roc-obj-extract"),
        ],
    )
    def test_subprocess_exceptions_are_caught(self, binary, tmp_path, exc):
        """Both documented failure modes return False rather than propagating."""
        with (
            patch("shutil.which", which_stub(**{"roc-obj-extract": "/x"})),
            patch("subprocess.run", side_effect=exc),
        ):
            ok = extract_hsaco_from_binary(binary, "kern", "gfx90a", str(tmp_path / "out.hsaco"))

        assert ok is False

    def test_nonzero_exit_returns_false(self, binary, tmp_path):
        with (
            patch("shutil.which", which_stub(**{"roc-obj-extract": "/x"})),
            patch("subprocess.run", writes_co_files(returncode=2)),
        ):
            ok = extract_hsaco_from_binary(binary, "kern", "gfx90a", str(tmp_path / "out.hsaco"))

        assert ok is False

    def test_no_code_objects_produced_returns_false(self, binary, tmp_path):
        """Exit 0 but an empty output directory is still a failure."""
        with (
            patch("shutil.which", which_stub(**{"roc-obj-extract": "/x"})),
            patch("subprocess.run", writes_co_files()),
        ):
            ok = extract_hsaco_from_binary(binary, "kern", "gfx90a", str(tmp_path / "out.hsaco"))

        assert ok is False

    def test_unrelated_files_are_not_treated_as_code_objects(self, binary, tmp_path):
        """Only .co / .hsaco count; roc-obj-extract may drop other files."""
        with (
            patch("shutil.which", which_stub(**{"roc-obj-extract": "/x"})),
            patch("subprocess.run", writes_co_files("notes.txt", "manifest.json")),
        ):
            ok = extract_hsaco_from_binary(binary, "kern", "gfx90a", str(tmp_path / "out.hsaco"))

        assert ok is False

    def test_no_symbol_match_returns_false(self, binary, tmp_path):
        """When the matcher finds nothing, nothing is written."""
        out = tmp_path / "out.hsaco"
        with (
            patch("shutil.which", which_stub(**{"roc-obj-extract": "/x"})),
            patch("subprocess.run", writes_co_files("a.co")),
            patch("kerncap.hsaco_extractor._find_matching_code_object", return_value=None),
        ):
            ok = extract_hsaco_from_binary(binary, "kern", "gfx90a", str(out))

        assert ok is False
        assert not out.exists()


# --------------------------------------------------------------------------
# extract_hsaco_from_binary — success
# --------------------------------------------------------------------------


class TestExtractSuccess:
    def test_copies_the_matched_object_to_output(self, binary, tmp_path):
        out = tmp_path / "out.hsaco"
        with (
            patch("shutil.which", which_stub(**{"roc-obj-extract": "/x"})),
            patch("subprocess.run", writes_co_files("kernel_gfx90a.co")),
        ):
            ok = extract_hsaco_from_binary(binary, "kern", "gfx90a", str(out))

        assert ok is True
        assert out.exists()
        assert out.read_bytes() == b"\x7fELF fake code object"

    def test_accepts_hsaco_extension_too(self, binary, tmp_path):
        out = tmp_path / "out.hsaco"
        with (
            patch("shutil.which", which_stub(**{"roc-obj-extract": "/x"})),
            patch("subprocess.run", writes_co_files("kernel_gfx90a.hsaco")),
        ):
            ok = extract_hsaco_from_binary(binary, "kern", "gfx90a", str(out))

        assert ok is True
        assert out.exists()

    def test_invokes_roc_obj_extract_with_a_temp_output_dir(self, binary, tmp_path):
        """The -o directory must be a scratch dir, not the caller's output path."""
        seen = {}

        def _run(argv, **kwargs):
            seen["argv"] = list(argv)
            seen["timeout"] = kwargs.get("timeout")
            return writes_co_files("k.co")(argv, **kwargs)

        out = tmp_path / "out.hsaco"
        with (
            patch("shutil.which", which_stub(**{"roc-obj-extract": "/x"})),
            patch("subprocess.run", _run),
        ):
            extract_hsaco_from_binary(binary, "kern", "gfx90a", str(out))

        argv = seen["argv"]
        assert argv[0] == "roc-obj-extract"
        assert argv[-1] == binary
        outdir = argv[argv.index("-o") + 1]
        assert outdir != str(out)
        assert "kerncap_hsaco_" in outdir
        assert seen["timeout"] == 60

    def test_temp_directory_is_cleaned_up(self, binary, tmp_path):
        """The scratch dir must not outlive the call."""
        captured = {}

        def _run(argv, **kwargs):
            captured["outdir"] = argv[argv.index("-o") + 1]
            return writes_co_files("k.co")(argv, **kwargs)

        with (
            patch("shutil.which", which_stub(**{"roc-obj-extract": "/x"})),
            patch("subprocess.run", _run),
        ):
            extract_hsaco_from_binary(binary, "kern", "gfx90a", str(tmp_path / "out.hsaco"))

        assert not os.path.exists(captured["outdir"])


# --------------------------------------------------------------------------
# _find_matching_code_object
# --------------------------------------------------------------------------


def make_co(tmp_path, name: str) -> str:
    path = tmp_path / name
    path.write_bytes(b"\x7fELF")
    return str(path)


class TestFindMatchingWithoutNm:
    """No symbol inspector available — selection falls back to filenames."""

    def test_prefers_an_arch_matching_filename(self, tmp_path):
        files = [make_co(tmp_path, "other_gfx942.co"), make_co(tmp_path, "kern_gfx90a.co")]
        with patch("shutil.which", which_stub()):
            assert _find_matching_code_object(files, "kern", "gfx90a") == files[1]

    def test_falls_back_to_the_first_file(self, tmp_path):
        files = [make_co(tmp_path, "a.co"), make_co(tmp_path, "b.co")]
        with patch("shutil.which", which_stub()):
            assert _find_matching_code_object(files, "kern", "gfx90a") == files[0]

    def test_empty_list_returns_none(self):
        with patch("shutil.which", which_stub()):
            assert _find_matching_code_object([], "kern", "gfx90a") is None


class TestFindMatchingWithNm:
    def test_returns_the_object_whose_symbols_contain_the_kernel(self, tmp_path):
        a = make_co(tmp_path, "a.co")
        b = make_co(tmp_path, "b.co")

        def _nm(argv, **_kwargs):
            target = argv[1]
            out = "T my_kernel\n" if target == b else "T something_else\n"
            return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

        with (
            patch("shutil.which", which_stub(**{"llvm-nm": "/usr/bin/llvm-nm"})),
            patch("subprocess.run", _nm),
        ):
            assert _find_matching_code_object([a, b], "my_kernel", "gfx90a") == b

    def test_prefers_llvm_nm_over_nm(self, tmp_path):
        co = make_co(tmp_path, "a.co")
        seen = {}

        def _nm(argv, **_kwargs):
            seen["tool"] = argv[0]
            return subprocess.CompletedProcess(argv, 0, stdout="T k\n", stderr="")

        with (
            patch(
                "shutil.which", which_stub(**{"llvm-nm": "/usr/bin/llvm-nm", "nm": "/usr/bin/nm"})
            ),
            patch("subprocess.run", _nm),
        ):
            _find_matching_code_object([co], "k", "gfx90a")

        assert seen["tool"] == "/usr/bin/llvm-nm"

    def test_falls_back_to_plain_nm(self, tmp_path):
        co = make_co(tmp_path, "a.co")
        seen = {}

        def _nm(argv, **_kwargs):
            seen["tool"] = argv[0]
            return subprocess.CompletedProcess(argv, 0, stdout="T k\n", stderr="")

        with (
            patch("shutil.which", which_stub(**{"nm": "/usr/bin/nm"})),
            patch("subprocess.run", _nm),
        ):
            _find_matching_code_object([co], "k", "gfx90a")

        assert seen["tool"] == "/usr/bin/nm"

    def test_arch_matching_files_are_inspected_first(self, tmp_path):
        """Ordering matters: the arch-matching object should win a tie."""
        wrong = make_co(tmp_path, "kern_gfx942.co")
        right = make_co(tmp_path, "kern_gfx90a.co")
        order = []

        def _nm(argv, **_kwargs):
            order.append(os.path.basename(argv[1]))
            return subprocess.CompletedProcess(argv, 0, stdout="T my_kernel\n", stderr="")

        with (
            patch("shutil.which", which_stub(**{"nm": "/usr/bin/nm"})),
            patch("subprocess.run", _nm),
        ):
            chosen = _find_matching_code_object([wrong, right], "my_kernel", "gfx90a")

        assert order[0] == "kern_gfx90a.co"
        assert chosen == right

    def test_nonzero_nm_exit_is_skipped(self, tmp_path):
        a = make_co(tmp_path, "a.co")
        b = make_co(tmp_path, "b.co")

        def _nm(argv, **_kwargs):
            if argv[1] == a:
                return subprocess.CompletedProcess(argv, 1, stdout="T my_kernel\n", stderr="")
            return subprocess.CompletedProcess(argv, 0, stdout="T my_kernel\n", stderr="")

        with (
            patch("shutil.which", which_stub(**{"nm": "/usr/bin/nm"})),
            patch("subprocess.run", _nm),
        ):
            assert _find_matching_code_object([a, b], "my_kernel", "gfx90a") == b

    @pytest.mark.parametrize(
        "exc", [subprocess.TimeoutExpired("nm", 10), OSError("exec format error")]
    )
    def test_nm_exceptions_skip_that_object(self, tmp_path, exc):
        a = make_co(tmp_path, "a.co")
        b = make_co(tmp_path, "b.co")

        def _nm(argv, **_kwargs):
            if argv[1] == a:
                raise exc
            return subprocess.CompletedProcess(argv, 0, stdout="T my_kernel\n", stderr="")

        with (
            patch("shutil.which", which_stub(**{"nm": "/usr/bin/nm"})),
            patch("subprocess.run", _nm),
        ):
            assert _find_matching_code_object([a, b], "my_kernel", "gfx90a") == b

    def test_no_symbol_match_falls_back_to_arch(self, tmp_path):
        """A best guess is still returned — the caller decides what to do."""
        other = make_co(tmp_path, "other_gfx942.co")
        arch = make_co(tmp_path, "other_gfx90a.co")

        def _nm(argv, **_kwargs):
            return subprocess.CompletedProcess(argv, 0, stdout="T unrelated\n", stderr="")

        with (
            patch("shutil.which", which_stub(**{"nm": "/usr/bin/nm"})),
            patch("subprocess.run", _nm),
        ):
            assert _find_matching_code_object([other, arch], "missing", "gfx90a") == arch

    def test_no_symbol_and_no_arch_match_returns_first(self, tmp_path):
        a = make_co(tmp_path, "a.co")
        b = make_co(tmp_path, "b.co")

        def _nm(argv, **_kwargs):
            return subprocess.CompletedProcess(argv, 0, stdout="T unrelated\n", stderr="")

        with (
            patch("shutil.which", which_stub(**{"nm": "/usr/bin/nm"})),
            patch("subprocess.run", _nm),
        ):
            assert _find_matching_code_object([a, b], "missing", "gfx90a") == a

    def test_empty_list_with_nm_returns_none(self):
        with patch("shutil.which", which_stub(**{"nm": "/usr/bin/nm"})):
            assert _find_matching_code_object([], "k", "gfx90a") is None
