"""Unit tests for kerncap.extract's source-failure classifier and Triton pipeline.

The classifier is what the user reads when extraction cannot find source, and
its whole purpose is to distinguish user error (wrong ``--source-dir``, wrong
``--language``) from a code object that has no source trail at all. Getting
that wrong sends people hunting for a file that does not exist, so the three
branches are pinned separately here.

``llvm-dwarfdump`` / ``llvm-readelf`` are stubbed at their call sites, so no
ROCm and no GPU are needed.
"""

import logging
import os
import subprocess
from unittest.mock import patch

import pytest

from kerncap.extract import (
    _explain_source_not_found,
    _generate_hsaco,
    _generate_reproducer,
    _generate_triton,
    _generate_triton_from_hsa,
    _hsaco_has_debug_section,
    _llvm_dwarfdump_paths,
    _triton_kernel_in_tree,
    _TritonHsaReproducerUnavailable,
)


@pytest.fixture(autouse=True)
def allow_log_propagation():
    """caplog reads through the root logger, so propagation must be on.

    ``cli._setup_logging`` sets ``propagate = False`` on the ``kerncap``
    logger; this keeps these tests independent of whether a CLI test ran
    first.
    """
    log = logging.getLogger("kerncap")
    saved = log.propagate
    log.propagate = True
    yield
    log.propagate = saved


def dwarfdump_stub(stdout="", returncode=0, fail_with=None):
    """Stand in for the dwarfdump/readelf subprocess, recording the tools tried."""
    tried = []

    def _run(argv, **_kwargs):
        tried.append(argv[0])
        if fail_with is not None:
            raise fail_with
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")

    _run.tried = tried
    return _run


def dwarf_output(*names):
    """Render dwarfdump --debug-line output containing the given name entries."""
    return "\n".join(f'  name: "{n}"' for n in names)


@pytest.fixture
def hsaco(tmp_path):
    path = tmp_path / "kernel.hsaco"
    path.write_bytes(b"\x7fELF")
    return str(path)


@pytest.fixture
def runner():
    """Only used to capture click's echo output when driving _print_next_steps."""
    from click.testing import CliRunner

    return CliRunner()


def kernel_source(**overrides):
    """A stand-in for ``KernelSource`` carrying only what extract.py reads."""
    fields = {
        "main_file": "attn.py",
        "language": "triton",
        "source_files": [],
        "translation_unit": None,
        "compile_command": "hipcc -c attn.hip",
    }
    fields.update(overrides)
    return type("FakeKernelSource", (), fields)()


# --------------------------------------------------------------------------
# _llvm_dwarfdump_paths
# --------------------------------------------------------------------------


class TestLlvmDwarfdumpPaths:
    def test_returns_none_when_no_tool_can_run(self, hsaco):
        """None means 'could not run', which is distinct from 'found nothing'."""
        with patch("subprocess.run", dwarfdump_stub(fail_with=FileNotFoundError())):
            assert _llvm_dwarfdump_paths(hsaco) is None

    def test_returns_none_when_every_tool_exits_nonzero(self, hsaco):
        with patch("subprocess.run", dwarfdump_stub(returncode=1)):
            assert _llvm_dwarfdump_paths(hsaco) is None

    def test_returns_empty_list_when_there_is_no_debug_info(self, hsaco):
        """[] means 'ran fine, no source trail' — the TYPE B signal."""
        with patch("subprocess.run", dwarfdump_stub(stdout="no line table")):
            assert _llvm_dwarfdump_paths(hsaco) == []

    def test_extracts_source_paths(self, hsaco):
        out = dwarf_output("/src/kernel.hip", "/src/util.cuh")
        with patch("subprocess.run", dwarfdump_stub(stdout=out)):
            assert _llvm_dwarfdump_paths(hsaco) == ["/src/kernel.hip", "/src/util.cuh"]

    def test_filters_synthetic_entries(self, hsaco):
        """The AMDGPU back-end emits <built-in> and friends; they aren't source."""
        out = dwarf_output("<built-in>", "<command line>", "<unknown>", "/src/real.cpp")
        with patch("subprocess.run", dwarfdump_stub(stdout=out)):
            assert _llvm_dwarfdump_paths(hsaco) == ["/src/real.cpp"]

    def test_filters_non_source_extensions(self, hsaco):
        out = dwarf_output("/src/kernel.cpp", "/src/notes.txt", "/src/data.bin")
        with patch("subprocess.run", dwarfdump_stub(stdout=out)):
            assert _llvm_dwarfdump_paths(hsaco) == ["/src/kernel.cpp"]

    @pytest.mark.parametrize(
        "ext", [".cpp", ".hip", ".cu", ".cxx", ".cc", ".hpp", ".h", ".cuh", ".c"]
    )
    def test_every_documented_source_extension_is_accepted(self, hsaco, ext):
        with patch("subprocess.run", dwarfdump_stub(stdout=dwarf_output(f"/src/k{ext}"))):
            assert _llvm_dwarfdump_paths(hsaco) == [f"/src/k{ext}"]

    def test_deduplicates_while_preserving_order(self, hsaco):
        """A line table repeats the same file once per line; report it once."""
        out = dwarf_output("/src/b.cpp", "/src/a.cpp", "/src/b.cpp", "/src/a.cpp")
        with patch("subprocess.run", dwarfdump_stub(stdout=out)):
            assert _llvm_dwarfdump_paths(hsaco) == ["/src/b.cpp", "/src/a.cpp"]

    def test_prefers_the_rocm_path_tool(self, hsaco, monkeypatch):
        monkeypatch.setenv("ROCM_PATH", "/custom/rocm")
        stub = dwarfdump_stub(stdout=dwarf_output("/src/k.cpp"))
        with patch("subprocess.run", stub):
            _llvm_dwarfdump_paths(hsaco)

        assert stub.tried[0] == "/custom/rocm/llvm/bin/llvm-dwarfdump"

    def test_defaults_to_opt_rocm(self, hsaco, monkeypatch):
        monkeypatch.delenv("ROCM_PATH", raising=False)
        stub = dwarfdump_stub(stdout="")
        with patch("subprocess.run", stub):
            _llvm_dwarfdump_paths(hsaco)

        assert stub.tried[0] == "/opt/rocm/llvm/bin/llvm-dwarfdump"

    def test_falls_through_to_a_later_candidate(self, hsaco):
        """A missing ROCm tool must not stop the PATH lookup."""
        calls = []

        def _run(argv, **_kwargs):
            calls.append(argv[0])
            if len(calls) < 3:
                raise FileNotFoundError()
            return subprocess.CompletedProcess(
                argv, 0, stdout=dwarf_output("/src/k.cpp"), stderr=""
            )

        with patch("subprocess.run", _run):
            assert _llvm_dwarfdump_paths(hsaco) == ["/src/k.cpp"]
        assert calls[-1] == "llvm-dwarfdump"

    @pytest.mark.parametrize(
        "exc", [subprocess.TimeoutExpired("llvm-dwarfdump", 15), OSError("bad exec")]
    )
    def test_tool_exceptions_are_swallowed(self, hsaco, exc):
        with patch("subprocess.run", dwarfdump_stub(fail_with=exc)):
            assert _llvm_dwarfdump_paths(hsaco) is None

    def test_passes_debug_line_and_a_timeout(self, hsaco):
        seen = {}

        def _run(argv, **kwargs):
            seen["argv"] = list(argv)
            seen["timeout"] = kwargs.get("timeout")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        with patch("subprocess.run", _run):
            _llvm_dwarfdump_paths(hsaco)

        assert "--debug-line" in seen["argv"]
        assert seen["argv"][-1] == hsaco
        assert seen["timeout"] == 15


# --------------------------------------------------------------------------
# _hsaco_has_debug_section
# --------------------------------------------------------------------------


class TestHsacoHasDebugSection:
    def test_true_when_debug_sections_present(self, hsaco):
        out = "  [ 5] .debug_info  PROGBITS"
        with patch("subprocess.run", dwarfdump_stub(stdout=out)):
            assert _hsaco_has_debug_section(hsaco) is True

    def test_false_when_absent(self, hsaco):
        with patch("subprocess.run", dwarfdump_stub(stdout="  [ 1] .text PROGBITS")):
            assert _hsaco_has_debug_section(hsaco) is False

    def test_none_when_no_tool_runs(self, hsaco):
        """None is 'could not tell', not 'no debug info'."""
        with patch("subprocess.run", dwarfdump_stub(fail_with=FileNotFoundError())):
            assert _hsaco_has_debug_section(hsaco) is None

    def test_none_when_every_tool_exits_nonzero(self, hsaco):
        with patch("subprocess.run", dwarfdump_stub(returncode=1)):
            assert _hsaco_has_debug_section(hsaco) is None

    def test_falls_back_to_system_readelf(self, hsaco):
        calls = []

        def _run(argv, **_kwargs):
            calls.append(argv[0])
            if argv[0] != "readelf":
                raise FileNotFoundError()
            return subprocess.CompletedProcess(argv, 0, stdout=".debug_line", stderr="")

        with patch("subprocess.run", _run):
            assert _hsaco_has_debug_section(hsaco) is True
        assert calls[-1] == "readelf"


# --------------------------------------------------------------------------
# _triton_kernel_in_tree
# --------------------------------------------------------------------------


def write_py(tmp_path, name: str, body: str) -> None:
    (tmp_path / name).write_text(body)


class TestTritonKernelInTree:
    def test_finds_a_jit_decorated_function(self, tmp_path):
        write_py(tmp_path, "k.py", "import triton\n@triton.jit\ndef my_kernel(x):\n    pass\n")
        assert _triton_kernel_in_tree("my_kernel", str(tmp_path)) is True

    def test_matches_a_kernel_name_substring(self, tmp_path):
        """Captured names are mangled, e.g. triton_poi_fused_my_kernel_0."""
        write_py(tmp_path, "k.py", "import triton\n@triton.jit\ndef my_kernel(x):\n    pass\n")
        assert _triton_kernel_in_tree("triton_poi_my_kernel_0", str(tmp_path)) is True

    def test_matches_in_the_other_direction_too(self, tmp_path):
        write_py(tmp_path, "k.py", "import triton\n@triton.jit\ndef attn_fwd_inner(x):\n    pass\n")
        assert _triton_kernel_in_tree("attn_fwd", str(tmp_path)) is True

    @pytest.mark.parametrize(
        "decorator",
        [
            "@triton.jit",  # ast.Attribute
            "@triton.autotune(configs=[], key=[])",  # ast.Call of ast.Attribute
        ],
    )
    def test_dotted_decorator_forms_are_recognised(self, tmp_path, decorator):
        write_py(tmp_path, "k.py", f"import triton\n{decorator}\ndef my_kernel(x):\n    pass\n")
        assert _triton_kernel_in_tree("my_kernel", str(tmp_path)) is True

    @pytest.mark.parametrize("decorator", ["@jit", "@autotune(configs=[], key=[])"])
    def test_bare_decorator_forms_are_missed_by_the_text_prefilter(self, tmp_path, decorator):
        """``from triton import jit`` + ``@jit`` is not detected.

        The AST walk handles ``ast.Name`` and ``ast.Call``-of-``ast.Name``
        decorators, but it is never reached: the function first requires the
        literal text ``@triton.jit`` or ``@triton.autotune`` to appear in the
        file, and a bare import form contains neither.

        Consequence is limited to diagnostics — this only drives the TYPE T
        "did you mean --language triton?" hint, so a miss degrades the message
        rather than breaking extraction. Pinned as current behaviour, not
        endorsed; the fix would be to widen the prefilter to ``"triton"``.
        """
        write_py(
            tmp_path,
            "k.py",
            f"from triton import jit, autotune\n{decorator}\ndef my_kernel(x):\n    pass\n",
        )
        assert _triton_kernel_in_tree("my_kernel", str(tmp_path)) is False

    @pytest.mark.parametrize("decorator", ["@jit", "@autotune(configs=[], key=[])"])
    def test_bare_decorator_matches_once_the_prefilter_is_satisfied(self, tmp_path, decorator):
        """The bare-name branches do work — it is only the gate that blocks them.

        Here another function carries the dotted form, so the file passes the
        text prefilter and the bare decorator on the target is then matched.
        """
        write_py(
            tmp_path,
            "k.py",
            "import triton\nfrom triton import jit, autotune\n\n"
            "@triton.jit\ndef unrelated():\n    pass\n\n"
            f"{decorator}\ndef my_kernel(x):\n    pass\n",
        )
        assert _triton_kernel_in_tree("my_kernel", str(tmp_path)) is True

    def test_searches_recursively(self, tmp_path):
        nested = tmp_path / "pkg" / "kernels"
        nested.mkdir(parents=True)
        (nested / "k.py").write_text("import triton\n@triton.jit\ndef my_kernel():\n    pass\n")
        assert _triton_kernel_in_tree("my_kernel", str(tmp_path)) is True

    def test_non_python_files_are_ignored(self, tmp_path):
        (tmp_path / "k.txt").write_text("@triton.jit\ndef my_kernel(): pass\n")
        assert _triton_kernel_in_tree("my_kernel", str(tmp_path)) is False

    def test_files_without_triton_markers_are_skipped(self, tmp_path):
        write_py(tmp_path, "k.py", "def my_kernel(x):\n    pass\n")
        assert _triton_kernel_in_tree("my_kernel", str(tmp_path)) is False

    def test_undecorated_function_does_not_match(self, tmp_path):
        """The marker text is present but the target function is plain."""
        write_py(
            tmp_path,
            "k.py",
            "import triton\n@triton.jit\ndef other(x):\n    pass\n\ndef my_kernel(y):\n    pass\n",
        )
        assert _triton_kernel_in_tree("my_kernel", str(tmp_path)) is False

    def test_an_unparseable_file_is_skipped(self, tmp_path):
        """Sole file in the tree, so the SyntaxError branch must be taken.

        Deliberately not paired with a good file: ``os.walk`` yields entries
        in filesystem order, so a two-file tree only reaches ``ast.parse`` on
        the broken one when the OS happens to list it first. That made an
        earlier version of this test cover the branch on one machine and not
        another.
        """
        write_py(tmp_path, "broken.py", "@triton.jit\ndef ( bad syntax\n")
        assert _triton_kernel_in_tree("my_kernel", str(tmp_path)) is False

    def test_a_syntax_error_does_not_hide_a_later_match(self, tmp_path):
        """Behavioural pair to the above, independent of walk order."""
        write_py(tmp_path, "broken.py", "@triton.jit\ndef ( bad syntax\n")
        write_py(tmp_path, "good.py", "import triton\n@triton.jit\ndef my_kernel():\n    pass\n")
        assert _triton_kernel_in_tree("my_kernel", str(tmp_path)) is True

    def test_an_unreadable_file_is_skipped(self, tmp_path):
        """Sole file again, for the same walk-order reason."""
        write_py(tmp_path, "bin.py", "@triton.jit\ndef my_kernel():\n    pass\n")

        real_open = open

        def _open(path, *args, **kwargs):
            if str(path).endswith("bin.py"):
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", _open):
            assert _triton_kernel_in_tree("my_kernel", str(tmp_path)) is False

    def test_an_unreadable_file_does_not_hide_a_later_match(self, tmp_path):
        write_py(tmp_path, "bin.py", "@triton.jit\n")
        write_py(tmp_path, "good.py", "import triton\n@triton.jit\ndef my_kernel():\n    pass\n")

        real_open = open

        def _open(path, *args, **kwargs):
            if str(path).endswith("bin.py"):
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", _open):
            assert _triton_kernel_in_tree("my_kernel", str(tmp_path)) is True

    def test_empty_tree_returns_false(self, tmp_path):
        assert _triton_kernel_in_tree("my_kernel", str(tmp_path)) is False


# --------------------------------------------------------------------------
# _explain_source_not_found — the three classifications
# --------------------------------------------------------------------------


class TestExplainSourceNotFound:
    def test_missing_hsaco_cannot_be_classified(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            _explain_source_not_found("k", str(tmp_path), None, None)

        assert "cannot classify" in caplog.text

    def test_type_a_reports_the_dwarf_paths(self, tmp_path, caplog):
        """Source trail exists — the user pointed --source-dir somewhere wrong."""
        (tmp_path / "kernel.hsaco").write_bytes(b"\x7fELF")
        paths = ["/real/src/kernel.hip", "/real/src/util.cuh"]

        with (
            patch("kerncap.extract._llvm_dwarfdump_paths", return_value=paths),
            caplog.at_level(logging.WARNING),
        ):
            _explain_source_not_found("k", str(tmp_path), "/wrong/dir", "hip")

        assert "a source trail DOES exist" in caplog.text
        assert "/real/src/kernel.hip" in caplog.text

    def test_type_a_lists_at_most_three_paths(self, tmp_path, caplog):
        (tmp_path / "kernel.hsaco").write_bytes(b"\x7fELF")
        paths = [f"/src/f{i}.cpp" for i in range(10)]

        with (
            patch("kerncap.extract._llvm_dwarfdump_paths", return_value=paths),
            caplog.at_level(logging.WARNING),
        ):
            _explain_source_not_found("k", str(tmp_path), "/wrong", None)

        assert "/src/f2.cpp" in caplog.text
        assert "/src/f3.cpp" not in caplog.text

    def test_type_t_suggests_the_triton_language(self, tmp_path, caplog):
        (tmp_path / "kernel.hsaco").write_bytes(b"\x7fELF")

        with (
            patch("kerncap.extract._llvm_dwarfdump_paths", return_value=[]),
            patch("kerncap.extract._triton_kernel_in_tree", return_value=True),
            caplog.at_level(logging.WARNING),
        ):
            _explain_source_not_found("my_kernel", str(tmp_path), "/src", "hip")

        assert "Re-run with --language triton" in caplog.text

    def test_type_t_applies_when_language_was_omitted(self, tmp_path, caplog):
        (tmp_path / "kernel.hsaco").write_bytes(b"\x7fELF")

        with (
            patch("kerncap.extract._llvm_dwarfdump_paths", return_value=[]),
            patch("kerncap.extract._triton_kernel_in_tree", return_value=True),
            caplog.at_level(logging.WARNING),
        ):
            _explain_source_not_found("my_kernel", str(tmp_path), "/src", None)

        assert "--language triton" in caplog.text

    def test_type_t_is_not_offered_when_already_triton(self, tmp_path, caplog):
        """Suggesting --language triton to someone who passed it is noise."""
        (tmp_path / "kernel.hsaco").write_bytes(b"\x7fELF")

        with (
            patch("kerncap.extract._llvm_dwarfdump_paths", return_value=[]),
            patch("kerncap.extract._triton_kernel_in_tree", return_value=True),
            patch("kerncap.extract._hsaco_has_debug_section", return_value=None),
            caplog.at_level(logging.WARNING),
        ):
            _explain_source_not_found("my_kernel", str(tmp_path), "/src", "triton")

        assert "Re-run with --language triton" not in caplog.text
        assert "no source trail to walk" in caplog.text

    def test_type_t_needs_a_source_dir(self, tmp_path, caplog):
        (tmp_path / "kernel.hsaco").write_bytes(b"\x7fELF")

        with (
            patch("kerncap.extract._llvm_dwarfdump_paths", return_value=[]),
            patch("kerncap.extract._triton_kernel_in_tree", return_value=True) as scan,
            patch("kerncap.extract._hsaco_has_debug_section", return_value=None),
            caplog.at_level(logging.WARNING),
        ):
            _explain_source_not_found("my_kernel", str(tmp_path), None, "hip")

        scan.assert_not_called()
        assert "no source trail to walk" in caplog.text

    def test_type_b_when_no_trail_exists(self, tmp_path, caplog):
        """Tensile / hand-written assembly / vendor blobs land here."""
        (tmp_path / "kernel.hsaco").write_bytes(b"\x7fELF")

        with (
            patch("kerncap.extract._llvm_dwarfdump_paths", return_value=None),
            patch("kerncap.extract._hsaco_has_debug_section", return_value=None),
            caplog.at_level(logging.WARNING),
        ):
            _explain_source_not_found("Cijk_Ailk_Bljk", str(tmp_path), "/src", "hip")

        assert "no source trail to walk" in caplog.text
        assert "Tensile" in caplog.text
        assert "make recompile' is not possible" in caplog.text

    def test_type_b_notes_a_stripped_binary(self, tmp_path, caplog):
        (tmp_path / "kernel.hsaco").write_bytes(b"\x7fELF")

        with (
            patch("kerncap.extract._llvm_dwarfdump_paths", return_value=[]),
            patch("kerncap.extract._triton_kernel_in_tree", return_value=False),
            patch("kerncap.extract._hsaco_has_debug_section", return_value=True),
            caplog.at_level(logging.WARNING),
        ):
            _explain_source_not_found("k", str(tmp_path), "/src", "hip")

        assert "built with -g but stripped" in caplog.text

    def test_type_b_notes_an_undebuggable_binary(self, tmp_path, caplog):
        (tmp_path / "kernel.hsaco").write_bytes(b"\x7fELF")

        with (
            patch("kerncap.extract._llvm_dwarfdump_paths", return_value=[]),
            patch("kerncap.extract._triton_kernel_in_tree", return_value=False),
            patch("kerncap.extract._hsaco_has_debug_section", return_value=False),
            caplog.at_level(logging.WARNING),
        ):
            _explain_source_not_found("k", str(tmp_path), "/src", "hip")

        assert "no .debug_* sections" in caplog.text


# --------------------------------------------------------------------------
# _generate_triton_from_hsa — source resolution chain
# --------------------------------------------------------------------------


class TestGenerateTritonFromHsa:
    def test_raises_when_there_is_no_source_anywhere(self, tmp_path):
        """No --source-dir, no snapshot, no name_map — nothing to author from."""
        capture = tmp_path / "capture"
        capture.mkdir()

        with pytest.raises(_TritonHsaReproducerUnavailable, match="no source directory"):
            _generate_triton_from_hsa("k", str(capture), str(tmp_path / "out"), None, None, {})

    def test_falls_back_to_the_compile_shim_snapshot(self, tmp_path):
        """capture/triton_sources/ stands in when --source-dir was omitted."""
        capture = tmp_path / "capture"
        (capture / "triton_sources").mkdir(parents=True)

        with patch("kerncap.source_finder.find_kernel_source", return_value=None) as find:
            with pytest.raises(_TritonHsaReproducerUnavailable, match="not found under"):
                _generate_triton_from_hsa("k", str(capture), str(tmp_path / "out"), None, None, {})

        assert find.call_args.kwargs["source_dir"] == str(capture / "triton_sources")

    def test_explicit_source_dir_wins_over_the_snapshot(self, tmp_path):
        capture = tmp_path / "capture"
        (capture / "triton_sources").mkdir(parents=True)

        with patch("kerncap.source_finder.find_kernel_source", return_value=None) as find:
            with pytest.raises(_TritonHsaReproducerUnavailable):
                _generate_triton_from_hsa(
                    "k", str(capture), str(tmp_path / "out"), "/my/src", None, {}
                )

        assert find.call_args.kwargs["source_dir"] == "/my/src"

    def test_uses_the_source_file_recorded_in_name_map(self, tmp_path):
        """name_map.json names the @triton.jit function's actual file."""
        capture = tmp_path / "capture"
        capture.mkdir()
        recorded = tmp_path / "real_src" / "attn.py"
        recorded.parent.mkdir()
        recorded.write_text("# kernel\n")
        (capture / "name_map.json").write_text(
            f'[{{"user_name": "attn_fwd", "source_file": "{recorded}"}}]'
        )

        with patch("kerncap.source_finder.find_kernel_source", return_value=None) as find:
            with pytest.raises(_TritonHsaReproducerUnavailable):
                _generate_triton_from_hsa(
                    "attn_fwd", str(capture), str(tmp_path / "out"), None, None, {}
                )

        assert find.call_args.kwargs["source_dir"] == str(recorded.parent)

    def test_source_snapshot_is_preferred_over_source_file(self, tmp_path):
        capture = tmp_path / "capture"
        capture.mkdir()
        snap = tmp_path / "snap" / "attn.py"
        snap.parent.mkdir()
        snap.write_text("# snapshot\n")
        (capture / "name_map.json").write_text(
            f'[{{"user_name": "attn_fwd", "source_file": "/orig/attn.py",'
            f' "source_snapshot": "{snap}"}}]'
        )

        with patch("kerncap.source_finder.find_kernel_source", return_value=None) as find:
            with pytest.raises(_TritonHsaReproducerUnavailable):
                _generate_triton_from_hsa(
                    "attn_fwd", str(capture), str(tmp_path / "out"), None, None, {}
                )

        assert find.call_args.kwargs["source_dir"] == str(snap.parent)

    def test_malformed_name_map_is_tolerated(self, tmp_path):
        """A corrupt name_map must not crash extraction outright."""
        capture = tmp_path / "capture"
        capture.mkdir()
        (capture / "name_map.json").write_text("{not json")

        with pytest.raises(_TritonHsaReproducerUnavailable, match="no source directory"):
            _generate_triton_from_hsa("k", str(capture), str(tmp_path / "out"), None, None, {})

    def test_missing_name_map_is_reported_distinctly(self, tmp_path):
        """Source was found but the compile shim never fired — a cache hit."""
        capture = tmp_path / "capture"
        capture.mkdir()
        src = tmp_path / "src"
        src.mkdir()

        fake_src = kernel_source()

        with patch("kerncap.source_finder.find_kernel_source", return_value=fake_src):
            with pytest.raises(_TritonHsaReproducerUnavailable, match="no name_map.json"):
                _generate_triton_from_hsa(
                    "k", str(capture), str(tmp_path / "out"), str(src), None, {}
                )

    def test_success_reports_kernel_variant_when_written(self, tmp_path):
        """Package-source path: the reproducer writes a clean kernel_variant.py."""
        capture = tmp_path / "capture"
        capture.mkdir()
        (capture / "name_map.json").write_text('[{"user_name": "k"}]')
        out = tmp_path / "out"
        out.mkdir()
        (out / "kernel_variant.py").write_text("# variant\n")
        src = tmp_path / "src"
        src.mkdir()

        with (
            patch("kerncap.source_finder.find_kernel_source", return_value=kernel_source()),
            patch("kerncap.reproducer.generate_triton_hsa_reproducer") as gen,
        ):
            result = _generate_triton_from_hsa("k", str(capture), str(out), str(src), None, {})

        gen.assert_called_once()
        assert result.language == "triton"
        assert result.has_source is True
        assert result.generated_files == [
            "reproducer.py",
            "kernel_variant.py",
            "capture/",
            "reference_output/",
        ]

    def test_success_lists_copied_sources_on_the_flat_file_path(self, tmp_path):
        """Without kernel_variant.py the list must name what was really copied.

        A hardcoded list would lie here — the flat-file path copies the user's
        own source files instead of authoring a variant.
        """
        capture = tmp_path / "capture"
        capture.mkdir()
        (capture / "name_map.json").write_text('[{"user_name": "k"}]')
        out = tmp_path / "out"
        out.mkdir()
        (out / "attn.py").write_text("# copied\n")
        (out / "helpers.py").write_text("# copied\n")
        src = tmp_path / "src"
        src.mkdir()

        ks = kernel_source(source_files=["/orig/attn.py", "/orig/helpers.py", "/orig/absent.py"])

        with (
            patch("kerncap.source_finder.find_kernel_source", return_value=ks),
            patch("kerncap.reproducer.generate_triton_hsa_reproducer"),
        ):
            result = _generate_triton_from_hsa("k", str(capture), str(out), str(src), None, {})

        # Only files that actually landed in output_dir are listed.
        assert result.generated_files == [
            "reproducer.py",
            "attn.py",
            "helpers.py",
            "capture/",
            "reference_output/",
        ]


# --------------------------------------------------------------------------
# _generate_triton (legacy python backend)
# --------------------------------------------------------------------------


class TestGenerateTriton:
    def test_requires_a_source_dir(self, tmp_path):
        with pytest.raises(RuntimeError, match="requires located kernel source"):
            _generate_triton("k", str(tmp_path), str(tmp_path / "out"), None, None)

    def test_unlocatable_source_explains_before_raising(self, tmp_path, caplog):
        """The user gets the classifier message, not just a bare RuntimeError."""
        capture = tmp_path / "capture"
        capture.mkdir()

        with (
            patch("kerncap.source_finder.find_kernel_source", return_value=None),
            patch("kerncap.extract._explain_source_not_found") as explain,
            pytest.raises(RuntimeError, match="requires located kernel source"),
        ):
            _generate_triton("k", str(capture), str(tmp_path / "out"), str(tmp_path), None)

        explain.assert_called_once()

    def test_success(self, tmp_path):
        with (
            patch("kerncap.source_finder.find_kernel_source", return_value=kernel_source()),
            patch("kerncap.reproducer.generate_triton_reproducer") as gen,
        ):
            result = _generate_triton(
                "k", str(tmp_path), str(tmp_path / "out"), str(tmp_path), "triton"
            )

        gen.assert_called_once()
        assert result.language == "triton"
        assert result.has_source is True
        assert result.generated_files == ["reproducer.py", "capture/"]


# --------------------------------------------------------------------------
# _generate_hsaco
# --------------------------------------------------------------------------


class TestGenerateHsaco:
    def test_warns_when_the_code_object_is_missing(self, tmp_path, caplog):
        """Without a .hsaco the reproducer cannot replay at all."""
        capture = tmp_path / "capture"
        capture.mkdir()

        with (
            patch("kerncap.reproducer.generate_hsaco_reproducer"),
            caplog.at_level(logging.WARNING),
        ):
            _generate_hsaco("k", str(capture), str(tmp_path / "out"), None, None, [], {})

        assert "Replay will not work" in caplog.text

    def test_without_source_dir_the_result_is_hsaco_only(self, tmp_path):
        capture = tmp_path / "capture"
        capture.mkdir()
        (capture / "kernel.hsaco").write_bytes(b"\x7fELF")

        with patch("kerncap.reproducer.generate_hsaco_reproducer"):
            result = _generate_hsaco("k", str(capture), str(tmp_path / "out"), None, None, [], {})

        assert result.has_source is False
        assert result.generated_files == ["capture/", "Makefile"]
        assert result.language == "hip"

    def test_language_is_echoed_when_given(self, tmp_path):
        capture = tmp_path / "capture"
        capture.mkdir()
        (capture / "kernel.hsaco").write_bytes(b"\x7fELF")

        with patch("kerncap.reproducer.generate_hsaco_reproducer"):
            result = _generate_hsaco(
                "k", str(capture), str(tmp_path / "out"), None, "triton", [], {}
            )

        assert result.language == "triton"

    def test_lists_the_copied_hsaco(self, tmp_path):
        capture = tmp_path / "capture"
        capture.mkdir()
        (capture / "kernel.hsaco").write_bytes(b"\x7fELF")
        out = tmp_path / "out"
        (out / "capture").mkdir(parents=True)
        (out / "capture" / "kernel.hsaco").write_bytes(b"\x7fELF")

        with patch("kerncap.reproducer.generate_hsaco_reproducer"):
            result = _generate_hsaco("k", str(capture), str(out), None, None, [], {})

        assert "kernel.hsaco" in result.generated_files

    def test_located_source_adds_the_editable_files(self, tmp_path):
        capture = tmp_path / "capture"
        capture.mkdir()
        (capture / "kernel.hsaco").write_bytes(b"\x7fELF")

        ks = kernel_source(language="hip", main_file="k.hip", translation_unit="k.cu")

        with (
            patch("kerncap.source_finder.find_kernel_source", return_value=ks),
            patch("kerncap.reproducer.generate_hsaco_reproducer"),
        ):
            result = _generate_hsaco(
                "k", str(capture), str(tmp_path / "out"), str(tmp_path), "hip", [], {}
            )

        assert result.has_source is True
        assert "kernel_variant.cpp" in result.generated_files
        assert "vfs.yaml" in result.generated_files

    def test_forwards_defines_and_mangled_name(self, tmp_path):
        """Both are needed to pick the right templated instantiation."""
        capture = tmp_path / "capture"
        capture.mkdir()
        (capture / "kernel.hsaco").write_bytes(b"\x7fELF")

        with (
            patch("kerncap.source_finder.find_kernel_source", return_value=kernel_source()) as find,
            patch("kerncap.reproducer.generate_hsaco_reproducer"),
        ):
            _generate_hsaco(
                "k",
                str(capture),
                str(tmp_path / "out"),
                str(tmp_path),
                "hip",
                ["GGML_USE_HIP"],
                {"mangled_name": "_Z3mulIiEvPT_"},
            )

        assert find.call_args.kwargs["extra_defines"] == ["GGML_USE_HIP"]
        assert find.call_args.kwargs["mangled_name"] == "_Z3mulIiEvPT_"

    def test_empty_defines_become_none(self, tmp_path):
        capture = tmp_path / "capture"
        capture.mkdir()
        (capture / "kernel.hsaco").write_bytes(b"\x7fELF")

        with (
            patch("kerncap.source_finder.find_kernel_source", return_value=kernel_source()) as find,
            patch("kerncap.reproducer.generate_hsaco_reproducer"),
        ):
            _generate_hsaco("k", str(capture), str(tmp_path / "out"), str(tmp_path), "hip", [], {})

        assert find.call_args.kwargs["extra_defines"] is None

    def test_warns_when_no_compile_command_for_a_hip_source(self, tmp_path, caplog):
        """Without a compile command there is no 'make recompile' target."""
        capture = tmp_path / "capture"
        capture.mkdir()
        (capture / "kernel.hsaco").write_bytes(b"\x7fELF")
        ks = kernel_source(language="hip", compile_command=None)

        with (
            patch("kerncap.source_finder.find_kernel_source", return_value=ks),
            patch("kerncap.reproducer.generate_hsaco_reproducer"),
            caplog.at_level(logging.WARNING),
        ):
            _generate_hsaco("k", str(capture), str(tmp_path / "out"), str(tmp_path), "hip", [], {})

        assert "make recompile' target will not be available" in caplog.text

    def test_no_recompile_warning_for_triton(self, tmp_path, caplog):
        """Triton is JIT-compiled, so a missing compile command is expected."""
        capture = tmp_path / "capture"
        capture.mkdir()
        (capture / "kernel.hsaco").write_bytes(b"\x7fELF")
        ks = kernel_source(language="triton", compile_command=None)

        with (
            patch("kerncap.source_finder.find_kernel_source", return_value=ks),
            patch("kerncap.reproducer.generate_hsaco_reproducer"),
            caplog.at_level(logging.WARNING),
        ):
            _generate_hsaco(
                "k", str(capture), str(tmp_path / "out"), str(tmp_path), "triton", [], {}
            )

        assert "make recompile" not in caplog.text

    def test_unlocatable_source_is_explained(self, tmp_path):
        capture = tmp_path / "capture"
        capture.mkdir()
        (capture / "kernel.hsaco").write_bytes(b"\x7fELF")

        with (
            patch("kerncap.source_finder.find_kernel_source", return_value=None),
            patch("kerncap.extract._explain_source_not_found") as explain,
            patch("kerncap.reproducer.generate_hsaco_reproducer"),
        ):
            result = _generate_hsaco(
                "k", str(capture), str(tmp_path / "out"), str(tmp_path), "hip", [], {}
            )

        explain.assert_called_once()
        assert result.has_source is False


# --------------------------------------------------------------------------
# _generate_reproducer — routing and the HSA-Triton fallback
# --------------------------------------------------------------------------


def write_capture(tmp_path, metadata: dict, filename="dispatch.json") -> str:
    import json as _json

    capture = tmp_path / "capture"
    capture.mkdir(exist_ok=True)
    (capture / filename).write_text(_json.dumps(metadata))
    (capture / "kernel.hsaco").write_bytes(b"\x7fELF")
    return str(capture)


class TestGenerateReproducerRouting:
    def test_requires_dispatch_or_metadata_json(self, tmp_path):
        capture = tmp_path / "capture"
        capture.mkdir()

        with pytest.raises(FileNotFoundError, match="No dispatch.json or metadata.json"):
            _generate_reproducer("k", str(capture), str(tmp_path / "out"), None, None, [])

    def test_falls_back_to_metadata_json(self, tmp_path):
        capture = write_capture(tmp_path, {"language": "hip"}, filename="metadata.json")

        with patch("kerncap.reproducer.generate_hsaco_reproducer"):
            result = _generate_reproducer("k", capture, str(tmp_path / "out"), None, None, [])

        assert result.language == "hip"

    def test_language_from_metadata_routes_to_triton(self, tmp_path):
        capture = write_capture(tmp_path, {"language": "triton"})

        with patch("kerncap.extract._generate_triton_from_hsa") as gen:
            _generate_reproducer("k", capture, str(tmp_path / "out"), None, None, [])

        gen.assert_called_once()

    def test_explicit_language_overrides_metadata(self, tmp_path):
        capture = write_capture(tmp_path, {"language": "triton"})

        with patch("kerncap.reproducer.generate_hsaco_reproducer") as gen:
            _generate_reproducer("k", capture, str(tmp_path / "out"), None, "hip", [])

        gen.assert_called_once()

    def test_python_backend_routes_to_legacy_triton(self, tmp_path):
        capture = write_capture(tmp_path, {"language": "triton"})

        with patch("kerncap.extract._generate_triton") as gen:
            _generate_reproducer(
                "k", capture, str(tmp_path / "out"), None, None, [], triton_backend="python"
            )

        gen.assert_called_once()

    def test_unavailable_triton_reproducer_falls_back_to_the_hip_harness(self, tmp_path, caplog):
        """Losing the edit loop is better than producing nothing replayable."""
        capture = write_capture(tmp_path, {"language": "triton"})

        with (
            patch(
                "kerncap.extract._generate_triton_from_hsa",
                side_effect=_TritonHsaReproducerUnavailable("no name_map.json"),
            ),
            patch("kerncap.reproducer.generate_hsaco_reproducer") as gen,
            caplog.at_level(logging.WARNING),
        ):
            result = _generate_reproducer("k", capture, str(tmp_path / "out"), None, None, [])

        gen.assert_called_once()
        assert "falling back to HIP harness" in caplog.text
        assert "no name_map.json" in caplog.text

    @pytest.mark.parametrize("user_language", [None, "triton"])
    def test_the_fallback_is_always_labelled_hip(self, tmp_path, user_language):
        """The fallback produces an HSACO-only HIP harness, so it says "hip".

        ``ExtractResult.language`` drives how the CLI tells the user to *use*
        the reproducer, and this path writes no ``reproducer.py`` — so the
        label has to describe what was generated, not what the kernel was
        written in.

        Parametrised over both ways the language can be determined because
        that used to change the answer: ``_generate_hsaco`` returns
        ``language or "hip"``, so an explicit ``--language triton`` came back
        as "triton" while the same capture auto-detected from metadata came
        back as "hip".
        """
        capture = write_capture(tmp_path, {"language": "triton"})

        with (
            patch(
                "kerncap.extract._generate_triton_from_hsa",
                side_effect=_TritonHsaReproducerUnavailable("no name_map.json"),
            ),
            patch("kerncap.reproducer.generate_hsaco_reproducer"),
        ):
            result = _generate_reproducer(
                "k", capture, str(tmp_path / "out"), None, user_language, []
            )

        assert result.language == "hip"

    def test_the_fallback_tells_the_user_to_use_make_not_reproducer_py(self, tmp_path, runner):
        """End to end: the label must not produce instructions for a missing file.

        This is the reason the label matters. With ``language="triton"`` the
        CLI would print ``python3 reproducer.py`` for a directory that has no
        such file.
        """
        from kerncap import cli

        capture = write_capture(tmp_path, {"language": "triton"})
        out = tmp_path / "out"

        with (
            patch(
                "kerncap.extract._generate_triton_from_hsa",
                side_effect=_TritonHsaReproducerUnavailable("no name_map.json"),
            ),
            patch("kerncap.reproducer.generate_hsaco_reproducer"),
        ):
            result = _generate_reproducer("k", capture, str(out), None, "triton", [])

        with runner.isolation() as streams:
            cli._print_next_steps(result)
            # Read inside the block: Click closes these streams on exit,
            # so getvalue() afterwards raises "I/O operation on closed file".
            printed = streams[0].getvalue().decode()

        assert "make recompile" in printed
        assert "reproducer.py" not in printed

    def test_a_successful_triton_reproducer_is_still_labelled_triton(self, tmp_path):
        """The normalisation must not leak into the path that does write one."""
        capture = write_capture(tmp_path, {"language": "triton"})
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "kernel_variant.py").write_text("# variant\n")
        (tmp_path / "capture" / "name_map.json").write_text('[{"user_name": "k"}]')
        src = tmp_path / "src"
        src.mkdir()

        with (
            patch("kerncap.source_finder.find_kernel_source", return_value=kernel_source()),
            patch("kerncap.reproducer.generate_triton_hsa_reproducer"),
        ):
            result = _generate_reproducer("k", capture, str(tmp_path / "out"), str(src), None, [])

        assert result.language == "triton"
