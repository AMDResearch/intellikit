"""Unit tests for source_finder's compile_commands.json handling.

These are the functions that decide which preprocessor defines and include
paths a reproducer is rebuilt with. Getting them wrong does not fail loudly —
it produces a reproducer that compiles into something subtly different from
the kernel that was captured, which is the worst failure mode kerncap has.

All pure JSON and string parsing plus a filesystem walk, so nothing here needs
ROCm, a compiler, or a GPU.
"""

import json
import os

import pytest

from kerncap.source_finder import (
    _defines_from_compile_commands,
    _extract_defines_from_command,
    _extract_includes_from_command,
    _extract_output_path,
    _find_compile_commands,
    _includes_from_compile_commands,
    _match_tu_via_object_symbols,
    _nm_has_symbol,
)


def write_cc(directory, entries):
    """Write a compile_commands.json and return its path."""
    path = directory / "compile_commands.json"
    path.write_text(json.dumps(entries))
    return str(path)


def entry(file, command="", arguments=None, directory="/build"):
    e = {"file": file, "directory": directory}
    if command:
        e["command"] = command
    if arguments:
        e["arguments"] = arguments
    return e


# --------------------------------------------------------------------------
# _find_compile_commands
# --------------------------------------------------------------------------


class TestFindCompileCommands:
    def test_finds_it_beside_the_binary(self, tmp_path):
        cc = write_cc(tmp_path, [])
        binary = tmp_path / "my_app"
        binary.write_bytes(b"\x7fELF")

        assert _find_compile_commands(str(binary)) == cc

    def test_walks_up_the_tree(self, tmp_path):
        cc = write_cc(tmp_path, [])
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        binary = nested / "my_app"
        binary.write_bytes(b"\x7fELF")

        assert _find_compile_commands(str(binary)) == cc

    def test_stops_at_a_build_directory(self, tmp_path):
        """A build/ dir is the project boundary; do not escape into the parent.

        Walking past it could pick up an unrelated project's database and
        rebuild the kernel with the wrong flags.
        """
        write_cc(tmp_path, [])
        build = tmp_path / "build"
        build.mkdir()
        binary = build / "my_app"
        binary.write_bytes(b"\x7fELF")

        assert _find_compile_commands(str(binary)) is None

    def test_a_database_inside_the_build_dir_is_still_found(self, tmp_path):
        """The boundary check happens after the file check, not before."""
        build = tmp_path / "build"
        build.mkdir()
        cc = write_cc(build, [])
        binary = build / "my_app"
        binary.write_bytes(b"\x7fELF")

        assert _find_compile_commands(str(binary)) == cc

    def test_returns_none_when_absent(self, tmp_path):
        binary = tmp_path / "my_app"
        binary.write_bytes(b"\x7fELF")

        assert _find_compile_commands(str(binary)) is None


# --------------------------------------------------------------------------
# _extract_defines_from_command
# --------------------------------------------------------------------------


class TestExtractDefines:
    def test_attached_form(self):
        assert _extract_defines_from_command("hipcc -DFOO -DBAR=1 x.cu", []) == ["FOO", "BAR=1"]

    def test_separated_form(self):
        """``-D FOO`` consumes the next token."""
        assert _extract_defines_from_command("hipcc -D FOO -D BAR=2 x.cu", []) == ["FOO", "BAR=2"]

    def test_mixed_forms(self):
        assert _extract_defines_from_command("hipcc -DA -D B -DC x.cu", []) == ["A", "B", "C"]

    def test_arguments_win_over_command(self):
        """A database may carry either; ``arguments`` is the reliable one."""
        result = _extract_defines_from_command(
            "hipcc -DFROM_COMMAND x.cu", ["hipcc", "-DFROM_ARGS"]
        )
        assert result == ["FROM_ARGS"]

    def test_the_consumed_token_is_not_re_examined(self):
        """``-D -DFOO`` takes ``-DFOO`` as the value, not as another define."""
        assert _extract_defines_from_command("hipcc -D -DFOO x.cu", []) == ["-DFOO"]

    def test_trailing_bare_d_is_ignored(self):
        assert _extract_defines_from_command("hipcc x.cu -D", []) == []

    def test_no_defines(self):
        assert _extract_defines_from_command("hipcc -O3 x.cu", []) == []

    def test_empty_input(self):
        assert _extract_defines_from_command("", []) == []


# --------------------------------------------------------------------------
# _defines_from_compile_commands
# --------------------------------------------------------------------------


class TestDefinesFromCompileCommands:
    def test_no_database_returns_empty(self, tmp_path):
        binary = tmp_path / "app"
        binary.write_bytes(b"\x7f")

        assert _defines_from_compile_commands(str(binary), ["/src/k.cu"]) == []

    def test_matches_by_absolute_path(self, tmp_path):
        src = tmp_path / "kernels.cu"
        src.write_text("")
        write_cc(tmp_path, [entry(str(src), command="hipcc -DGGML_USE_HIP -c kernels.cu")])
        binary = tmp_path / "app"
        binary.write_bytes(b"\x7f")

        assert _defines_from_compile_commands(str(binary), [str(src)]) == ["GGML_USE_HIP"]

    def test_matches_by_basename(self, tmp_path):
        """The database may record a path that no longer resolves."""
        write_cc(tmp_path, [entry("/elsewhere/kernels.cu", command="hipcc -DFOO -c kernels.cu")])
        binary = tmp_path / "app"
        binary.write_bytes(b"\x7f")

        assert _defines_from_compile_commands(str(binary), ["/my/tree/kernels.cu"]) == ["FOO"]

    def test_relative_entry_files_resolve_against_directory(self, tmp_path):
        src = tmp_path / "sub" / "kernels.cu"
        src.parent.mkdir()
        src.write_text("")
        write_cc(
            tmp_path,
            [entry("kernels.cu", command="hipcc -DREL", directory=str(tmp_path / "sub"))],
        )
        binary = tmp_path / "app"
        binary.write_bytes(b"\x7f")

        assert _defines_from_compile_commands(str(binary), [str(src)]) == ["REL"]

    def test_the_first_matching_entry_wins(self, tmp_path):
        src = tmp_path / "k.cu"
        src.write_text("")
        write_cc(
            tmp_path,
            [
                entry(str(src), command="hipcc -DFIRST"),
                entry(str(src), command="hipcc -DSECOND"),
            ],
        )
        binary = tmp_path / "app"
        binary.write_bytes(b"\x7f")

        assert _defines_from_compile_commands(str(binary), [str(src)]) == ["FIRST"]

    def test_no_matching_entry_returns_empty(self, tmp_path):
        write_cc(tmp_path, [entry("/other/thing.cu", command="hipcc -DNOPE")])
        binary = tmp_path / "app"
        binary.write_bytes(b"\x7f")

        assert _defines_from_compile_commands(str(binary), ["/my/k.cu"]) == []

    @pytest.mark.parametrize("content", ["{not json", ""])
    def test_a_malformed_database_is_tolerated(self, tmp_path, content):
        """A broken database must degrade to 'no defines', not crash extraction."""
        (tmp_path / "compile_commands.json").write_text(content)
        binary = tmp_path / "app"
        binary.write_bytes(b"\x7f")

        assert _defines_from_compile_commands(str(binary), ["/my/k.cu"]) == []


# --------------------------------------------------------------------------
# _extract_includes_from_command
# --------------------------------------------------------------------------


class TestExtractIncludes:
    def test_attached_form_resolved_to_absolute(self, tmp_path):
        inc = tmp_path / "include"
        inc.mkdir()

        result = _extract_includes_from_command(f"hipcc -I{inc} x.cu", [], str(tmp_path))

        assert result == [str(inc)]

    def test_separated_form(self, tmp_path):
        inc = tmp_path / "include"
        inc.mkdir()

        result = _extract_includes_from_command(f"hipcc -I {inc} x.cu", [], str(tmp_path))

        assert result == [str(inc)]

    def test_relative_paths_resolve_against_the_working_dir(self, tmp_path):
        inc = tmp_path / "include"
        inc.mkdir()

        result = _extract_includes_from_command("hipcc -Iinclude x.cu", [], str(tmp_path))

        assert result == [str(inc)]

    def test_parent_relative_paths_are_normalised(self, tmp_path):
        inc = tmp_path / "include"
        inc.mkdir()
        sub = tmp_path / "build"
        sub.mkdir()

        result = _extract_includes_from_command("hipcc -I../include x.cu", [], str(sub))

        assert result == [str(inc)]

    def test_nonexistent_directories_are_dropped(self, tmp_path):
        """A stale -I would silently widen the include path on rebuild."""
        real = tmp_path / "real"
        real.mkdir()

        result = _extract_includes_from_command(
            f"hipcc -I{real} -I{tmp_path / 'gone'} x.cu", [], str(tmp_path)
        )

        assert result == [str(real)]

    def test_a_file_is_not_a_valid_include_dir(self, tmp_path):
        f = tmp_path / "notadir.h"
        f.write_text("")

        assert _extract_includes_from_command(f"hipcc -I{f} x.cu", [], str(tmp_path)) == []

    def test_arguments_win_over_command(self, tmp_path):
        inc = tmp_path / "from_args"
        inc.mkdir()

        result = _extract_includes_from_command(
            "hipcc -I/from/command x.cu", ["hipcc", f"-I{inc}"], str(tmp_path)
        )

        assert result == [str(inc)]

    def test_trailing_bare_i_is_ignored(self, tmp_path):
        assert _extract_includes_from_command("hipcc x.cu -I", [], str(tmp_path)) == []

    def test_the_consumed_token_is_not_re_examined(self, tmp_path):
        inc = tmp_path / "inc"
        inc.mkdir()

        result = _extract_includes_from_command(f"hipcc -I {inc} -O3", [], str(tmp_path))

        assert result == [str(inc)]

    def test_without_a_working_dir_a_relative_path_resolves_against_cwd(
        self, tmp_path, monkeypatch
    ):
        """With ``working_dir=""`` the path stays relative, so ``os.path.isdir``
        resolves it against the *process* working directory.

        Real callers pass the compile_commands entry's ``directory``, so this
        fallback does not bite in practice — but it means the function's result
        depends on where the process happens to be. Pinned in both directions
        below rather than left implicit.

        ``chdir`` is what makes this deterministic: without it the assertion
        silently depends on whether an ``include/`` directory happens to exist
        wherever pytest was invoked from.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "include").mkdir()

        assert _extract_includes_from_command("hipcc -Iinclude x.cu", [], "") == ["include"]

    def test_without_a_working_dir_an_unresolvable_path_is_dropped(self, tmp_path, monkeypatch):
        """Same call, a CWD with no matching directory: the path is dropped."""
        monkeypatch.chdir(tmp_path)

        assert _extract_includes_from_command("hipcc -Iinclude x.cu", [], "") == []


# --------------------------------------------------------------------------
# _includes_from_compile_commands
# --------------------------------------------------------------------------


class TestIncludesFromCompileCommands:
    def test_no_database_returns_empty(self, tmp_path):
        binary = tmp_path / "app"
        binary.write_bytes(b"\x7f")

        assert _includes_from_compile_commands(str(binary), ["/src/k.cu"]) == []

    def test_extracts_includes_for_a_matching_entry(self, tmp_path):
        src = tmp_path / "k.cu"
        src.write_text("")
        inc = tmp_path / "include"
        inc.mkdir()
        write_cc(
            tmp_path,
            [entry(str(src), command=f"hipcc -I{inc} -c k.cu", directory=str(tmp_path))],
        )
        binary = tmp_path / "app"
        binary.write_bytes(b"\x7f")

        assert _includes_from_compile_commands(str(binary), [str(src)]) == [str(inc)]

    def test_matches_by_basename(self, tmp_path):
        inc = tmp_path / "include"
        inc.mkdir()
        write_cc(
            tmp_path,
            [entry("/elsewhere/k.cu", command=f"hipcc -I{inc}", directory=str(tmp_path))],
        )
        binary = tmp_path / "app"
        binary.write_bytes(b"\x7f")

        assert _includes_from_compile_commands(str(binary), ["/my/k.cu"]) == [str(inc)]

    def test_falls_back_to_the_database_dir_when_no_directory_field(self, tmp_path):
        """Relative -I paths still resolve when the entry omits 'directory'."""
        inc = tmp_path / "include"
        inc.mkdir()
        (tmp_path / "compile_commands.json").write_text(
            json.dumps([{"file": "/elsewhere/k.cu", "command": "hipcc -Iinclude"}])
        )
        binary = tmp_path / "app"
        binary.write_bytes(b"\x7f")

        assert _includes_from_compile_commands(str(binary), ["/my/k.cu"]) == [str(inc)]

    def test_no_matching_entry_returns_empty(self, tmp_path):
        write_cc(tmp_path, [entry("/other/thing.cu", command="hipcc -I/x")])
        binary = tmp_path / "app"
        binary.write_bytes(b"\x7f")

        assert _includes_from_compile_commands(str(binary), ["/my/k.cu"]) == []

    def test_a_malformed_database_is_tolerated(self, tmp_path):
        (tmp_path / "compile_commands.json").write_text("[[[")
        binary = tmp_path / "app"
        binary.write_bytes(b"\x7f")

        assert _includes_from_compile_commands(str(binary), ["/my/k.cu"]) == []


# --------------------------------------------------------------------------
# _extract_output_path
# --------------------------------------------------------------------------


class TestExtractOutputPath:
    def test_separated_form(self):
        result = _extract_output_path(["hipcc", "-o", "obj/k.o", "k.cu"], "/build")
        assert result == os.path.normpath("/build/obj/k.o")

    def test_attached_form(self):
        result = _extract_output_path(["hipcc", "-oobj/k.o", "k.cu"], "/build")
        assert result == os.path.normpath("/build/obj/k.o")

    def test_absolute_paths_are_left_alone(self):
        assert _extract_output_path(["hipcc", "-o", "/abs/k.o"], "/build") == "/abs/k.o"

    def test_no_working_dir(self):
        assert _extract_output_path(["hipcc", "-o", "k.o"], "") == "k.o"

    def test_no_output_flag(self):
        assert _extract_output_path(["hipcc", "-c", "k.cu"], "/build") is None

    def test_trailing_bare_o_is_ignored(self):
        assert _extract_output_path(["hipcc", "k.cu", "-o"], "/build") is None

    def test_optimisation_flags_are_not_mistaken_for_output(self):
        """``-O3`` starts with ``-O``, not ``-o`` — case matters."""
        assert _extract_output_path(["hipcc", "-O3", "k.cu"], "/build") is None


# --------------------------------------------------------------------------
# _nm_has_symbol
# --------------------------------------------------------------------------


class TestNmHasSymbol:
    def test_three_column_form(self):
        out = "0000000000001234 T _Z6kernelPf\n"
        assert _nm_has_symbol(out, "_Z6kernelPf") is True

    def test_two_column_form(self):
        """Undefined symbols have no address column."""
        assert _nm_has_symbol("         U _Z6kernelPf\n", "_Z6kernelPf") is True

    def test_requires_an_exact_match(self):
        """A prefix match would pick the wrong translation unit."""
        out = "0000000000001234 T _Z6kernelPfSuffix\n"
        assert _nm_has_symbol(out, "_Z6kernelPf") is False

    def test_a_substring_elsewhere_on_the_line_does_not_count(self):
        assert _nm_has_symbol("0000 T other _Z6kernelPf\n", "_Z6kernelPf") is False

    def test_empty_output(self):
        assert _nm_has_symbol("", "_Z6kernelPf") is False

    def test_finds_the_symbol_among_many(self):
        out = "0000 T _Z1aPf\n0010 T _Z6kernelPf\n0020 T _Z1bPf\n"
        assert _nm_has_symbol(out, "_Z6kernelPf") is True


# --------------------------------------------------------------------------
# _match_tu_via_object_symbols
# --------------------------------------------------------------------------


class TestMatchTuViaObjectSymbols:
    """Picking the right translation unit for a templated kernel.

    Templated HIP kernels are instantiated across many .cu files (llama.cpp's
    ``mmq-instance-*.cu`` is the motivating case). Only one object file
    actually defines the captured mangled symbol, and choosing the wrong one
    produces a reproducer that rebuilds a different instantiation.
    """

    @staticmethod
    def setup_tu(tmp_path, name="k.cu", obj="k.o"):
        """Create a source file plus its object file; return both paths."""
        src = tmp_path / name
        src.write_text("")
        obj_path = tmp_path / obj
        obj_path.write_bytes(b"\x7fELF")
        return str(src), str(obj_path)

    def test_matches_the_object_containing_the_symbol(self, tmp_path, monkeypatch):
        src_a, obj_a = self.setup_tu(tmp_path, "a.cu", "a.o")
        src_b, obj_b = self.setup_tu(tmp_path, "b.cu", "b.o")
        cc = write_cc(
            tmp_path,
            [
                entry(src_a, command=f"hipcc -o {obj_a} -c a.cu", directory=str(tmp_path)),
                entry(src_b, command=f"hipcc -o {obj_b} -c b.cu", directory=str(tmp_path)),
            ],
        )

        def _nm(argv, **_kwargs):
            import subprocess as sp

            found = argv[1] == obj_b
            out = "0000 T _Z6kernelIiEvPT_\n" if found else "0000 T _Z5otherv\n"
            return sp.CompletedProcess(argv, 0, stdout=out, stderr="")

        monkeypatch.setattr("subprocess.run", _nm)
        candidates = [(src_a, "cmd_a", "dir_a"), (src_b, "cmd_b", "dir_b")]

        result = _match_tu_via_object_symbols("_Z6kernelIiEvPT_", candidates, cc)

        assert result == (src_b, "cmd_b", "dir_b")

    def test_returns_none_when_no_object_defines_the_symbol(self, tmp_path, monkeypatch):
        src, obj = self.setup_tu(tmp_path)
        cc = write_cc(
            tmp_path, [entry(src, command=f"hipcc -o {obj} -c k.cu", directory=str(tmp_path))]
        )

        monkeypatch.setattr(
            "subprocess.run",
            lambda argv, **kw: __import__("subprocess").CompletedProcess(
                argv, 0, stdout="0000 T _Z5otherv\n", stderr=""
            ),
        )

        assert _match_tu_via_object_symbols("_Z6missing", [(src, "c", "d")], cc) is None

    def test_a_malformed_database_returns_none(self, tmp_path):
        cc = tmp_path / "compile_commands.json"
        cc.write_text("{not json")

        assert _match_tu_via_object_symbols("_Z6kernel", [("/x.cu", "c", "d")], str(cc)) is None

    def test_a_missing_database_returns_none(self, tmp_path):
        missing = str(tmp_path / "compile_commands.json")

        assert _match_tu_via_object_symbols("_Z6kernel", [("/x.cu", "c", "d")], missing) is None

    def test_candidates_without_a_database_entry_are_skipped(self, tmp_path, monkeypatch):
        src, obj = self.setup_tu(tmp_path)
        cc = write_cc(tmp_path, [entry("/unrelated.cu", command="hipcc -o /u.o -c u.cu")])
        called = []
        monkeypatch.setattr("subprocess.run", lambda argv, **kw: called.append(argv))

        assert _match_tu_via_object_symbols("_Z6kernel", [(src, "c", "d")], cc) is None
        assert called == []

    def test_entries_without_an_output_flag_are_skipped(self, tmp_path, monkeypatch):
        """A link step or a syntax-only invocation has no object to inspect."""
        src, _obj = self.setup_tu(tmp_path)
        cc = write_cc(
            tmp_path, [entry(src, command="hipcc -fsyntax-only k.cu", directory=str(tmp_path))]
        )
        called = []
        monkeypatch.setattr("subprocess.run", lambda argv, **kw: called.append(argv))

        assert _match_tu_via_object_symbols("_Z6kernel", [(src, "c", "d")], cc) is None
        assert called == []

    def test_an_unbuilt_object_file_is_skipped(self, tmp_path, monkeypatch):
        """The database may reference objects a clean tree has not built."""
        src = tmp_path / "k.cu"
        src.write_text("")
        cc = write_cc(
            tmp_path,
            [
                entry(
                    str(src),
                    command=f"hipcc -o {tmp_path / 'never_built.o'} -c k.cu",
                    directory=str(tmp_path),
                )
            ],
        )
        called = []
        monkeypatch.setattr("subprocess.run", lambda argv, **kw: called.append(argv))

        assert _match_tu_via_object_symbols("_Z6kernel", [(str(src), "c", "d")], cc) is None
        assert called == []

    def test_a_failing_nm_skips_that_candidate(self, tmp_path, monkeypatch):
        src_a, obj_a = self.setup_tu(tmp_path, "a.cu", "a.o")
        src_b, obj_b = self.setup_tu(tmp_path, "b.cu", "b.o")
        cc = write_cc(
            tmp_path,
            [
                entry(src_a, command=f"hipcc -o {obj_a} -c a.cu", directory=str(tmp_path)),
                entry(src_b, command=f"hipcc -o {obj_b} -c b.cu", directory=str(tmp_path)),
            ],
        )

        def _nm(argv, **_kwargs):
            import subprocess as sp

            if argv[1] == obj_a:
                return sp.CompletedProcess(argv, 1, stdout="", stderr="not an object")
            return sp.CompletedProcess(argv, 0, stdout="0000 T _Z6kernel\n", stderr="")

        monkeypatch.setattr("subprocess.run", _nm)
        candidates = [(src_a, "ca", "da"), (src_b, "cb", "db")]

        assert _match_tu_via_object_symbols("_Z6kernel", candidates, cc) == (src_b, "cb", "db")

    @pytest.mark.parametrize(
        "exc",
        [
            FileNotFoundError("nm"),
            __import__("subprocess").TimeoutExpired("nm", 10),
            OSError("exec format error"),
        ],
    )
    def test_nm_exceptions_skip_that_candidate(self, tmp_path, monkeypatch, exc):
        """No nm on PATH must not crash source finding."""
        src, obj = self.setup_tu(tmp_path)
        cc = write_cc(
            tmp_path, [entry(src, command=f"hipcc -o {obj} -c k.cu", directory=str(tmp_path))]
        )

        def _raise(argv, **_kwargs):
            raise exc

        monkeypatch.setattr("subprocess.run", _raise)

        assert _match_tu_via_object_symbols("_Z6kernel", [(src, "c", "d")], cc) is None

    def test_relative_entry_files_resolve_against_directory(self, tmp_path, monkeypatch):
        sub = tmp_path / "sub"
        sub.mkdir()
        src = sub / "k.cu"
        src.write_text("")
        obj = sub / "k.o"
        obj.write_bytes(b"\x7fELF")
        cc = write_cc(
            tmp_path, [entry("k.cu", command=f"hipcc -o {obj} -c k.cu", directory=str(sub))]
        )

        monkeypatch.setattr(
            "subprocess.run",
            lambda argv, **kw: __import__("subprocess").CompletedProcess(
                argv, 0, stdout="0000 T _Z6kernel\n", stderr=""
            ),
        )

        assert _match_tu_via_object_symbols("_Z6kernel", [(str(src), "c", "d")], cc) == (
            str(src),
            "c",
            "d",
        )

    def test_the_arguments_field_is_used_when_present(self, tmp_path, monkeypatch):
        src, obj = self.setup_tu(tmp_path)
        cc = write_cc(
            tmp_path,
            [entry(src, arguments=["hipcc", "-o", obj, "-c", "k.cu"], directory=str(tmp_path))],
        )

        monkeypatch.setattr(
            "subprocess.run",
            lambda argv, **kw: __import__("subprocess").CompletedProcess(
                argv, 0, stdout="0000 T _Z6kernel\n", stderr=""
            ),
        )

        assert _match_tu_via_object_symbols("_Z6kernel", [(src, "c", "d")], cc) is not None
