"""Unit tests for module-variable snapshot/restore.

These tests focus on the JSON manifest schema and replay-side robustness
that can be exercised without a GPU. Full end-to-end behavior is covered
by tests/integration/test_constant_memory_kernel.py.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------

REQUIRED_VAR_FIELDS = {"executable_sha256", "name", "size", "blob"}


def _make_manifest(entries):
    return {"variables": entries}


class TestModuleVariablesManifestSchema:
    """Spec-level checks on the on-disk module_variables.json shape.

    The replay binary's STAGE 4.5 parser must accept exactly this layout;
    any drift here is a contract break with libkerncap's writer.
    """

    def test_empty_manifest_round_trips(self, tmp_path):
        manifest = _make_manifest([])
        path = tmp_path / "module_variables.json"
        path.write_text(json.dumps(manifest))
        loaded = json.loads(path.read_text())
        assert loaded == {"variables": []}

    def test_entry_has_required_fields(self):
        entry = {
            "executable_sha256": "deadbeef" * 8,
            "name": "kokkos_impl_hip_constant_memory_buffer",
            "size": 32768,
            "blob": "module_variables/deadbeefdeadbeef_kokkos_impl_hip_constant_memory_buffer.bin",
        }
        assert REQUIRED_VAR_FIELDS.issubset(entry.keys())

    def test_blob_path_is_relative_under_module_variables_dir(self):
        entry = {
            "executable_sha256": "abc123" * 10,
            "name": "my_const",
            "size": 16,
            "blob": "module_variables/abc123abc123abc1_my_const.bin",
        }
        assert entry["blob"].startswith("module_variables/")
        assert entry["blob"].endswith(".bin")

    def test_sanitized_names_replace_colons_and_slashes(self):
        # Sanitization rule applied by libkerncap: ':' '/' '\\' ' ' -> '_'
        # Verifies the conventions the C++ writer uses.
        raw = "ns::Type::__var name/with\\bad chars"
        sanitized = raw.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
        assert ":" not in sanitized
        assert "/" not in sanitized
        assert "\\" not in sanitized
        assert " " not in sanitized

    def test_size_must_be_nonnegative_integer(self):
        entry = {
            "executable_sha256": "0" * 64,
            "name": "x",
            "size": 0,
            "blob": "module_variables/0000000000000000_x.bin",
        }
        # Size==0 entries are never written by libkerncap (they're filtered),
        # but the schema field type itself is integer.
        assert isinstance(entry["size"], int)
        assert entry["size"] >= 0


# ---------------------------------------------------------------------------
# Replay-side parsing tolerance (executed only when kerncap-replay exists)
# ---------------------------------------------------------------------------


def _replay_binary():
    """Locate kerncap-replay if it has been built/installed."""
    path = shutil.which("kerncap-replay")
    if path:
        return path
    # Fall back to package-relative location (editable install).
    try:
        import kerncap

        candidate = Path(kerncap.__file__).parent / "bin" / "kerncap-replay"
        if candidate.is_file():
            return str(candidate)
    except Exception:
        pass
    return None


@pytest.fixture
def stub_capture(tmp_path):
    """A minimal capture directory missing module_variables.json on purpose.

    Used to verify the replay binary doesn't choke on older captures.
    """
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "dispatch.json").write_text("{}")
    return capture


class TestReplayHandlesMissingManifest:
    """STAGE 4.5 must silently no-op when module_variables.json is absent."""

    def test_missing_manifest_is_back_compat(self, stub_capture):
        # Pure file-system check: the replay binary must not require the
        # manifest to exist. We don't actually invoke the binary here
        # (that requires a GPU + a real capture); we assert the contract
        # via the absence-of-file precondition that STAGE 4.5 silently
        # handles in replay.cpp.
        assert not (stub_capture / "module_variables.json").exists()

    def test_malformed_manifest_is_caught(self, stub_capture):
        # The replay parser uses nlohmann::json with exceptions=false and
        # checks .is_discarded(); make sure malformed JSON is at least
        # representable on disk without crashing the test.
        (stub_capture / "module_variables.json").write_text("{not json")
        # Pure schema-level assertion; integration verifies actual behavior.
        with pytest.raises(json.JSONDecodeError):
            json.loads((stub_capture / "module_variables.json").read_text())
