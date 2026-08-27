# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the IPC layer's file and process handling.

Only the parts that are pure OS/file logic are covered here. ``get_kern_arg_data``
copies device memory through ctypes and is deliberately left alone -- stubbing
the GPU out of it would leave a test that only exercises the stub.
"""

import errno
import os
from unittest.mock import patch

import pytest

from accordo._internal.ipc import communication as comm
from accordo.exceptions import AccordoKernelNeverDispatched

RECORD_LEN = 72
HANDLE_LEN = 64


def _record(handle_byte=0xAB, size=4096):
    """One well-formed 72-byte IPC record: 64-byte handle + 8-byte little-endian size."""
    return bytes([handle_byte]) * HANDLE_LEN + size.to_bytes(8, "little")


def _framed(*records):
    return b"".join(b"BEGIN\n" + r + b"END\n" for r in records)


def _write(path, payload):
    path.write_bytes(payload)
    return str(path)


class TestProcessIsAlive:
    def test_none_pid_treated_as_alive(self):
        assert comm._process_is_alive(None) is True

    def test_reaped_child_is_dead(self):
        with patch("os.waitpid", return_value=(1234, 0)):
            assert comm._process_is_alive(1234) is False

    def test_child_process_error_is_dead(self):
        with patch("os.waitpid", side_effect=ChildProcessError()):
            assert comm._process_is_alive(1234) is False

    def test_echild_is_dead(self):
        with patch("os.waitpid", side_effect=OSError(errno.ECHILD, "no child")):
            assert comm._process_is_alive(1234) is False

    def test_other_oserror_falls_through_to_kill_probe(self):
        """A non-ECHILD waitpid error must not itself decide liveness."""
        with (
            patch("os.waitpid", side_effect=OSError(errno.EPERM, "denied")),
            patch("os.kill", return_value=None),
        ):
            assert comm._process_is_alive(1234) is True

    def test_still_running_when_kill_probe_succeeds(self):
        with patch("os.waitpid", return_value=(0, 0)), patch("os.kill", return_value=None):
            assert comm._process_is_alive(1234) is True

    def test_dead_when_kill_probe_fails(self):
        with (
            patch("os.waitpid", return_value=(0, 0)),
            patch("os.kill", side_effect=OSError(errno.ESRCH, "no such process")),
        ):
            assert comm._process_is_alive(1234) is False

    def test_kill_probe_uses_signal_zero(self):
        with patch("os.waitpid", return_value=(0, 0)), patch("os.kill") as kill:
            comm._process_is_alive(4321)
        kill.assert_called_once_with(4321, 0)


class TestReadIpcRecords:
    def test_absent_file_yields_nothing(self, tmp_path):
        assert comm._read_ipc_records(str(tmp_path / "missing")) == []

    def test_single_record_parsed(self, tmp_path):
        path = _write(tmp_path / "ipc", _framed(_record(0x01, 1024)))
        records = comm._read_ipc_records(path)
        assert len(records) == 1
        handle, size = records[0]
        assert size == 1024
        assert len(handle) == HANDLE_LEN
        assert set(handle.tolist()) == {0x01}

    def test_records_returned_in_order(self, tmp_path):
        path = _write(tmp_path / "ipc", _framed(_record(1, 10), _record(2, 20), _record(3, 30)))
        assert [s for _, s in comm._read_ipc_records(path)] == [10, 20, 30]

    def test_duplicates_are_preserved(self, tmp_path):
        """Unlike read_ipc_handles, this reader does not de-duplicate."""
        path = _write(tmp_path / "ipc", _framed(_record(7, 99), _record(7, 99)))
        assert len(comm._read_ipc_records(path)) == 2

    def test_unterminated_message_skipped(self, tmp_path):
        path = _write(tmp_path / "ipc", b"BEGIN\n" + _record() + _framed(_record(2, 8)))
        assert [s for _, s in comm._read_ipc_records(path)] == [8]

    def test_wrong_length_content_skipped(self, tmp_path):
        path = _write(tmp_path / "ipc", _framed(b"tooshort") + _framed(_record(9, 64)))
        assert [s for _, s in comm._read_ipc_records(path)] == [64]

    def test_empty_file_yields_nothing(self, tmp_path):
        assert comm._read_ipc_records(_write(tmp_path / "ipc", b"")) == []

    def test_size_decoded_little_endian(self, tmp_path):
        path = _write(tmp_path / "ipc", _framed(_record(0, 0x0102030405060708)))
        assert comm._read_ipc_records(path)[0][1] == 0x0102030405060708


class TestReadIpcHandles:
    def test_returns_immediately_when_no_pointers_expected(self, tmp_path):
        """count==0 means the loop never runs, so a missing file is fine."""
        handles, sizes = comm.read_ipc_handles(["int", "const float*"], str(tmp_path / "absent"))
        assert handles == [] and sizes == []

    def test_counts_only_non_const_pointers(self, tmp_path):
        path = _write(tmp_path / "ipc", _framed(_record(1, 16)))
        handles, sizes = comm.read_ipc_handles(["const float*", "float*", "int"], path)
        assert len(handles) == 1
        assert sizes == [16]

    def test_collects_multiple_distinct_handles(self, tmp_path):
        path = _write(tmp_path / "ipc", _framed(_record(1, 10), _record(2, 20)))
        handles, sizes = comm.read_ipc_handles(["float*", "double*"], path)
        assert len(handles) == 2
        assert sizes == [10, 20]

    def test_duplicate_handles_are_deduplicated(self, tmp_path):
        """The same handle appearing twice must not satisfy two expected pointers."""
        path = _write(tmp_path / "ipc", _framed(_record(5, 32), _record(5, 32), _record(6, 64)))
        handles, sizes = comm.read_ipc_handles(["float*", "int*"], path)
        assert len(handles) == 2
        assert sizes == [32, 64]

    def test_sentinel_raises_kernel_never_dispatched(self, tmp_path):
        sentinel = tmp_path / "sentinel"
        sentinel.touch()
        with pytest.raises(AccordoKernelNeverDispatched, match="never dispatched"):
            comm.read_ipc_handles(["float*"], str(tmp_path / "absent"), sentinel_file=str(sentinel))

    def test_waits_for_missing_file_then_proceeds(self, tmp_path):
        """Absent file must poll rather than fail; appearing later must be picked up."""
        path = tmp_path / "ipc"
        state = {"n": 0}
        real_exists = os.path.exists

        def fake_exists(p):
            if str(p) == str(path):
                state["n"] += 1
                if state["n"] == 1:
                    return False
                path.write_bytes(_framed(_record(1, 8)))
            return real_exists(p)

        with patch("accordo._internal.ipc.communication.os.path.exists", side_effect=fake_exists):
            with patch("accordo._internal.ipc.communication.time.sleep"):
                handles, sizes = comm.read_ipc_handles(["float*"], str(path))
        assert sizes == [8]
        assert state["n"] >= 2  # polled at least twice

    def test_polls_until_all_handles_present(self, tmp_path):
        """Two pointers expected but only one available at first."""
        path = tmp_path / "ipc"
        path.write_bytes(_framed(_record(1, 10)))
        calls = {"n": 0}

        def grow(_seconds):
            calls["n"] += 1
            path.write_bytes(_framed(_record(1, 10), _record(2, 20)))

        with patch("accordo._internal.ipc.communication.time.sleep", side_effect=grow):
            handles, sizes = comm.read_ipc_handles(["float*", "float*"], str(path))
        assert sizes == [10, 20]
        assert calls["n"] >= 1


class TestSendResponse:
    def test_writes_done_to_the_pipe(self, tmp_path):
        pipe = tmp_path / "fifo"
        comm.send_response(str(pipe))
        assert pipe.read_text() == "done\n"
