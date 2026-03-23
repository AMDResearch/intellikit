# SPDX-License-Identifier: MIT

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from linex import Linex
from linex.distributed import detect_distributed_context, normalize_command_argv


def _write_code_json(ui_dir: Path, source_location: str) -> None:
    ui_dir.mkdir(parents=True, exist_ok=True)
    code = {
        "code": [
            [
                "s_add_u32 s0, s0, s1",
                0,
                0,
                source_location,
                1,
                0x1000,
                4,
                100,
                20,
                0,
            ]
        ]
    }
    (ui_dir / "code.json").write_text(json.dumps(code))


def test_distributed_helpers_parse_common_env():
    ctx = detect_distributed_context({"SLURM_PROCID": "3", "SLURM_LOCALID": "1", "SLURM_NTASKS": "8"})
    assert ctx.global_rank == 3
    assert ctx.local_rank == 1
    assert ctx.world_size == 8
    assert ctx.launcher == "srun"
    assert normalize_command_argv('torchrun --nproc_per_node=2 train.py --arg "two words"')[-1] == (
        "two words"
    )


def test_profile_uses_rank_scoped_output_and_loads_deterministic_ui_dir(tmp_path):
    dummy_decoder = tmp_path / "decoder" / "librocprof-trace-decoder.so"
    dummy_decoder.parent.mkdir(parents=True, exist_ok=True)
    dummy_decoder.write_text("placeholder")

    def fake_run(cmd, **kwargs):
        output_dir = Path(cmd[cmd.index("-d") + 1])
        _write_code_json(output_dir / "ui_output_200", "kernel2.hip:20")
        _write_code_json(output_dir / "ui_output_100", "kernel1.hip:10")
        m = MagicMock()
        m.returncode = 0
        m.stdout = ""
        m.stderr = ""
        return m

    with patch.object(Linex, "_ensure_decoder", return_value=dummy_decoder), \
         patch("subprocess.run", side_effect=fake_run):
        profiler = Linex()
        profiler.profile(
            command='python -c "print(1)"',
            output_dir=str(tmp_path / "linex_out"),
            env={"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"},
        )

    # Primary profile should come from lexicographically first ui_output directory.
    assert profiler.source_lines[0].source_location == "kernel1.hip:10"
    assert profiler.distributed_context.is_distributed
    assert profiler.distributed_context.global_rank == 1
    assert len(profiler.rank_profiles) == 2


def test_profile_with_launcher_builds_correct_command_order(tmp_path):
    """When launcher is provided, the subprocess command should be
    launcher_argv + rocprofv3 ... -- app_argv."""
    dummy_decoder = tmp_path / "decoder" / "librocprof-trace-decoder.so"
    dummy_decoder.parent.mkdir(parents=True, exist_ok=True)
    dummy_decoder.write_text("placeholder")

    captured_cmd = []

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        output_dir = Path(cmd[cmd.index("-d") + 1])
        ui_dir = output_dir / "ui_output_000"
        ui_dir.mkdir(parents=True, exist_ok=True)
        code = {"code": [["s_nop 0", 0, 0, "test.hip:1", 1, 0x1000, 4, 10, 2, 0]]}
        (ui_dir / "code.json").write_text(json.dumps(code))
        m = MagicMock()
        m.returncode = 0
        m.stdout = ""
        m.stderr = ""
        return m

    with patch.object(Linex, "_ensure_decoder", return_value=dummy_decoder), \
         patch("subprocess.run", side_effect=fake_run):
        profiler = Linex()
        profiler.profile(
            command="train.py --lr 0.01",
            launcher="torchrun --nproc_per_node=4",
            output_dir=str(tmp_path / "out"),
        )

    # Command should be: torchrun --nproc_per_node=4 rocprofv3 ... -- train.py --lr 0.01
    assert captured_cmd[0] == "torchrun"
    assert captured_cmd[1] == "--nproc_per_node=4"
    rocprofv3_idx = captured_cmd.index("rocprofv3")
    assert rocprofv3_idx == 2
    separator_idx = captured_cmd.index("--")
    assert captured_cmd[separator_idx + 1] == "train.py"
    assert captured_cmd[separator_idx + 2] == "--lr"
    assert captured_cmd[separator_idx + 3] == "0.01"


def test_profile_without_launcher_uses_plain_rocprofv3(tmp_path):
    """Without launcher, command should be: rocprofv3 ... -- app_argv."""
    dummy_decoder = tmp_path / "decoder" / "librocprof-trace-decoder.so"
    dummy_decoder.parent.mkdir(parents=True, exist_ok=True)
    dummy_decoder.write_text("placeholder")

    captured_cmd = []

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        output_dir = Path(cmd[cmd.index("-d") + 1])
        ui_dir = output_dir / "ui_output_000"
        ui_dir.mkdir(parents=True, exist_ok=True)
        code = {"code": [["s_nop 0", 0, 0, "test.hip:1", 1, 0x1000, 4, 10, 2, 0]]}
        (ui_dir / "code.json").write_text(json.dumps(code))
        m = MagicMock()
        m.returncode = 0
        m.stdout = ""
        m.stderr = ""
        return m

    with patch.object(Linex, "_ensure_decoder", return_value=dummy_decoder), \
         patch("subprocess.run", side_effect=fake_run):
        profiler = Linex()
        profiler.profile(
            command="./my_app --size 1024",
            output_dir=str(tmp_path / "out"),
        )

    assert captured_cmd[0] == "rocprofv3"
    separator_idx = captured_cmd.index("--")
    assert captured_cmd[separator_idx + 1] == "./my_app"
