# SPDX-License-Identifier: MIT

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from linex import Linex
from linex.distributed import detect_distributed_context, normalize_command_argv, split_launcher_command


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

    with (
        patch.object(Linex, "_ensure_decoder", return_value=dummy_decoder),
        patch("subprocess.run", side_effect=fake_run),
    ):
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


def test_split_launcher_command_torchrun():
    from linex.distributed import split_launcher_command
    split = split_launcher_command([
        "torchrun", "--nproc_per_node=8", "--nnodes", "2", "train.py", "--lr", "0.01"
    ])
    assert split.is_distributed
    assert split.launcher_name == "torchrun"
    assert split.launcher_argv == ["torchrun", "--nproc_per_node=8", "--nnodes", "2"]
    assert split.app_argv == ["train.py", "--lr", "0.01"]


def test_split_launcher_command_mpirun():
    from linex.distributed import split_launcher_command
    split = split_launcher_command([
        "mpirun", "-np", "4", "--bind-to", "core", "./my_app", "--size", "1024"
    ])
    assert split.is_distributed
    assert split.launcher_name == "mpirun"
    assert split.launcher_argv == ["mpirun", "-np", "4", "--bind-to", "core"]
    assert split.app_argv == ["./my_app", "--size", "1024"]


def test_split_launcher_command_srun():
    from linex.distributed import split_launcher_command
    split = split_launcher_command([
        "srun", "-N", "2", "-n", "16", "--gpus-per-node", "8", "./app"
    ])
    assert split.is_distributed
    assert split.launcher_name == "srun"
    assert split.launcher_argv == ["srun", "-N", "2", "-n", "16", "--gpus-per-node", "8"]
    assert split.app_argv == ["./app"]


def test_split_launcher_command_no_launcher():
    from linex.distributed import split_launcher_command
    split = split_launcher_command(["python3", "train.py", "--epochs", "10"])
    assert not split.is_distributed
    assert split.launcher_argv == []
    assert split.app_argv == ["python3", "train.py", "--epochs", "10"]


def test_split_launcher_command_python_m_torch_distributed():
    from linex.distributed import split_launcher_command
    split = split_launcher_command([
        "python3", "-m", "torch.distributed.run", "--nproc_per_node=4", "train.py"
    ])
    assert split.is_distributed
    assert split.launcher_name == "torchrun"
    assert split.launcher_argv == ["python3", "-m", "torch.distributed.run", "--nproc_per_node=4"]
    assert split.app_argv == ["train.py"]


def test_profile_builds_correct_command_order_with_launcher(tmp_path):
    """Verify that when a launcher is detected, the subprocess command is
    launcher_args + rocprofv3 ... -- app_args (not rocprofv3 -- launcher app)."""
    import json
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from linex import Linex

    dummy_decoder = tmp_path / "decoder" / "librocprof-trace-decoder.so"
    dummy_decoder.parent.mkdir(parents=True, exist_ok=True)
    dummy_decoder.write_text("placeholder")

    captured_cmd = []

    def fake_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        output_dir = Path(cmd[cmd.index("-d") + 1])
        # Write dummy trace data
        ui_dir = output_dir / "ui_output_000"
        ui_dir.mkdir(parents=True, exist_ok=True)
        code = {"code": [["s_nop 0", 0, 0, "test.hip:1", 1, 0x1000, 4, 10, 2, 0]]}
        (ui_dir / "code.json").write_text(json.dumps(code))
        m = MagicMock()
        m.returncode = 0
        m.stdout = ""
        m.stderr = ""
        return m

    with (
        patch.object(Linex, "_ensure_decoder", return_value=dummy_decoder),
        patch("subprocess.run", side_effect=fake_run),
    ):
        profiler = Linex()
        profiler.profile(
            command="torchrun --nproc_per_node=4 train.py --lr 0.01",
            output_dir=str(tmp_path / "out"),
        )

    # The command should start with torchrun, then rocprofv3
    assert captured_cmd[0] == "torchrun"
    assert captured_cmd[1] == "--nproc_per_node=4"
    rocprofv3_idx = captured_cmd.index("rocprofv3")
    assert rocprofv3_idx > 1  # rocprofv3 comes after launcher args
    # After --, the app args should be train.py, not torchrun
    separator_idx = captured_cmd.index("--")
    assert captured_cmd[separator_idx + 1] == "train.py"
    assert captured_cmd[separator_idx + 2] == "--lr"
    assert captured_cmd[separator_idx + 3] == "0.01"
