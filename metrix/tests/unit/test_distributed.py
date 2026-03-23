"""
Unit tests for distributed launcher helpers.
"""

from metrix.utils.distributed import (
    DistributedContext,
    apply_rank_suffix,
    detect_distributed_context,
    normalize_command_argv,
    split_launcher_command,
)


def test_detect_distributed_context_torchrun_env():
    env = {"RANK": "2", "LOCAL_RANK": "0", "WORLD_SIZE": "4", "TORCHELASTIC_RUN_ID": "run-1"}
    ctx = detect_distributed_context(env)
    assert ctx.global_rank == 2
    assert ctx.local_rank == 0
    assert ctx.world_size == 4
    assert ctx.launcher == "torchrun"


def test_detect_distributed_context_mpi_env():
    env = {"OMPI_COMM_WORLD_RANK": "5", "OMPI_COMM_WORLD_LOCAL_RANK": "1", "OMPI_COMM_WORLD_SIZE": "8"}
    ctx = detect_distributed_context(env)
    assert ctx.global_rank == 5
    assert ctx.local_rank == 1
    assert ctx.world_size == 8
    assert ctx.launcher == "mpirun"


def test_apply_rank_suffix_distributed_file_path():
    ctx = DistributedContext(global_rank=3, world_size=8)
    assert apply_rank_suffix("results.json", ctx) == "results.rank0003.json"


def test_normalize_command_argv_accepts_string_and_sequence():
    assert normalize_command_argv('torchrun --nproc_per_node=2 train.py --arg "two words"') == [
        "torchrun",
        "--nproc_per_node=2",
        "train.py",
        "--arg",
        "two words",
    ]
    assert normalize_command_argv(["mpirun", "-np", "4", "./app"]) == [
        "mpirun",
        "-np",
        "4",
        "./app",
    ]


def test_split_launcher_command_torchrun():
    split = split_launcher_command([
        "torchrun", "--nproc_per_node=8", "train.py", "--lr", "0.01"
    ])
    assert split.is_distributed
    assert split.launcher_name == "torchrun"
    assert split.launcher_argv == ["torchrun", "--nproc_per_node=8"]
    assert split.app_argv == ["train.py", "--lr", "0.01"]


def test_split_launcher_command_mpirun():
    split = split_launcher_command([
        "mpirun", "-np", "4", "./my_app", "--size", "1024"
    ])
    assert split.is_distributed
    assert split.launcher_name == "mpirun"
    assert split.launcher_argv == ["mpirun", "-np", "4"]
    assert split.app_argv == ["./my_app", "--size", "1024"]


def test_split_launcher_command_no_launcher():
    split = split_launcher_command(["./my_app", "--size", "1024"])
    assert not split.is_distributed
    assert split.app_argv == ["./my_app", "--size", "1024"]
