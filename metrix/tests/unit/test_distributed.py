"""
Unit tests for distributed launcher helpers.
"""

from metrix.utils.distributed import (
    DistributedContext,
    apply_rank_suffix,
    detect_distributed_context,
    normalize_command_argv,
)


def test_detect_distributed_context_torchrun_env():
    env = {"RANK": "2", "LOCAL_RANK": "0", "WORLD_SIZE": "4", "TORCHELASTIC_RUN_ID": "run-1"}
    ctx = detect_distributed_context(env)
    assert ctx.global_rank == 2
    assert ctx.local_rank == 0
    assert ctx.world_size == 4
    assert ctx.launcher == "torchrun"


def test_detect_distributed_context_mpi_env():
    env = {
        "OMPI_COMM_WORLD_RANK": "5",
        "OMPI_COMM_WORLD_LOCAL_RANK": "1",
        "OMPI_COMM_WORLD_SIZE": "8",
    }
    ctx = detect_distributed_context(env)
    assert ctx.global_rank == 5
    assert ctx.local_rank == 1
    assert ctx.world_size == 8
    assert ctx.launcher == "mpirun"


def test_apply_rank_suffix_distributed_file_path():
    ctx = DistributedContext(global_rank=3, world_size=8)
    assert apply_rank_suffix("results.json", ctx) == "results.rank0003.json"


def test_apply_rank_suffix_no_extension():
    ctx = DistributedContext(global_rank=0, world_size=4)
    assert apply_rank_suffix("results", ctx) == "results.rank0000"


def test_apply_rank_suffix_single_process():
    ctx = DistributedContext(global_rank=0, world_size=1)
    assert apply_rank_suffix("results.json", ctx) == "results.json"


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
