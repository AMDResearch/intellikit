"""
Helpers for distributed launch environments.
"""

from __future__ import annotations

import os
import shlex
import socket
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DistributedContext:
    global_rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    node_rank: int = 0
    hostname: str = ""
    launcher: str = "single"

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def rank_tag(self) -> str:
        return f"rank{self.global_rank:04d}"


def _first_int(env: Mapping[str, str], keys: Sequence[str], default: int) -> int:
    for key in keys:
        value = env.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return default


def detect_distributed_context(env: Mapping[str, str] | None = None) -> DistributedContext:
    env_map = os.environ if env is None else env
    global_rank = _first_int(
        env_map,
        [
            "RANK",
            "OMPI_COMM_WORLD_RANK",
            "PMI_RANK",
            "PMIX_RANK",
            "SLURM_PROCID",
            "HOROVOD_RANK",
        ],
        0,
    )
    local_rank = _first_int(
        env_map,
        [
            "LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "SLURM_LOCALID",
            "HOROVOD_LOCAL_RANK",
        ],
        0,
    )
    world_size = _first_int(
        env_map,
        [
            "WORLD_SIZE",
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "PMIX_SIZE",
            "SLURM_NTASKS",
            "HOROVOD_SIZE",
        ],
        1,
    )
    node_rank = _first_int(
        env_map,
        ["GROUP_RANK", "NODE_RANK", "OMPI_COMM_WORLD_NODE_RANK", "SLURM_NODEID"],
        0,
    )
    hostname = env_map.get("HOSTNAME", "") or socket.gethostname()

    launcher = "single"
    if "TORCHELASTIC_RUN_ID" in env_map or "LOCAL_RANK" in env_map:
        launcher = "torchrun"
    elif "OMPI_COMM_WORLD_RANK" in env_map:
        launcher = "mpirun"
    elif "SLURM_PROCID" in env_map:
        launcher = "srun"
    elif "HOROVOD_RANK" in env_map:
        launcher = "horovodrun"

    return DistributedContext(
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        node_rank=node_rank,
        hostname=hostname,
        launcher=launcher,
    )


def normalize_command_argv(command: str | Sequence[str]) -> list[str]:
    if isinstance(command, (list, tuple)):
        argv = [str(arg) for arg in command]
        # Handle single-element lists containing spaces (e.g. from
        # argparse.REMAINDER passing a quoted command as one element).
        if len(argv) == 1 and " " in argv[0]:
            argv = shlex.split(argv[0])
    else:
        argv = shlex.split(command)
    if not argv:
        raise ValueError("Command is empty")
    return argv


def apply_rank_suffix(path: str, context: DistributedContext) -> str:
    """Append rank suffix to output paths for distributed runs."""
    if not context.is_distributed:
        return path

    from pathlib import Path as _Path

    p = _Path(path)
    suffix = p.suffix
    if suffix:
        return str(p.with_name(f"{p.stem}.{context.rank_tag}{suffix}"))
    return str(p.with_name(f"{p.name}.{context.rank_tag}"))
