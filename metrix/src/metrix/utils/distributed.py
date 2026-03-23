"""
Distributed launcher helpers for Linex.
"""

from __future__ import annotations

import os
import shlex
import socket
from dataclasses import dataclass, field
from typing import Mapping, Sequence


KNOWN_LAUNCHERS = {
    "torchrun": "torchrun",
    "python": None,  # only when followed by -m torch.distributed
    "mpirun": "mpirun",
    "mpiexec": "mpirun",
    "srun": "srun",
    "horovodrun": "horovodrun",
}


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


@dataclass
class LauncherSplit:
    """Result of splitting a command into launcher prefix and application suffix."""
    launcher_argv: list[str] = field(default_factory=list)
    app_argv: list[str] = field(default_factory=list)
    launcher_name: str = "single"

    @property
    def is_distributed(self) -> bool:
        return len(self.launcher_argv) > 0


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
    else:
        argv = shlex.split(command)
    if not argv:
        raise ValueError("Command is empty")
    return argv


def split_launcher_command(argv: list[str]) -> LauncherSplit:
    """Split a command argv into launcher prefix and application suffix.

    Recognizes torchrun, mpirun/mpiexec, srun, horovodrun.
    For torchrun/python -m torch.distributed.*, all flags (--nproc_per_node etc.)
    before the script name are launcher args; the script and everything after are app args.
    For mpirun/mpiexec/srun/horovodrun, we split at the first positional arg that
    looks like an executable (not a flag).

    Returns a LauncherSplit with launcher_argv (empty if no launcher detected)
    and app_argv.
    """
    if not argv:
        return LauncherSplit(app_argv=argv)

    binary = os.path.basename(argv[0])

    # --- torchrun ---
    if binary == "torchrun":
        return _split_torchrun(argv)

    # --- python -m torch.distributed.launch / python -m torch.distributed.run ---
    if binary in ("python", "python3") and len(argv) >= 3:
        if argv[1] == "-m" and argv[2].startswith("torch.distributed"):
            return _split_torchrun(argv)

    # --- mpirun / mpiexec ---
    if binary in ("mpirun", "mpiexec"):
        return _split_mpi(argv)

    # --- srun ---
    if binary == "srun":
        return _split_srun(argv)

    # --- horovodrun ---
    if binary == "horovodrun":
        return _split_horovodrun(argv)

    # No launcher detected
    return LauncherSplit(app_argv=argv)


def _split_torchrun(argv: list[str]) -> LauncherSplit:
    """Split torchrun command. Flags before the script are launcher args."""
    # torchrun [flags] script.py [script args]
    # Flags all start with -- and some take a value argument.
    # We find the first arg that doesn't start with - and isn't a value of a flag.
    TORCHRUN_VALUE_FLAGS = {
        "--nproc_per_node", "--nproc-per-node", "--nnodes",
        "--node_rank", "--node-rank", "--master_addr", "--master-addr",
        "--master_port", "--master-port", "--rdzv_id", "--rdzv-id",
        "--rdzv_backend", "--rdzv-backend", "--rdzv_endpoint", "--rdzv-endpoint",
        "--rdzv_conf", "--rdzv-conf", "--max_restarts", "--max-restarts",
        "--monitor_interval", "--monitor-interval", "--log_dir", "--log-dir",
        "--redirects", "--tee", "-r", "-t", "--role", "--local_addr",
        "--local-addr", "--logs_specs", "--logs-specs",
        "--start_method", "--start-method", "--run_path", "--run-path",
        "--omp_num_threads", "--omp-num-threads",
    }
    i = 1  # skip argv[0] (torchrun / python)
    # skip python -m torch.distributed.run if present
    if os.path.basename(argv[0]) in ("python", "python3") and len(argv) > 2 and argv[1] == "-m":
        i = 3  # skip python -m torch.distributed.run

    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            # Explicit separator
            return LauncherSplit(
                launcher_argv=argv[:i],
                app_argv=argv[i + 1:],
                launcher_name="torchrun",
            )
        if arg.startswith("-"):
            # Check if this flag takes a value
            flag_name = arg.split("=")[0]
            if "=" not in arg and flag_name in TORCHRUN_VALUE_FLAGS:
                i += 2  # skip flag and its value
            else:
                i += 1
        else:
            # First positional = script name, everything from here is app
            return LauncherSplit(
                launcher_argv=argv[:i],
                app_argv=argv[i:],
                launcher_name="torchrun",
            )

    # All flags, no script found — treat entire thing as app
    return LauncherSplit(app_argv=argv)


def _split_mpi(argv: list[str]) -> LauncherSplit:
    """Split mpirun/mpiexec command."""
    MPI_VALUE_FLAGS = {
        "-np", "-n", "--np", "-N", "--map-by", "--bind-to", "--rank-by",
        "-H", "--host", "--hostfile", "-x", "--mca", "-wdir", "--wdir",
        "-oversubscribe", "--oversubscribe", "--prefix", "-output-filename",
        "--output-filename", "--report-bindings",
    }
    # Flags that take TWO values
    MPI_DOUBLE_VALUE_FLAGS = {"--mca"}

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == ":":
            # MPMD separator — everything before is one command spec
            # For simplicity, treat everything up to : as launcher
            return LauncherSplit(
                launcher_argv=argv[:i],
                app_argv=argv[i:],
                launcher_name="mpirun",
            )
        if arg.startswith("-"):
            flag_name = arg.split("=")[0]
            if "=" not in arg and flag_name in MPI_DOUBLE_VALUE_FLAGS:
                i += 3  # --mca key value
            elif "=" not in arg and flag_name in MPI_VALUE_FLAGS:
                i += 2
            else:
                i += 1
        else:
            # First positional = executable
            return LauncherSplit(
                launcher_argv=argv[:i],
                app_argv=argv[i:],
                launcher_name="mpirun",
            )

    return LauncherSplit(app_argv=argv)


def _split_srun(argv: list[str]) -> LauncherSplit:
    """Split srun command."""
    SRUN_VALUE_FLAGS = {
        "-N", "--nodes", "-n", "--ntasks", "-c", "--cpus-per-task",
        "-G", "--gpus", "--gpus-per-node", "--gpus-per-task",
        "-p", "--partition", "-w", "--nodelist", "-x", "--exclude",
        "-t", "--time", "-J", "--job-name", "-o", "--output", "-e", "--error",
        "--mem", "--mem-per-cpu", "--mem-per-gpu", "-D", "--chdir",
        "--export", "--mpi", "--distribution",
    }
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            return LauncherSplit(
                launcher_argv=argv[:i],
                app_argv=argv[i + 1:],
                launcher_name="srun",
            )
        if arg.startswith("-"):
            flag_name = arg.split("=")[0]
            if "=" not in arg and flag_name in SRUN_VALUE_FLAGS:
                i += 2
            else:
                i += 1
        else:
            return LauncherSplit(
                launcher_argv=argv[:i],
                app_argv=argv[i:],
                launcher_name="srun",
            )

    return LauncherSplit(app_argv=argv)


def _split_horovodrun(argv: list[str]) -> LauncherSplit:
    """Split horovodrun command."""
    HOROVOD_VALUE_FLAGS = {
        "-np", "-p", "--num-proc", "-H", "--hosts", "--hostfile",
        "--start-timeout", "--network-interface", "--output-filename",
        "--gloo-timeout-seconds",
    }
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("-"):
            flag_name = arg.split("=")[0]
            if "=" not in arg and flag_name in HOROVOD_VALUE_FLAGS:
                i += 2
            else:
                i += 1
        else:
            return LauncherSplit(
                launcher_argv=argv[:i],
                app_argv=argv[i:],
                launcher_name="horovodrun",
            )

    return LauncherSplit(app_argv=argv)


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
