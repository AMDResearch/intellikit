#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Run the same steps as test_pytorch_examples_mi300_1x.sbatch on a GPU-capable
# machine: either an active Slurm step (SLURM_JOB_ID set) or any shell where
# ROCm + the Apptainer image can see the GPU (e.g. ssh to a compute node).
#
# With Slurm (recommended on shared clusters):
#   salloc -N 1 -n 1 -p mi3001x --time=0:45:00 --gpus-per-node=1   # adjust to your site
#   /path/to/intellikit/kerncap/slurm/run_pytorch_examples_in_allocation.sh
#
# One shot:
#   salloc ... bash /path/to/intellikit/kerncap/slurm/run_pytorch_examples_in_allocation.sh
#
# Environment:
#   INTELLIKIT_ROOT           Optional; defaults to git top-level from this script's location.
#   INTELLIKIT_SIF            Optional Apptainer image (default: ${INTELLIKIT_ROOT}/apptainer/images/intellikit.sif)
#   INTELLIKIT_SKIP_APPTAINER Set to 1 to run kerncap checks on the host instead of in the .sif
#   INTELLIKIT_APPTAINER_ROCM Set to 1 if your cluster needs `apptainer exec --rocm` (optional; CI does not set it)

set -euo pipefail

_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_repo_default=$(cd "${_here}/../.." && pwd)

if [[ -z "${INTELLIKIT_ROOT:-}" ]]; then
  if command -v git >/dev/null 2>&1 && git -C "${_repo_default}" rev-parse --show-toplevel >/dev/null 2>&1; then
    INTELLIKIT_ROOT=$(git -C "${_repo_default}" rev-parse --show-toplevel)
  else
    INTELLIKIT_ROOT="${_repo_default}"
  fi
fi
export INTELLIKIT_ROOT

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[warn] SLURM_JOB_ID unset (not inside salloc/srun/sbatch). Continuing anyway." >&2
  echo "       Use Slurm on shared systems: salloc ... then re-run this script." >&2
fi

if [[ ! -d "${INTELLIKIT_ROOT}/kerncap/examples/02_pytorch_tensor_add" ]]; then
  echo "[error] INTELLIKIT_ROOT=${INTELLIKIT_ROOT} does not look like the IntelliKit repo." >&2
  exit 1
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "[info] Slurm: JOB_ID=${SLURM_JOB_ID} STEP=${SLURM_STEP_ID:-} NODE=${SLURMD_NODENAME:-$(hostname)}"
else
  echo "[info] host=$(hostname) (no Slurm job id)"
fi
echo "[info] INTELLIKIT_ROOT=${INTELLIKIT_ROOT}"
if [[ "${INTELLIKIT_SKIP_APPTAINER:-0}" != "1" ]]; then
  echo "[info] workload will run in Apptainer (default: ${INTELLIKIT_ROOT}/apptainer/images/intellikit.sif)"
fi

# The .sbatch file is plain bash; #SBATCH directives are comments when not submitted via sbatch.
# It re-execs inside ${INTELLIKIT_SIF:-$INTELLIKIT_ROOT/apptainer/images/intellikit.sif} unless INTELLIKIT_SKIP_APPTAINER=1.
exec bash "${_here}/test_pytorch_examples_mi300_1x.sbatch"
