#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Run kerncap PyTorch example workloads + profile drivers + integration pytest
# on a single GPU Slurm node (ROCm + PyTorch ROCm wheel).
#
# Submit from the IntelliKit repository root (recommended):
#   sbatch kerncap/slurm/test_pytorch_examples_mi300_1x.sbatch
#
# To run the same steps inside an existing allocation (salloc / interactive):
#   kerncap/slurm/run_pytorch_examples_in_allocation.sh
#
# Or set INTELLIKIT_ROOT to the clone and submit from elsewhere:
#   INTELLIKIT_ROOT=/path/to/intellikit sbatch /path/to/intellikit/kerncap/slurm/test_pytorch_examples_mi300_1x.sbatch
#
# Tune --partition / --account / --qos / GPU flags for your site. Examples:
#   #SBATCH -p mi3001x
#   #SBATCH --gpus-per-node=1
#   #SBATCH --gres=gpu:1
#
# Container path matches CI and apptainer/run_tests_and_examples.sh:
#   bash .github/scripts/container_exec.sh "…"
# Writable installs use an ephemeral overlay (not a repo-local venv).
# Optional GPU bind-through: sbatch … --export=ALL,INTELLIKIT_APPTAINER_ROCM=1
# (or export before sbatch) to pass --rocm to apptainer; same default as CI otherwise.
#
# Image: INTELLIKIT_SIF, or INTELLIKIT_SIF_HOME=1 ($HOME/.../intellikit-dev.sif),
#       else <repo>/apptainer/images/intellikit.sif (see container_exec.sh).
# Host-only (no SIF): INTELLIKIT_SKIP_APPTAINER=1

#SBATCH --job-name=kerncap-pytorch-ex
#SBATCH --time=00:45:00
#SBATCH --output=kerncap-pytorch-examples-%x-%j.out
#SBATCH --error=kerncap-pytorch-examples-%x-%j.err
#SBATCH --nodes=1
#SBATCH -p mi3001x

set -euo pipefail

umask 022

echo "[info] host=$(hostname) date=$(date -Is) job_id=${SLURM_JOB_ID:-}"

# --- Resolve IntelliKit root ---
ROOT=""
if [[ -n "${INTELLIKIT_ROOT:-}" && -d "${INTELLIKIT_ROOT}/kerncap/examples/02_pytorch_tensor_add" ]]; then
  ROOT="${INTELLIKIT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/kerncap/examples/02_pytorch_tensor_add" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  _batch="${SLURM_JOB_SCRIPT:-${BASH_SOURCE[0]}}"
  _here=$(cd "$(dirname "${_batch}")" && pwd)
  if [[ -d "${_here}/../examples/02_pytorch_tensor_add" ]]; then
    ROOT=$(cd "${_here}/../.." && pwd)
  fi
fi

if [[ -z "${ROOT}" || ! -d "${ROOT}/kerncap/examples/02_pytorch_tensor_add" ]]; then
  echo "[error] IntelliKit root not found (expected kerncap/examples/02_pytorch_tensor_add)." >&2
  echo "        Submit from repo root or set INTELLIKIT_ROOT." >&2
  exit 1
fi

_PAYLOAD="${ROOT}/kerncap/slurm/run_pytorch_examples_payload.sh"
if [[ ! -f "${_PAYLOAD}" ]]; then
  echo "[error] missing payload script: ${_PAYLOAD}" >&2
  exit 1
fi

cd "${ROOT}"
chmod +x "${ROOT}/.github/scripts/container_exec.sh" 2>/dev/null || true

if [[ "${INTELLIKIT_SKIP_APPTAINER:-0}" == "1" ]]; then
  echo "[info] INTELLIKIT_SKIP_APPTAINER=1 — running on host Python/ROCm (no SIF)"
  bash "${_PAYLOAD}"
  exit 0
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "[error] apptainer not in PATH." >&2
  echo "        Set INTELLIKIT_SKIP_APPTAINER=1 to use host ROCm/Python instead." >&2
  exit 1
fi

# Same image resolution as container_exec.sh (optional explicit SIF)
if [[ -n "${INTELLIKIT_SIF:-}" ]]; then
  _image="${INTELLIKIT_SIF}"
elif [[ "${INTELLIKIT_SIF_HOME:-}" == "1" ]]; then
  _image="${HOME}/apptainer/intellikit-dev.sif"
else
  _image="${ROOT}/apptainer/images/intellikit.sif"
fi
if [[ ! -f "${_image}" ]]; then
  echo "[error] Apptainer image not found: ${_image}" >&2
  echo "        Build it, set INTELLIKIT_SIF, or INTELLIKIT_SIF_HOME=1 for CI-style path." >&2
  exit 1
fi

echo "[info] INTELLIKIT_ROOT=${ROOT}"
if [[ "${INTELLIKIT_APPTAINER_ROCM:-0}" == "1" ]]; then
  echo "[info] container image=${_image} (INTELLIKIT_APPTAINER_ROCM=1 → apptainer --rocm)"
else
  echo "[info] container image=${_image} (same apptainer flags as CI; set INTELLIKIT_APPTAINER_ROCM=1 if your site needs --rocm)"
fi
# Ensure container_exec picks the same image we validated (including INTELLIKIT_SIF from user env)
export INTELLIKIT_SIF="${_image}"

exec bash "${ROOT}/.github/scripts/container_exec.sh" "bash kerncap/slurm/run_pytorch_examples_payload.sh"
