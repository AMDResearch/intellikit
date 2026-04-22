#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Build apptainer/intellikit.def on a single-MI300 Slurm node.
# Submit from the intellikit repository root (or set INTELLIKIT_ROOT).
#
#   sbatch apptainer/slurm_build_mi300_1x.sbatch
#
# Tune --partition / --account / --qos for your site; MI300 names differ by cluster.

#SBATCH --job-name=intellikit-sif
#SBATCH --time=03:00:00
#SBATCH --output=intellikit-apptainer-%x-%j.out
#SBATCH --error=intellikit-apptainer-%x-%j.err
#SBATCH -p mi3001x
#SBATCH --nodes=1

set -euo pipefail

umask 022

# --- Apptainer cache / temp (avoid filling $HOME; reuse across rebuilds when on scratch) ---
# Prefer Slurm job-local SSD when available; otherwise a repo-local cache directory.
if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" ]]; then
  export APPTAINER_TMPDIR="${SLURM_TMPDIR}/apptainer-tmp"
  export APPTAINER_CACHEDIR="${SLURM_TMPDIR}/apptainer-cache"
else
  _cache_root="${SLURM_SUBMIT_DIR:-${PWD}}/.apptainer-build-cache"
  export APPTAINER_TMPDIR="${_cache_root}/tmp"
  export APPTAINER_CACHEDIR="${_cache_root}/cache"
fi

# Apptainer reads these; some workflows still set Singularity names — set both.
export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"

mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

echo "[info] APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR}"
echo "[info] APPTAINER_TMPDIR=${APPTAINER_TMPDIR}"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "[error] apptainer not in PATH; load a module or install Apptainer." >&2
  exit 1
fi

# Resolve clone root: INTELLIKIT_ROOT, submit dir, or directory containing this batch file.
ROOT=""
if [[ -n "${INTELLIKIT_ROOT:-}" && -f "${INTELLIKIT_ROOT}/apptainer/intellikit.def" ]]; then
  ROOT="${INTELLIKIT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/apptainer/intellikit.def" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  _batch="${SLURM_JOB_SCRIPT:-${BASH_SOURCE[0]}}"
  _adir=$(cd "$(dirname "${_batch}")" && pwd)
  if [[ -f "${_adir}/intellikit.def" ]]; then
    ROOT=$(cd "${_adir}/.." && pwd)
  fi
fi

if [[ -z "${ROOT}" || ! -f "${ROOT}/apptainer/intellikit.def" ]]; then
  echo "[error] intellikit repo root not found (expected apptainer/intellikit.def)." >&2
  echo "        Set INTELLIKIT_ROOT, submit from repo root, or keep this .sbatch under apptainer/." >&2
  exit 1
fi

cd "${ROOT}"

DEF="apptainer/intellikit.def"
OUTDIR="apptainer/images"
IMAGE="${OUTDIR}/intellikit.sif"

mkdir -p "${OUTDIR}"

# Use fakeroot for unprivileged builds when the cluster supports it.
if apptainer build --help 2>&1 | grep -q fakeroot; then
  BUILD=(apptainer build --fakeroot --force "${IMAGE}" "${DEF}")
else
  BUILD=(apptainer build --force "${IMAGE}" "${DEF}")
fi

echo "[info] running: ${BUILD[*]}"
"${BUILD[@]}"

echo "[info] built: ${ROOT}/${IMAGE}"
ls -lh "${IMAGE}"
