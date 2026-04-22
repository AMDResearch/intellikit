#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Run kerncap PyTorch examples + profile drivers + integration pytest.
# Intended cwd: IntelliKit repository root (same as CI: /intellikit_workspace in
# container_exec.sh, or repo root on the host when skipping Apptainer).
#
# kerncap installs follow README.md § Install literally (commands run from kerncap/).
# This job runs both documented flows back-to-back; `pip uninstall` between them
# is only so two installs can coexist in one process — it is not part of the README.
#
# Invoked by test_pytorch_examples_mi300_1x.sbatch (and can be run manually
# after `cd $INTELLIKIT_ROOT`).

set -euo pipefail

ROOT="$(pwd)"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "${ROOT}/.container_tmp"
export TMPDIR="${ROOT}/.container_tmp"
export TMP="${TMPDIR}" TEMP="${TMPDIR}"

for _bin in hipcc rocprofv3; do
  if ! command -v "${_bin}" >/dev/null 2>&1; then
    echo "[error] ${_bin} not in PATH (ROCm not visible in this environment?)." >&2
    exit 1
  fi
done
echo "[info] hipcc=$(command -v hipcc) rocprofv3=$(command -v rocprofv3)"

python3 <<'PYCHECK'
import sys

try:
    import torch
except Exception as e:
    print(f"[error] import torch failed: {type(e).__name__}: {e}", file=sys.stderr)
    raise SystemExit(1) from e
if not torch.cuda.is_available():
    print(
        "[error] torch.cuda.is_available() is False; need GPU + ROCm PyTorch.",
        file=sys.stderr,
    )
    raise SystemExit(1)
print("[info] torch", torch.__version__, "cuda_available=", torch.cuda.is_available())
PYCHECK

_run_kerncap_examples_and_pytest() {
  cd "${ROOT}/kerncap"

  echo "[info] --- tensor_add.py ---"
  python3 examples/02_pytorch_tensor_add/tensor_add.py --size 512

  echo "[info] --- tensor_matmul.py ---"
  python3 examples/03_pytorch_matmul/tensor_matmul.py --size 256

  echo "[info] --- profile_with_kerncap.py (tensor add) ---"
  python3 examples/02_pytorch_tensor_add/profile_with_kerncap.py

  echo "[info] --- profile_with_kerncap.py (matmul) ---"
  python3 examples/03_pytorch_matmul/profile_with_kerncap.py

  echo "[info] --- pytest tests/integration/test_pytorch_tensor_add_example.py ---"
  python3 -m pytest tests/integration/test_pytorch_tensor_add_example.py -v --tb=short

  if [[ -n "${KERNCAP_SLURM_PYTORCH_KERNEL:-}" ]]; then
    echo "[info] KERNCAP_SLURM_PYTORCH_KERNEL=${KERNCAP_SLURM_PYTORCH_KERNEL} set; running extract pipeline on tensor add example."
    python3 examples/02_pytorch_tensor_add/profile_with_kerncap.py \
      --kernel "${KERNCAP_SLURM_PYTORCH_KERNEL}" \
      --iterations 3
  fi
}

_readme_install_then_test() {
  cd "${ROOT}/kerncap"
  pip uninstall -y kerncap >/dev/null 2>&1 || true
  "$@"
  _run_kerncap_examples_and_pytest
}

# README.md § Install — comment "From local source"
echo "[info] README.md Install: pip install ."
_readme_install_then_test pip install .

# README.md § Install — comment "Editable install for development"
# README shows: pip install -e .[dev]  (quote '.[dev]' in bash so [ is not glob)
echo "[info] README.md Install: pip install -e .[dev]"
_readme_install_then_test pip install -e '.[dev]'

echo "[info] all kerncap PyTorch example checks finished OK (both README.md Install flows)"
