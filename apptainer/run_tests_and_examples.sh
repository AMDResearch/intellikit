#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# One Apptainer session: editable installs, pytest per package, key examples.
# From intellikit repo root:
#   bash apptainer/run_tests_and_examples.sh
#
# TMPDIR is under the repo bind mount so Nexus/KernelDB builds avoid full /tmp.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
chmod +x "$ROOT/.github/scripts/container_exec.sh" 2>/dev/null || true

bash "$ROOT/.github/scripts/container_exec.sh" "
set -u
cd /intellikit_workspace
mkdir -p .container_tmp
export TMPDIR=/intellikit_workspace/.container_tmp
export TMP=\$TMPDIR TEMP=\$TMPDIR

ec=0
fail() { echo \"[FAIL] \$*\"; ec=1; }

echo '========== pip install =========='
pip install -e accordo -q && pip install -e kerncap -q && pip install -e linex -q && pip install -e metrix -q && pip install -e nexus -q || fail pip

echo ''
echo '========== Accordo pytest =========='
( cd accordo && python3 -m pytest tests -v --tb=short ) || fail accordo-pytest

echo ''
echo '========== Accordo examples 03 + 04 =========='
( cd accordo && python3 examples/03_pytorch_tensor_add/validate.py ) || fail accordo-ex03
( cd accordo && python3 examples/04_pytorch_matmul/validate.py ) || fail accordo-ex04

echo ''
echo '========== Metrix pytest =========='
( cd metrix && python3 -m pytest tests -v --tb=short ) || fail metrix-pytest

echo ''
echo '========== Metrix tensor_add example =========='
( cd metrix && python3 examples/02_pytorch_tensor_add/tensor_add.py --size 256 ) || fail metrix-ex

echo ''
echo '========== Kerncap pytest =========='
( cd kerncap && python3 -m pytest tests -v --tb=short ) || fail kerncap-pytest

echo ''
echo '========== Kerncap tensor_add example =========='
( cd kerncap && python3 examples/02_pytorch_tensor_add/tensor_add.py --size 256 ) || fail kerncap-ex

echo ''
echo '========== Linex pytest =========='
( cd linex && python3 -m pytest tests -v --tb=short ) || fail linex-pytest

echo ''
echo '========== Nexus pytest =========='
( cd nexus && python3 -m pytest tests -v --tb=short ) || fail nexus-pytest

echo ''
echo '========== Nexus trace_pytorch example =========='
( cd nexus/examples/06_pytorch_tensor_add && python3 trace_pytorch.py ) || fail nexus-ex

exit \$ec
"
