#!/bin/bash
# Setup metrix on a compute node and run example.py
# Usage: bash setup_and_run.sh
set -e

WORKDIR="/tmp/metrix-test-$$"
REPO="https://github.com/AMDResearch/IntelliKit.git"
BRANCH="muhaawad/rdna"

echo "=== Setting up metrix test environment ==="
echo "Work directory: $WORKDIR"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Clone if not already there
if [ ! -d "intellikit" ]; then
    echo "Cloning IntelliKit ($BRANCH)..."
    git clone --branch "$BRANCH" "$REPO" intellikit
else
    echo "Repo already cloned, pulling latest..."
    cd intellikit && git pull && cd ..
fi

cd intellikit/metrix

echo "Installing metrix..."
pip install --user . 2>&1 | tail -1

echo "Running example.py..."
echo ""
python example.py
