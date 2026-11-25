#!/bin/bash
#SBATCH --job-name=swift-treepm-smoke
#SBATCH --partition=a
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm-smoke-%j.log
set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
SWIFT_BUILD_DIR="${SWIFT_BUILD_DIR:?Set SWIFT_BUILD_DIR to a GPU-configured upstream SWIFT source/build directory}"
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
CXX_BIN="${CXX_BIN:-g++}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

echo "HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

make test-domain3d CXX="$CXX_BIN"
make test-fof CUDA_HOME="$CUDA_HOME" CXX="$CXX_BIN" PYTHON="$PYTHON_BIN"
make smoke SWIFT_BUILD_DIR="$SWIFT_BUILD_DIR" CUDA_HOME="$CUDA_HOME" CXX="$CXX_BIN" PYTHON="$PYTHON_BIN"
