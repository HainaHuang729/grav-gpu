#!/bin/bash
#SBATCH --job-name=gravityCU-smoke
#SBATCH --partition=a
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm-smoke-%j.log
set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
KERNEL_BUILD_DIR="${KERNEL_BUILD_DIR:?Set KERNEL_BUILD_DIR to a GPU-configured upstream SWIFT source/build directory}"
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
CXX_BIN="${CXX_BIN:-g++}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

echo "HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

make test-domain3d CXX="$CXX_BIN"
make test-fof CUDA_HOME="$CUDA_HOME" CXX="$CXX_BIN" PYTHON="$PYTHON_BIN"
make smoke KERNEL_BUILD_DIR="$KERNEL_BUILD_DIR" CUDA_HOME="$CUDA_HOME" CXX="$CXX_BIN" PYTHON="$PYTHON_BIN"
