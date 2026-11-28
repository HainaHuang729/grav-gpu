#!/bin/bash
#SBATCH --job-name=grav-mpi-test
#SBATCH --partition=b
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=grav-mpi-test-%j.out

set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/cuda/12.6}"
export PATH="${CUDA_HOME}/bin:/home/apps/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/math_libs/12.6/targets/x86_64-linux/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

GRAV_MPI_BIN="${GRAV_MPI_BIN:-./build/grav_mpi}"
GRAV_IC_BIN="${GRAV_IC_BIN:-./build/smoke_64.bin}"
GRAV_OUT_BIN="${GRAV_OUT_BIN:-/tmp/grav_mpi_out_${SLURM_JOB_ID}.bin}"

export GRAV_PM_GRID="${GRAV_PM_GRID:-128}"
export GRAV_LEAF_LIMIT="${GRAV_LEAF_LIMIT:-64}"
export GRAV_G="${GRAV_G:-43.0092}"
export GRAV_WARP_M2L="${GRAV_WARP_M2L:-1}"
export GRAV_FAST_SPLIT="${GRAV_FAST_SPLIT:-1}"
export GRAV_SPLIT_FORCE="${GRAV_SPLIT_FORCE:-1}"
export GRAV_DOMAIN_MODE="${GRAV_DOMAIN_MODE:-hilbert3d}"
export GRAV_LAZY_TREE="${GRAV_LAZY_TREE:-1}"

srun --mpi=pmix_v5 "${GRAV_MPI_BIN}" "${GRAV_IC_BIN}" "${GRAV_OUT_BIN}" "${GRAV_A_END:-1.0}"
