#!/bin/bash
#SBATCH --job-name=grav-p2pall
#SBATCH --partition=a
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-mgpu1024-p2pall-%j.log
set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
KERNEL_BUILD_DIR="${KERNEL_BUILD_DIR:?Set KERNEL_BUILD_DIR to a GPU-configured upstream SWIFT build/source directory}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR containing PL_25_1024_0055.hdf5}"
REPORT_DIR="${REPORT_DIR:-$REPO_DIR/reports}"
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
MATHLIB_HOME="${MATHLIB_HOME:-$CUDA_HOME}"
MATHLIB_INC="${MATHLIB_INC:-$MATHLIB_HOME/include}"
MATHLIB_LIB="${MATHLIB_LIB:-$MATHLIB_HOME/lib}"
[ -f "$MATHLIB_INC/cufft.h" ] || MATHLIB_INC="$CUDA_HOME/include"
[ -d "$MATHLIB_LIB" ] || MATHLIB_LIB="$CUDA_HOME/lib64"
CXX_BIN="${CXX_BIN:-g++}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$MATHLIB_LIB:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1

mkdir -p "$REPO_DIR/logs" "$REPORT_DIR"
work=$(mktemp -d /tmp/grav-p2pall.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

echo "HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

mkdir -p "$work/build/src"
for f in runner_gpu.h runner_gpu.c gpu_gravity_offload.cu gpu_gravity_functions.cuh kernel_gravity.h kernel_long_gravity.h gravity_derivatives.h multipole.h; do
    cp "$KERNEL_BUILD_DIR/src/$f" "$work/build/src/"
done
cp "$KERNEL_BUILD_DIR/config.h" "$work/build/"
"$PYTHON_BIN" "$REPO_DIR/src/make-evolution.py" "$work/build" "$REPO_DIR"
GRAV_M2P_MATH_ONLY=1 GRAV_GROUP_M2L=1 "$PYTHON_BIN" "$REPO_DIR/m2p/make-port.py" "$work/build"

"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -arch="$CUDA_ARCH" --fmad=true -DGRAV_FAST_M2P_EXP \
    -Xcompiler -fopenmp -ccbin="$CXX_BIN" \
    -I"$work/build" -I"$work/build/src" -I"$KERNEL_BUILD_DIR/src" -I"$REPO_DIR/src" -I"$MATHLIB_INC" \
    -L"$MATHLIB_LIB" \
    "$REPO_DIR/src/multi_gpu_slab.cu" "$REPO_DIR/src/grav_cuda_pm.cu" "$REPO_DIR/src/grav_cuda_pm_dist.cu" "$REPO_DIR/src/grav_cuda_fof.cu" \
    -lcufft -o "$work/multi_gpu_slab"
echo "BUILD_P2PALL_OK"

cd "$work"
"$PYTHON_BIN" "$REPO_DIR/scripts/prepare_1024_input.py" "$DATA_DIR/PL_25_1024_0055.hdf5" "$work/input_0055.bin" 1 >"$REPORT_DIR/mgpu4-1024-0055-p2pall-input.json"

OMP_NUM_THREADS=4 GRAV_NUM_GPUS=4 GRAV_P2P_MODE=all GRAV_UNLINK_INPUT_AFTER_LOAD=1 \
    GRAV_SKIP_OUTPUT_WRITE=1 GRAV_DYNAMIC_PM=1 GRAV_LEAF_LIMIT=64 \
    GRAV_G=43.0092 GRAV_WARP_M2L=1 GRAV_FAST_SPLIT=1 \
    stdbuf -oL -eL "$work/multi_gpu_slab" "$work/input_0055.bin" "$work/out.bin" 0.9425 \
    >"$REPORT_DIR/mgpu4-1024-0055-p2pall-probe.log" 2>&1

grep -E "^MEMORY_PLAN |^GPU_INPUT |^SANITY |^RESULT " "$REPORT_DIR/mgpu4-1024-0055-p2pall-probe.log"
echo "P2P_ALL_PROBE_DONE"
