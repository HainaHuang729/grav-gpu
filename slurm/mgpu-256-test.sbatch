#!/bin/bash
#SBATCH --job-name=swift-mgpu256
#SBATCH --partition=a
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=logs/slurm-mgpu256-%j.log
set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
SWIFT_BUILD_DIR="${SWIFT_BUILD_DIR:?Set SWIFT_BUILD_DIR to a GPU-configured upstream SWIFT build/source directory}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"
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
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "$REPO_DIR/logs" "$REPORT_DIR"
work=$(mktemp -d /tmp/swift-mgpu256.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

echo "HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv >"$REPORT_DIR/mgpu-256-device.csv"
nvidia-smi topo -m >>"$REPORT_DIR/mgpu-256-device.csv" || true

mkdir -p "$work/build/src"
for f in runner_gpu.h runner_gpu.c gpu_gravity_offload.cu gpu_gravity_functions.cuh kernel_gravity.h kernel_long_gravity.h gravity_derivatives.h multipole.h; do
    cp "$SWIFT_BUILD_DIR/src/$f" "$work/build/src/"
done
cp "$SWIFT_BUILD_DIR/config.h" "$work/build/"
"$PYTHON_BIN" "$REPO_DIR/src/make-evolution.py" "$work/build" "$REPO_DIR"
SWIFT_M2P_MATH_ONLY=1 SWIFT_GROUP_M2L=1 "$PYTHON_BIN" "$REPO_DIR/m2p/make-port.py" "$work/build"

"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -arch="$CUDA_ARCH" --fmad=true -DSWIFT_FAST_M2P_EXP \
    -Xcompiler -fopenmp -Xptxas=-v -ccbin="$CXX_BIN" \
    -I"$work/build" -I"$work/build/src" -I"$SWIFT_BUILD_DIR/src" -I"$REPO_DIR/src" -I"$MATHLIB_INC" \
    -L"$MATHLIB_LIB" \
    "$REPO_DIR/src/multi_gpu_slab.cu" "$REPO_DIR/src/swift_cuda_pm.cu" "$REPO_DIR/src/swift_cuda_pm_dist.cu" "$REPO_DIR/src/swift_cuda_fof.cu" \
    -lcufft -o "$work/multi_gpu_slab" >"$REPORT_DIR/mgpu-256-build.log" 2>&1
echo "BUILD_MGPU_OK"

"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -arch="$CUDA_ARCH" --fmad=true -DSWIFT_FAST_M2P_EXP \
    -Xcompiler -fopenmp -ccbin="$CXX_BIN" \
    -I"$work/build" -I"$work/build/src" -I"$SWIFT_BUILD_DIR/src" -I"$REPO_DIR/src" -I"$MATHLIB_INC" \
    -L"$MATHLIB_LIB" "$REPO_DIR/src/slab_resident.cu" "$REPO_DIR/src/swift_cuda_pm.cu" -lcufft \
    -o "$work/slab_resident" >>"$REPORT_DIR/mgpu-256-build.log" 2>&1
echo "BUILD_SLAB_BACKCOMPAT_OK"

cd "$work"
"$PYTHON_BIN" "$REPO_DIR/scripts/prepare_swift_input.py" "$DATA_DIR/PL_25_256_0054.hdf5" input.bin >"$REPORT_DIR/mgpu-256-input.json"

for g in 1 2 4; do
    echo "=== Running SWIFT_NUM_GPUS=$g ==="
    export SWIFT_NUM_GPUS="$g"
    export SWIFT_DOMAIN_MODE=auto
    export SWIFT_PM_DIST=1
    export SWIFT_SPLIT_FORCE=1
    export SWIFT_LAZY_TREE=1
    export SWIFT_LEAF_LIMIT=64
    export SWIFT_DYNAMIC_PM=1
    export SWIFT_GRAV_G=43.0092
    export SWIFT_WARP_M2L=1
    export SWIFT_FAST_SPLIT=1
    export SWIFT_P2P_MODE=nvlink
    stdbuf -oL -eL ./multi_gpu_slab input.bin "$work/mgpu${g}-256-092.bin" 0.92 >"$REPORT_DIR/mgpu${g}-256-092.log" 2>&1
    grep "^MEMORY_PLAN \|^SANITY \|^RESULT " "$REPORT_DIR/mgpu${g}-256-092.log"
    rm -f "$work/mgpu${g}-256-092.bin"
done

echo "MGPU_256_BENCH_DONE"
