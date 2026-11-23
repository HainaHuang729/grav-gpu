#!/bin/bash
#SBATCH --job-name=swift-p2pall
#SBATCH --partition=a
#SBATCH --nodelist=a9
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/home/user/hnhuang/swift_device_treepm_256_20261118/evolution/single-gpu-opt/reports/slurm-mgpu1024-p2pall-%j.log
set -euo pipefail
root="$HOME/swift_device_treepm_256_20261118"
opt="$root/evolution/single-gpu-opt"
tmp1024="$opt/data/tmp1024"
probe="$HOME/swift1024_production_probe_20261118"
base="$probe/build-gpu"
cuda=/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/cuda/12.6
mathlib=/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/math_libs/12.6/targets/x86_64-linux
export LD_LIBRARY_PATH="$cuda/lib64:$mathlib/lib"
export PYTHONPATH="$HOME/ic_gpu_trial_20261117/vendor"
export OPENBLAS_NUM_THREADS=1
py="$HOME/jfof_trial_20261116/env/bin/python"

work=$(mktemp -d /tmp/swift-p2pall.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

echo "HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
test "$(stat -c %s "$tmp1024/PL_25_1024_0055.hdf5")" = 36297437824

mkdir -p "$work/build/src"
for f in runner_gpu.h runner_gpu.c gpu_gravity_offload.cu gpu_gravity_functions.cuh kernel_gravity.h kernel_long_gravity.h gravity_derivatives.h multipole.h; do
    cp "$base/src/$f" "$work/build/src/"
done
cp "$base/config.h" "$work/build/"
python3 "$opt/src/make-evolution.py" "$work/build" "$root"
SWIFT_M2P_MATH_ONLY=1 SWIFT_GROUP_M2L=1 python3 "$root/m2p/make-port.py" "$work/build"

"$cuda/bin/nvcc" -O3 -std=c++17 -arch=sm_86 --fmad=true -DSWIFT_FAST_M2P_EXP \
    -Xcompiler -fopenmp -ccbin=/usr/bin/g++-13 \
    -I"$work/build" -I"$work/build/src" -I"$base/src" -I"$opt/src" -I"$root" -I"$root/evolution" -I"$mathlib/include" \
    -L"$mathlib/lib" "$opt/src/multi_gpu_slab.cu" "$opt/src/swift_cuda_pm.cu" -lcufft \
    -o "$work/multi_gpu_slab"
echo "BUILD_P2PALL_OK"

cd "$work"
"$py" "$opt/prepare_1024_input.py" "$tmp1024/PL_25_1024_0055.hdf5" "$work/input_0055.bin" 1 >"$opt/reports/mgpu4-1024-0055-p2pall-input.json"
test "$(stat -c %s "$work/input_0055.bin")" = 51539607584

echo "$(date -Is) Starting 4-GPU 1024^3 SWIFT_P2P_MODE=all probe on 0055 -> 0.9425"
OMP_NUM_THREADS=4 SWIFT_NUM_GPUS=4 SWIFT_P2P_MODE=all SWIFT_UNLINK_INPUT_AFTER_LOAD=1 \
    SWIFT_SKIP_OUTPUT_WRITE=1 SWIFT_DYNAMIC_PM=1 SWIFT_LEAF_LIMIT=64 \
    SWIFT_GRAV_G=43.0092 SWIFT_WARP_M2L=1 SWIFT_FAST_SPLIT=1 \
    /usr/bin/time -v -o "$opt/reports/mgpu4-1024-0055-p2pall-probe.time" \
    stdbuf -oL -eL "$work/multi_gpu_slab" "$work/input_0055.bin" "$work/out.bin" 0.9425 \
    >"$opt/reports/mgpu4-1024-0055-p2pall-probe.log" 2>&1

grep -E "^MEMORY_PLAN |^GPU_INPUT |^SANITY |^RESULT " "$opt/reports/mgpu4-1024-0055-p2pall-probe.log"
echo "P2P_ALL_PROBE_DONE"
