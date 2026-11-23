#!/bin/bash
#SBATCH --job-name=swift-mgpu256
#SBATCH --partition=a
#SBATCH --nodelist=a9
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/home/user/hnhuang/swift_device_treepm_256_20261118/evolution/single-gpu-opt/reports/slurm-mgpu256-%j.log
set -euo pipefail
root="$HOME/swift_device_treepm_256_20261118"
opt="$root/evolution/single-gpu-opt"
base="$HOME/swift1024_production_probe_20261118/build-gpu"
cuda=/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/cuda/12.6
mathlib=/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/math_libs/12.6/targets/x86_64-linux
export LD_LIBRARY_PATH="$cuda/lib64:$mathlib/lib"
export PYTHONPATH="$HOME/ic_gpu_trial_20261117/vendor"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16
py="$HOME/jfof_trial_20261116/env/bin/python"

work=$(mktemp -d /tmp/swift-mgpu256.XXXXXX)
trap 'rm -rf -- "$work"' EXIT
echo "HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv >"$opt/reports/mgpu-256-device.csv"
nvidia-smi topo -m >>"$opt/reports/mgpu-256-device.csv" || true

mkdir -p "$work/build/src"
for f in runner_gpu.h runner_gpu.c gpu_gravity_offload.cu gpu_gravity_functions.cuh kernel_gravity.h kernel_long_gravity.h gravity_derivatives.h multipole.h; do
    cp "$base/src/$f" "$work/build/src/"
done
cp "$base/config.h" "$work/build/"
python3 "$opt/src/make-evolution.py" "$work/build" "$root"
SWIFT_M2P_MATH_ONLY=1 SWIFT_GROUP_M2L=1 python3 "$root/m2p/make-port.py" "$work/build"

"$cuda/bin/nvcc" -O3 -std=c++17 -arch=sm_86 --fmad=true -DSWIFT_FAST_M2P_EXP \
    -Xcompiler -fopenmp -Xptxas=-v -ccbin=/usr/bin/g++-13 \
    -I"$work/build" -I"$work/build/src" -I"$base/src" -I"$opt/src" -I"$root" -I"$root/evolution" -I"$mathlib/include" \
    -L"$mathlib/lib" "$opt/src/multi_gpu_slab.cu" "$opt/src/swift_cuda_pm.cu" -lcufft \
    -o "$work/multi_gpu_slab" >"$opt/reports/mgpu-256-build.log" 2>&1
echo "BUILD_MGPU_OK"

# Also check backward-compatible slab_resident compile
"$cuda/bin/nvcc" -O3 -std=c++17 -arch=sm_86 --fmad=true -DSWIFT_FAST_M2P_EXP \
    -Xcompiler -fopenmp -ccbin=/usr/bin/g++-13 \
    -I"$work/build" -I"$work/build/src" -I"$base/src" -I"$opt/src" -I"$root" -I"$root/evolution" -I"$mathlib/include" \
    -L"$mathlib/lib" "$opt/src/slab_resident.cu" "$opt/src/swift_cuda_pm.cu" -lcufft \
    -o "$work/slab_resident" >>"$opt/reports/mgpu-256-build.log" 2>&1
echo "BUILD_SLAB_BACKCOMPAT_OK"

cd "$work"
"$py" "$root/evolution/prepare.py" "$opt/data/PL_25_256_0054.hdf5" input.bin full >"$opt/reports/mgpu-256-input.json"

for g in 1 2 4; do
    echo "=== Running SWIFT_NUM_GPUS=$g ==="
    export SWIFT_NUM_GPUS="$g"
    export SWIFT_LEAF_LIMIT=64
    export SWIFT_DYNAMIC_PM=1
    export SWIFT_GRAV_G=43.0092
    export SWIFT_WARP_M2L=1
    export SWIFT_FAST_SPLIT=1
    export SWIFT_FORCE_COMPONENTS="$work/mgpu${g}-forces.bin"
    /usr/bin/time -v -o "$opt/reports/mgpu${g}-256-092.time" stdbuf -oL -eL \
        ./multi_gpu_slab input.bin "$work/mgpu${g}-256-092.bin" 0.92 >"$opt/reports/mgpu${g}-256-092.log" 2>&1
    grep "^MEMORY_PLAN \|^SANITY \|^RESULT " "$opt/reports/mgpu${g}-256-092.log"

    "$py" "$opt/eval_accuracy.py" \
        --input "$opt/data/PL_25_256_0054.hdf5" \
        --gpu-bin "$work/mgpu${g}-256-092.bin" \
        --forces-bin "$work/mgpu${g}-forces.bin" \
        --cpu-hdf5 "$opt/data/native-092.hdf5" \
        --gpu-log "$opt/reports/mgpu${g}-256-092.log" \
        --grav-g 43.0092 \
        --out "$opt/reports/mgpu${g}-256-092-acc.json"

    rm -f "$work/mgpu${g}-256-092.bin" "$work/mgpu${g}-forces.bin"
done

echo "MGPU_256_GATE1_DONE"
