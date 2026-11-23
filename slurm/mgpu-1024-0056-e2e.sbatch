#!/bin/bash
#SBATCH --job-name=swift-mgpu0056
#SBATCH --partition=a
#SBATCH --nodelist=a9
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=05:00:00
#SBATCH --output=/home/user/hnhuang/swift_device_treepm_256_20261118/evolution/single-gpu-opt/reports/slurm-mgpu1024-0056-%j.log
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

work=$(mktemp -d /tmp/swift-mgpu1024-0056.XXXXXX)
SMI_PID=
trap 'if [[ -n "${SMI_PID:-}" ]]; then kill "$SMI_PID" 2>/dev/null || true; fi; rm -rf -- "$work"' EXIT

echo "HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
df -h /tmp
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv >"$opt/reports/mgpu4-1024-0056-device.csv"
nvidia-smi topo -m >>"$opt/reports/mgpu4-1024-0056-device.csv" || true

test "$(stat -c %s "$tmp1024/PL_25_1024_0055.hdf5")" = 36297437824
test "$(stat -c %s "$tmp1024/PL_25_1024_0056.hdf5")" = 36276517058
test "$(stat -c %s "$tmp1024/fof_output_0056.hdf5")" = 1118279196

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
    -o "$work/multi_gpu_slab" >"$opt/reports/mgpu4-1024-0056-build.log" 2>&1
echo "BUILD_MGPU4_0056_OK"

cd "$work"
echo "$(date -Is) Packing 1024^3 input_0055.bin into $work"
"$py" "$opt/prepare_1024_input.py" "$tmp1024/PL_25_1024_0055.hdf5" "$work/input_0055.bin" 1 >"$opt/reports/mgpu4-1024-0056-input.json"
test "$(stat -c %s "$work/input_0055.bin")" = 51539607584

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv -l 5 >"$opt/reports/mgpu4-1024-0056-gpu-telemetry.csv" &
SMI_PID=$!

echo "$(date -Is) Starting 4-GPU 1024^3 evolution 0055 -> 0056 (a_end=1.0, p2p=nvlink)"
OMP_NUM_THREADS=4 SWIFT_NUM_GPUS=4 SWIFT_P2P_MODE=nvlink SWIFT_UNLINK_INPUT_AFTER_LOAD=1 \
    SWIFT_DYNAMIC_PM=1 SWIFT_LEAF_LIMIT=64 SWIFT_GRAV_G=43.0092 SWIFT_WARP_M2L=1 SWIFT_FAST_SPLIT=1 \
    SWIFT_FORCE3_OUT="$work/force3-0056.bin" \
    /usr/bin/time -v -o "$opt/reports/mgpu4-1024-0056-evolve.time" \
    stdbuf -oL -eL "$work/multi_gpu_slab" "$work/input_0055.bin" "$work/evolved-0056.bin" 1.0 \
    >"$opt/reports/mgpu4-1024-0056-evolve.log" 2>&1

kill "$SMI_PID" 2>/dev/null || true
SMI_PID=
echo "$(date -Is) Finished 4-GPU evolution 0055 -> 0056"
test "$(stat -c %s "$work/evolved-0056.bin")" = 42949672984
test "$(stat -c %s "$work/force3-0056.bin")" = 25769803776
grep -E "^MEMORY_PLAN |^GPU_INPUT |^SANITY |^RESULT " "$opt/reports/mgpu4-1024-0056-evolve.log" || true
rm -f "$work/input_0055.bin"
df -h /tmp

echo "$(date -Is) Converting evolved-0056.bin -> snapshot_0056.hdf5"
t0=$(date +%s.%N)
/usr/bin/time -v -o "$opt/reports/mgpu4-1024-0056-tohdf5.time" \
    "$py" "$opt/to-hdf5-from-source.py" "$tmp1024/PL_25_1024_0055.hdf5" \
    "$work/evolved-0056.bin" "$work/snapshot_0056.hdf5" \
    >"$opt/reports/mgpu4-1024-0056-tohdf5.log" 2>&1
t1=$(date +%s.%N)
python3 -c "print(f'{float(${t1}) - float(${t0}):.6f}')" >"$opt/reports/mgpu4-1024-0056-tohdf5.seconds"
rm -f "$work/evolved-0056.bin"
df -h /tmp

lib="$probe/deps/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$lib:$lib/hdf5/serial:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1
cp "$root/evolution/final-fof/fof.yml" "$work/fof.yml"
echo "$(date -Is) Running standalone native SWIFT FoF on ${SLURM_CPUS_PER_TASK:-64} threads"
t0=$(date +%s.%N)
/usr/bin/time -v -o "$opt/reports/mgpu4-1024-0056-fof.time" \
    "$probe/build-cpu/fof" --cosmology --threads="${SLURM_CPUS_PER_TASK:-64}" --verbose=1 fof.yml \
    >"$opt/reports/mgpu4-1024-0056-fof.log" 2>&1
t1=$(date +%s.%N)
python3 -c "print(f'{float(${t1}) - float(${t0}):.6f}')" >"$opt/reports/mgpu4-1024-0056-fof.seconds"
echo "$(date -Is) Finished standalone SWIFT FoF"

gpu_fof=$(ls -1 "$work"/fof_output_*.hdf5 | head -n 1)
echo "$(date -Is) Running full 1024^3 scientific accuracy + kick-aligned P_theta evaluation ($gpu_fof)"
"$py" "$opt/eval_1024_snapshot.py" \
    "$tmp1024/PL_25_1024_0056.hdf5" \
    "$work/snapshot_0056.hdf5" \
    "$tmp1024/fof_output_0056.hdf5" \
    "$gpu_fof" \
    "$opt/reports/mgpu4-1024-0056-evolve.log" \
    "$opt/reports/mgpu4-1024-0056-fof.seconds" \
    "$opt/reports/mgpu4-1024-0056-tohdf5.seconds" \
    "$root/evolution/final-fof/historical-cpu.json" \
    "$opt/reports/mgpu4-1024-0056" \
    --force3-bin "$work/force3-0056.bin" \
    --grav-g 43.0092 \
    >"$opt/reports/mgpu4-1024-0056-accuracy.json"
cat "$opt/reports/mgpu4-1024-0056-accuracy.json"

rm -f "$work/snapshot_0056.hdf5" "$work/force3-0056.bin"
echo "MGPU4_1024_0056_E2E_ALL_DONE"
