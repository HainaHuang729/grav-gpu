#!/bin/bash
#SBATCH --job-name=grav-mgpu0056
#SBATCH --partition=a
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm-mgpu1024-0056-%j.log
set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
KERNEL_BUILD_DIR="${KERNEL_BUILD_DIR:?Set KERNEL_BUILD_DIR to a GPU-configured upstream SWIFT build/source directory}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR containing PL_25_1024_0055.hdf5, PL_25_1024_0056.hdf5, fof_output_0056.hdf5}"
HIST_CPU_JSON="${HIST_CPU_JSON:?Set HIST_CPU_JSON to reference historical-cpu.json for 0055->0056}"
REF_CPU_FOF="${REF_CPU_FOF:?Set REF_CPU_FOF to standalone CPU SWIFT fof binary path}"
FOF_YML="${FOF_YML:?Set FOF_YML to SWIFT FoF YAML config path}"
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
work=$(mktemp -d /tmp/grav-mgpu1024-0056.XXXXXX)
SMI_PID=
trap 'if [[ -n "${SMI_PID:-}" ]]; then kill "$SMI_PID" 2>/dev/null || true; fi; rm -rf -- "$work"' EXIT

echo "HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
df -h /tmp
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv >"$REPORT_DIR/mgpu4-1024-0056-device.csv"
nvidia-smi topo -m >>"$REPORT_DIR/mgpu4-1024-0056-device.csv" || true

mkdir -p "$work/build/src"
for f in runner_gpu.h runner_gpu.c gpu_gravity_offload.cu gpu_gravity_functions.cuh kernel_gravity.h kernel_long_gravity.h gravity_derivatives.h multipole.h; do
    cp "$KERNEL_BUILD_DIR/src/$f" "$work/build/src/"
done
cp "$KERNEL_BUILD_DIR/config.h" "$work/build/"
"$PYTHON_BIN" "$REPO_DIR/src/make-evolution.py" "$work/build" "$REPO_DIR"
GRAV_M2P_MATH_ONLY=1 GRAV_GROUP_M2L=1 "$PYTHON_BIN" "$REPO_DIR/m2p/make-port.py" "$work/build"

"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -arch="$CUDA_ARCH" --fmad=true -DGRAV_FAST_M2P_EXP \
    -Xcompiler -fopenmp -Xptxas=-v -ccbin="$CXX_BIN" \
    -I"$work/build" -I"$work/build/src" -I"$KERNEL_BUILD_DIR/src" -I"$REPO_DIR/src" -I"$MATHLIB_INC" \
    -L"$MATHLIB_LIB" \
    "$REPO_DIR/src/multi_gpu_slab.cu" "$REPO_DIR/src/grav_cuda_pm.cu" "$REPO_DIR/src/grav_cuda_pm_dist.cu" "$REPO_DIR/src/grav_cuda_fof.cu" \
    -lcufft -o "$work/multi_gpu_slab" >"$REPORT_DIR/mgpu4-1024-0056-build.log" 2>&1
echo "BUILD_MGPU4_0056_OK"

cd "$work"
"$PYTHON_BIN" "$REPO_DIR/scripts/prepare_1024_input.py" "$DATA_DIR/PL_25_1024_0055.hdf5" "$work/input_0055.bin" 1 >"$REPORT_DIR/mgpu4-1024-0056-input.json"

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv -l 5 >"$REPORT_DIR/mgpu4-1024-0056-gpu-telemetry.csv" &
SMI_PID=$!

OMP_NUM_THREADS=4 GRAV_NUM_GPUS=4 GRAV_P2P_MODE=nvlink GRAV_UNLINK_INPUT_AFTER_LOAD=1 \
    GRAV_DYNAMIC_PM=1 GRAV_LEAF_LIMIT=64 GRAV_G=43.0092 GRAV_WARP_M2L=1 GRAV_FAST_SPLIT=1 \
    GRAV_FORCE3_OUT="$work/force3-0056.bin" \
    stdbuf -oL -eL "$work/multi_gpu_slab" "$work/input_0055.bin" "$work/evolved-0056.bin" 1.0 \
    >"$REPORT_DIR/mgpu4-1024-0056-evolve.log" 2>&1

kill "$SMI_PID" 2>/dev/null || true
SMI_PID=
grep -E "^MEMORY_PLAN |^GPU_INPUT |^SANITY |^RESULT " "$REPORT_DIR/mgpu4-1024-0056-evolve.log" || true
rm -f "$work/input_0055.bin"

t0=$(date +%s.%N)
"$PYTHON_BIN" "$REPO_DIR/scripts/to-hdf5-from-source.py" "$DATA_DIR/PL_25_1024_0055.hdf5" \
    "$work/evolved-0056.bin" "$work/snapshot_0056.hdf5" \
    >"$REPORT_DIR/mgpu4-1024-0056-tohdf5.log" 2>&1
t1=$(date +%s.%N)
"$PYTHON_BIN" -c "print(f'{float(${t1}) - float(${t0}):.6f}')" >"$REPORT_DIR/mgpu4-1024-0056-tohdf5.seconds"
rm -f "$work/evolved-0056.bin"

cp "$FOF_YML" "$work/fof.yml"
t0=$(date +%s.%N)
"$REF_CPU_FOF" --cosmology --threads="${SLURM_CPUS_PER_TASK:-64}" --verbose=1 fof.yml \
    >"$REPORT_DIR/mgpu4-1024-0056-fof.log" 2>&1
t1=$(date +%s.%N)
"$PYTHON_BIN" -c "print(f'{float(${t1}) - float(${t0}):.6f}')" >"$REPORT_DIR/mgpu4-1024-0056-fof.seconds"

gpu_fof=$(ls -1 "$work"/fof_output_*.hdf5 | head -n 1)
"$PYTHON_BIN" "$REPO_DIR/scripts/eval_1024_snapshot.py" \
    "$DATA_DIR/PL_25_1024_0056.hdf5" \
    "$work/snapshot_0056.hdf5" \
    "$DATA_DIR/fof_output_0056.hdf5" \
    "$gpu_fof" \
    "$REPORT_DIR/mgpu4-1024-0056-evolve.log" \
    "$REPORT_DIR/mgpu4-1024-0056-fof.seconds" \
    "$REPORT_DIR/mgpu4-1024-0056-tohdf5.seconds" \
    "$HIST_CPU_JSON" \
    "$REPORT_DIR/mgpu4-1024-0056" \
    --force3-bin "$work/force3-0056.bin" \
    --grav-g 43.0092 \
    >"$REPORT_DIR/mgpu4-1024-0056-accuracy.json"
cat "$REPORT_DIR/mgpu4-1024-0056-accuracy.json"
echo "MGPU4_1024_0056_E2E_ALL_DONE"
