#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAMP="${STAMP:-20260604}"
PYTHON_BIN="${PYTHON:-/hdd10tb/hongyi_waller/miniconda3/envs/hybrid_ring/bin/python3.10}"
IMAGE="${IMAGE:-dendrites_dense}"
SIZE="${SIZE:-256}"
LOGDIR="${LOGDIR:-${PROJECT_ROOT}/outputs/cocoa_like_2d_mechanism/dendrites_dense_trend_sweeps_tunedprior_size256_${STAMP}_logs}"

PRETRAIN_PREFIX="pretrain_union_tunedprior_noskip6x128_size256_${IMAGE}_${STAMP}"
FOURIER_PREFIX="fourier_union_tunedprior_noskip6x128_pre400scalar5_size256_${IMAGE}_${STAMP}"
CAPACITY_PREFIX="capacity_union_tunedprior_size256_${IMAGE}_${STAMP}"

PRETRAIN_PAIRS=(
  400:5
  400:3
  400:1
  200:5
  200:3
  200:1
  100:5
  100:3
  100:1
  600:5
  800:5
  400:7.5
  600:7.5
  400:10
  500:5
  700:5
  600:4
  600:6
  500:6
  700:4
)
PRETRAIN_CANDIDATES=(
  pre400__scalar5
  pre400__scalar3
  pre400__scalar1
  pre200__scalar5
  pre200__scalar3
  pre200__scalar1
  pre100__scalar5
  pre100__scalar3
  pre100__scalar1
  pre600__scalar5
  pre800__scalar5
  pre400__scalar7p5
  pre600__scalar7p5
  pre400__scalar10
  pre500__scalar5
  pre700__scalar5
  pre600__scalar4
  pre600__scalar6
  pre500__scalar6
  pre700__scalar4
)

FOURIER_CANDIDATES=(
  oct7_ang60
  oct5_ang60
  oct4_ang60
  oct3_ang60
  oct7_ang30
  oct7_ang16
  oct8_ang60
  oct9_ang60
  oct7_ang90
  oct8_ang30
  oct8_ang90
)
FOURIER_GPU0=(
  oct7_ang60
  oct4_ang60
  oct7_ang30
  oct8_ang60
  oct7_ang90
  oct8_ang90
)
FOURIER_GPU1=(
  oct5_ang60
  oct3_ang60
  oct7_ang16
  oct9_ang60
  oct8_ang30
)

CAPACITY_CANDIDATES=(
  baseline
  depth_only
  width_only
  medium
  low
  skip_only
  noskip_6x128
  noskip_4x128
  noskip_6x64
  noskip_4x64
  noskip_3x32
)
CAPACITY_GPU0=(
  baseline
  width_only
  low
  noskip_4x128
  noskip_4x64
)
CAPACITY_GPU1=(
  depth_only
  medium
  skip_only
  noskip_6x128
  noskip_6x64
  noskip_3x32
)

mkdir -p "$LOGDIR"
cd "$PROJECT_ROOT"
exec >> "${LOGDIR}/watcher.log" 2>&1

echo "[start] $(date)"
echo "[identity] host=$(hostname) user=$(whoami) root=${PROJECT_ROOT}"
echo "[image] ${IMAGE} size=${SIZE}"
echo "[prefixes] pretrain=${PRETRAIN_PREFIX} fourier=${FOURIER_PREFIX} capacity=${CAPACITY_PREFIX}"

run_gpu_shard() {
  local physical_gpu="$1"
  local shard_index="$2"
  local log_path="$3"
  shift 3
  local fourier_profiles=()
  local capacity_profiles=()
  local mode="fourier"
  for arg in "$@"; do
    if [[ "$arg" == "--capacity" ]]; then
      mode="capacity"
      continue
    fi
    if [[ "$mode" == "fourier" ]]; then
      fourier_profiles+=("$arg")
    else
      capacity_profiles+=("$arg")
    fi
  done

  (
    set -euo pipefail
    cd "$PROJECT_ROOT"
    exec >> "$log_path" 2>&1
    echo "[gpu-start] $(date) physical_gpu=${physical_gpu} shard=${shard_index}/2"
    echo "[identity] host=$(hostname) user=$(whoami)"
    echo "[fourier-profiles] ${fourier_profiles[*]}"
    echo "[capacity-profiles] ${capacity_profiles[*]}"

    export PYTHON="$PYTHON_BIN"
    export PYTHONPATH=.
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$physical_gpu"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    nvidia-smi --query-gpu=index,pci.bus_id,name,memory.used,memory.total,utilization.gpu --format=csv

    bash scripts/run_pretrain_ablation_pairs.sh \
      --run-prefix "$PRETRAIN_PREFIX" \
      --image "$IMAGE" \
      --size "$SIZE" \
      --modes joint frozen \
      --pairs "${PRETRAIN_PAIRS[@]}" \
      --num-iter 1000 \
      --gt-preset ucla \
      --seidel-convention classical6d \
      --prior tuned-prior \
      --nerf-depth 6 \
      --nerf-width 128 \
      --nerf-skips none \
      --num-shards 2 \
      --shard-index "$shard_index"

    bash scripts/run_fourier_encoding_ablation_profiles.sh \
      --run-prefix "$FOURIER_PREFIX" \
      --image "$IMAGE" \
      --size "$SIZE" \
      --modes joint frozen \
      --pretrain-iter 400 \
      --pretrain-scalar 5 \
      --num-iter 1000 \
      --gt-preset ucla \
      --seidel-convention classical6d \
      --prior tuned-prior \
      --nerf-depth 6 \
      --nerf-width 128 \
      --nerf-skips none \
      --profiles "${fourier_profiles[@]}"

    bash scripts/run_mlp_capacity_ablation_profiles.sh \
      --run-prefix "$CAPACITY_PREFIX" \
      --image "$IMAGE" \
      --size "$SIZE" \
      --modes joint frozen \
      --pretrain-iter 400 \
      --num-iter 1000 \
      --gt-preset ucla \
      --seidel-convention classical6d \
      --prior tuned-prior \
      --profiles "${capacity_profiles[@]}"

    echo "[gpu-done] $(date) physical_gpu=${physical_gpu}"
  )
}

run_eval() {
  local log_path="$1"
  (
    set -euo pipefail
    cd "$PROJECT_ROOT"
    exec >> "$log_path" 2>&1
    echo "[eval-start] $(date)"
    export PYTHON="$PYTHON_BIN"
    export PYTHONPATH=.
    export CUDA_VISIBLE_DEVICES=
    bash scripts/run_operator_eval_for_run_prefix.sh \
      --run-prefix "$PRETRAIN_PREFIX" \
      --candidates "${PRETRAIN_CANDIDATES[@]}" \
      --dim "$SIZE" \
      --theta-convention classical6d \
      --poll-seconds 180
    bash scripts/run_operator_eval_for_run_prefix.sh \
      --run-prefix "$FOURIER_PREFIX" \
      --candidates "${FOURIER_CANDIDATES[@]}" \
      --dim "$SIZE" \
      --theta-convention classical6d \
      --poll-seconds 180
    bash scripts/run_operator_eval_for_run_prefix.sh \
      --run-prefix "$CAPACITY_PREFIX" \
      --candidates "${CAPACITY_CANDIDATES[@]}" \
      --dim "$SIZE" \
      --theta-convention classical6d \
      --poll-seconds 180
    echo "[eval-done] $(date)"
  )
}

echo "[launch] $(date)"
run_gpu_shard 0 0 "${LOGDIR}/gpu0.log" "${FOURIER_GPU0[@]}" --capacity "${CAPACITY_GPU0[@]}" &
pid0=$!
run_gpu_shard 1 1 "${LOGDIR}/gpu1.log" "${FOURIER_GPU1[@]}" --capacity "${CAPACITY_GPU1[@]}" &
pid1=$!
run_eval "${LOGDIR}/operator_eval.log" &
pide=$!

set +e
wait "$pid0"
st0=$?
wait "$pid1"
st1=$?
set -e
if (( st0 != 0 || st1 != 0 )); then
  echo "[error] gpu shard failed: gpu0=${st0} gpu1=${st1}; stopping evaluator"
  kill "$pide" 2>/dev/null || true
  wait "$pide" 2>/dev/null || true
  exit 1
fi

wait "$pide"
ste=$?
if (( ste != 0 )); then
  echo "[error] evaluator failed: ${ste}"
  exit "$ste"
fi

echo "[all-complete] $(date)"
date > "${LOGDIR}/all_complete.marker"
