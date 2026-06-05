#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAMP="${STAMP:-20260604}"
PYTHON_BIN="${PYTHON:-/hdd10tb/hongyi_waller/miniconda3/envs/hybrid_ring/bin/python3.10}"
POLL_SECONDS="${POLL_SECONDS:-300}"
GPU_FREE_MIB="${GPU_FREE_MIB:-500}"
LOGDIR="${LOGDIR:-${PROJECT_ROOT}/outputs/cocoa_like_2d_mechanism/fourier_encoding_ablation_tunedprior_size256_${STAMP}_logs}"

PRE_T1="pretrain_pairs_ext_tunedprior_noskip6x128_size256_Test_figure_1_${STAMP}"
PRE_IKS="pretrain_pairs_ext_tunedprior_noskip6x128_size256_Iksung_beads_${STAMP}"
PRE_DEN="pretrain_pairs_ext_tunedprior_noskip6x128_size256_dendrites_${STAMP}"

NEW_T1="fourier_encoding_ablation_tunedprior_noskip6x128_pre400scalar5_size256_Test_figure_1_${STAMP}"
NEW_IKS="fourier_encoding_ablation_tunedprior_noskip6x128_pre400scalar5_size256_Iksung_beads_${STAMP}"
NEW_DEN="fourier_encoding_ablation_tunedprior_noskip6x128_pre400scalar5_size256_dendrites_${STAMP}"

CANDIDATES=(oct7_ang60 oct5_ang60 oct4_ang60 oct3_ang60 oct7_ang30 oct7_ang16)
GPU0_PROFILES=(oct7_ang60 oct3_ang60 oct7_ang16)
GPU1_PROFILES=(oct5_ang60 oct4_ang60 oct7_ang30)

mkdir -p "$LOGDIR"
cd "$PROJECT_ROOT"
exec >> "${LOGDIR}/watcher.log" 2>&1

echo "[watch-start] $(date)"
echo "[identity] host=$(hostname) user=$(whoami) root=${PROJECT_ROOT}"
echo "[wait-target] selected-pairs evaluator for ${PRE_T1}, ${PRE_IKS}, ${PRE_DEN}"

selected_pair_eval_paths=(
  "outputs/cocoa_like_2d_mechanism/${PRE_T1}_operator_eval_dim256/seidel_physical_operator_metrics.csv"
  "outputs/cocoa_like_2d_mechanism/${PRE_IKS}_operator_eval_dim256/seidel_physical_operator_metrics.csv"
  "outputs/cocoa_like_2d_mechanism/${PRE_DEN}_operator_eval_dim256/seidel_physical_operator_metrics.csv"
)

while true; do
  complete=0
  for path in "${selected_pair_eval_paths[@]}"; do
    if [[ -f "$path" ]]; then
      complete=$((complete + 1))
    fi
  done
  echo "[waiting-selected-pairs-eval] $(date) complete=${complete}/${#selected_pair_eval_paths[@]}"
  find outputs/cocoa_like_2d_mechanism \
    -path "*pretrain_pairs_ext_tunedprior_noskip6x128_size256*${STAMP}_operator_eval_dim256/seidel_physical_operator_metrics.csv" \
    -print | sed 's#^#  #'
  if [[ "$complete" -eq "${#selected_pair_eval_paths[@]}" ]]; then
    echo "[selected-pairs-eval-complete] $(date)"
    break
  fi
  sleep "$POLL_SECONDS"
done

gpu_mem_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9'
}

while true; do
  used0="$(gpu_mem_mib 0)"
  used1="$(gpu_mem_mib 1)"
  echo "[gpu-check] $(date) gpu0=${used0}MiB gpu1=${used1}MiB"
  if (( used0 < GPU_FREE_MIB && used1 < GPU_FREE_MIB )); then
    echo "[gpus-free] $(date)"
    break
  fi
  sleep "$POLL_SECONDS"
done

run_image_profiles() {
  local run_prefix="$1"
  local image="$2"
  shift 2
  local profiles=("$@")

  bash scripts/run_fourier_encoding_ablation_profiles.sh \
    --run-prefix "$run_prefix" \
    --image "$image" \
    --size 256 \
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
    --profiles "${profiles[@]}"
}

run_gpu_shard() {
  local physical_gpu="$1"
  local log_path="$2"
  shift 2
  local profiles=("$@")
  (
    set -euo pipefail
    cd "$PROJECT_ROOT"
    exec >> "$log_path" 2>&1
    echo "[start] $(date) physical_gpu=${physical_gpu} profiles=${profiles[*]}"
    echo "[identity] host=$(hostname) user=$(whoami)"
    export PYTHON="$PYTHON_BIN"
    export PYTHONPATH=.
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$physical_gpu"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    nvidia-smi --query-gpu=index,pci.bus_id,name,memory.used,memory.total,utilization.gpu --format=csv
    run_image_profiles "$NEW_T1" Test_figure_1 "${profiles[@]}"
    run_image_profiles "$NEW_IKS" Iksung_beads "${profiles[@]}"
    run_image_profiles "$NEW_DEN" dendrites "${profiles[@]}"
    echo "[done] $(date) physical_gpu=${physical_gpu}"
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
      --run-prefix "$NEW_T1" \
      --candidates "${CANDIDATES[@]}" \
      --dim 256 \
      --theta-convention classical6d \
      --poll-seconds 180
    bash scripts/run_operator_eval_for_run_prefix.sh \
      --run-prefix "$NEW_IKS" \
      --candidates "${CANDIDATES[@]}" \
      --dim 256 \
      --theta-convention classical6d \
      --poll-seconds 180
    bash scripts/run_operator_eval_for_run_prefix.sh \
      --run-prefix "$NEW_DEN" \
      --candidates "${CANDIDATES[@]}" \
      --dim 256 \
      --theta-convention classical6d \
      --poll-seconds 180
    echo "[eval-done] $(date)"
  )
}

echo "[launch-fourier-ablation] $(date)"
run_gpu_shard 0 "${LOGDIR}/gpu0.log" "${GPU0_PROFILES[@]}" &
pid0=$!
run_gpu_shard 1 "${LOGDIR}/gpu1.log" "${GPU1_PROFILES[@]}" &
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
