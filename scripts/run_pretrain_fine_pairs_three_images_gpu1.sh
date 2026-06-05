#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAMP="${STAMP:-20260604}"
PYTHON_BIN="${PYTHON:-/hdd10tb/hongyi_waller/miniconda3/envs/hybrid_ring/bin/python3.10}"
PHYSICAL_GPU="${PHYSICAL_GPU:-1}"
LOGDIR="${LOGDIR:-${PROJECT_ROOT}/outputs/cocoa_like_2d_mechanism/pretrain_fine_pairs_tunedprior_size256_${STAMP}_logs}"

PAIRS=(
  500:5
  700:5
  600:4
  600:6
  500:6
  700:4
)
CANDIDATES=(
  pre500__scalar5
  pre700__scalar5
  pre600__scalar4
  pre600__scalar6
  pre500__scalar6
  pre700__scalar4
)

mkdir -p "$LOGDIR"
cd "$PROJECT_ROOT"
exec >> "${LOGDIR}/gpu${PHYSICAL_GPU}.log" 2>&1

echo "[start] $(date)"
echo "[identity] host=$(hostname) user=$(whoami) root=${PROJECT_ROOT}"
echo "[gpu] physical_gpu=${PHYSICAL_GPU}"
echo "[pairs] ${PAIRS[*]}"

export PYTHON="$PYTHON_BIN"
export PYTHONPATH=.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=index,pci.bus_id,name,memory.used,memory.total,utilization.gpu --format=csv

run_image() {
  local label="$1"
  local image="$2"
  local run_prefix="pretrain_fine_pairs_tunedprior_noskip6x128_size256_${label}_${STAMP}"

  echo "[image-start] $(date) label=${label} image=${image} run_prefix=${run_prefix}"
  bash scripts/run_pretrain_ablation_pairs.sh \
    --run-prefix "$run_prefix" \
    --image "$image" \
    --size 256 \
    --modes joint frozen \
    --pairs "${PAIRS[@]}" \
    --num-iter 1000 \
    --gt-preset ucla \
    --seidel-convention classical6d \
    --prior tuned-prior \
    --nerf-depth 6 \
    --nerf-width 128 \
    --nerf-skips none

  echo "[eval-start] $(date) run_prefix=${run_prefix}"
  bash scripts/run_operator_eval_for_run_prefix.sh \
    --run-prefix "$run_prefix" \
    --candidates "${CANDIDATES[@]}" \
    --dim 256 \
    --theta-convention classical6d \
    --poll-seconds 180
  echo "[image-done] $(date) label=${label}"
}

run_image Test_figure_1 Test_figure_1
run_image Iksung_beads Iksung_beads
run_image dendrites dendrites

echo "[all-complete] $(date)"
date > "${LOGDIR}/all_complete.marker"
