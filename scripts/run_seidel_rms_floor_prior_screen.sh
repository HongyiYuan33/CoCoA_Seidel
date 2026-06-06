#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON:-/hdd10tb/hongyi_waller/miniconda3/envs/hybrid_ring/bin/python3.10}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
RUN_PREFIX="${RUN_PREFIX:-seidel_rms_floor_prior_screen_${STAMP}}"

# Screening defaults. For the final size512 reproduction, override:
#   SIZE=512 PRETRAIN_ITER=400 NUM_ITER=1000 bash scripts/run_seidel_rms_floor_prior_screen.sh
SIZE="${SIZE:-256}"
PRETRAIN_ITER="${PRETRAIN_ITER:-200}"
NUM_ITER="${NUM_ITER:-500}"

IMAGES=(${IMAGES:-Iksung_beads dendrites})
DIRECTIONS=(${DIRECTIONS:-cocoa_signed signed_balanced astig_field coma_dominant})
STRENGTHS=(${STRENGTHS:-0.06 0.12 0.20})
CONVENTIONS=(${CONVENTIONS:-classical4d classical5d classical6d})
ALPHAS=(${ALPHAS:-0.8})
WEIGHTS=(${WEIGHTS:-0 10 50})
RMS_PRIOR_MODE="${RMS_PRIOR_MODE:-floor}"

LR_OBJ="${LR_OBJ:-0.01}"
LR_SEIDEL="${LR_SEIDEL:-0.0005}"
RSD_WEIGHT="${RSD_WEIGHT:-0.001}"
TV_WEIGHT="${TV_WEIGHT:-0.0}"
PRETRAIN_SCALAR="${PRETRAIN_SCALAR:-5.0}"
DEFOCUS_ANCHOR_WEIGHT="${DEFOCUS_ANCHOR_WEIGHT:-1.0}"
MAX_VAL="${MAX_VAL:-20.0}"
NERF_BETA="${NERF_BETA:-7.5}"
NERF_DEPTH="${NERF_DEPTH:-6}"
NERF_WIDTH="${NERF_WIDTH:-128}"
NERF_SKIPS="${NERF_SKIPS:-none}"
FOURIER_NUM_ANGLES="${FOURIER_NUM_ANGLES:-60}"
FOURIER_NUM_OCTAVES="${FOURIER_NUM_OCTAVES:-7}"
RMS_FLOOR_FIELD_SAMPLES="${RMS_FLOOR_FIELD_SAMPLES:-21}"
RMS_FLOOR_PUPIL_SAMPLES="${RMS_FLOOR_PUPIL_SAMPLES:-51}"
GT_FIXED_SEIDEL_INDICES=(${GT_FIXED_SEIDEL_INDICES:-})
SEIDEL_LR_MULTIPLIERS_JSON="${SEIDEL_LR_MULTIPLIERS_JSON:-}"

NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
CASE_SUBPROCESS="${CASE_SUBPROCESS:-1}"
SKIP_REPORT="${SKIP_REPORT:-1}"
DRY_RUN="${DRY_RUN:-0}"

tag_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//+/p}"
  echo "$value"
}

echo "[launcher] host=$(hostname) user=$(whoami) root=${PROJECT_ROOT}"
echo "[launcher] run_prefix=${RUN_PREFIX} size=${SIZE} pretrain=${PRETRAIN_ITER} joint=${NUM_ITER}"
echo "[launcher] images=${IMAGES[*]} directions=${DIRECTIONS[*]} strengths=${STRENGTHS[*]}"
echo "[launcher] conventions=${CONVENTIONS[*]} alphas=${ALPHAS[*]} weights=${WEIGHTS[*]} rms_prior_mode=${RMS_PRIOR_MODE}"
echo "[launcher] tuned params lr_obj=${LR_OBJ} lr_seidel=${LR_SEIDEL} rsd=${RSD_WEIGHT} max=${MAX_VAL} beta=${NERF_BETA}"
echo "[launcher] gt_fixed_seidel_indices=${GT_FIXED_SEIDEL_INDICES[*]:-none}"
echo "[launcher] seidel_lr_multipliers_json=${SEIDEL_LR_MULTIPLIERS_JSON:-none}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,pci.bus_id,name,memory.used,memory.total,utilization.gpu --format=csv || true
fi

cd "$PROJECT_ROOT"

for convention in "${CONVENTIONS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    for weight in "${WEIGHTS[@]}"; do
      alpha_tag="$(tag_value "$alpha")"
      weight_tag="$(tag_value "$weight")"
      run_name="${RUN_PREFIX}_${convention}_alpha${alpha_tag}_lambda${weight_tag}"
      cmd=(
        "$PYTHON_BIN" "${SCRIPT_DIR}/run_cocoa_like_seidel_accuracy_sweep.py"
        --run-name "$run_name"
        --stage stage1
        --images "${IMAGES[@]}"
        --directions "${DIRECTIONS[@]}"
        --strengths "${STRENGTHS[@]}"
        --seidel-convention "$convention"
        --stage1-size "$SIZE"
        --stage1-pretrain-iter "$PRETRAIN_ITER"
        --stage1-num-iter "$NUM_ITER"
        --lr-obj "$LR_OBJ"
        --lr-seidel "$LR_SEIDEL"
        --rsd-weight "$RSD_WEIGHT"
        --tv-weight "$TV_WEIGHT"
        --pretrain-scalar "$PRETRAIN_SCALAR"
        --defocus-anchor-weight "$DEFOCUS_ANCHOR_WEIGHT"
        --max-val "$MAX_VAL"
        --nerf-beta "$NERF_BETA"
        --nerf-depth "$NERF_DEPTH"
        --nerf-width "$NERF_WIDTH"
        --nerf-skips "$NERF_SKIPS"
        --fourier-num-angles "$FOURIER_NUM_ANGLES"
        --fourier-num-octaves "$FOURIER_NUM_OCTAVES"
        --seidel-rms-prior-mode "$RMS_PRIOR_MODE"
        --seidel-rms-floor-alpha "$alpha"
        --seidel-rms-floor-weight "$weight"
        --seidel-rms-floor-field-samples "$RMS_FLOOR_FIELD_SAMPLES"
        --seidel-rms-floor-pupil-samples "$RMS_FLOOR_PUPIL_SAMPLES"
        --num-shards "$NUM_SHARDS"
        --shard-index "$SHARD_INDEX"
      )
      if [[ ${#GT_FIXED_SEIDEL_INDICES[@]} -gt 0 ]]; then
        cmd+=(--gt-fixed-seidel-indices "${GT_FIXED_SEIDEL_INDICES[@]}")
      fi
      if [[ -n "$SEIDEL_LR_MULTIPLIERS_JSON" ]]; then
        cmd+=(--seidel-lr-multipliers-json "$SEIDEL_LR_MULTIPLIERS_JSON")
      fi
      if [[ "$CASE_SUBPROCESS" == "1" ]]; then
        cmd+=(--case-subprocess)
      fi
      if [[ "$SKIP_REPORT" == "1" ]]; then
        cmd+=(--skip-report)
      fi
      echo "[run] ${run_name}"
      if [[ "$DRY_RUN" == "1" ]]; then
        printf ' %q' "${cmd[@]}"
        printf '\n'
      else
        "${cmd[@]}"
      fi
    done
  done
done

echo "[launcher] complete"
