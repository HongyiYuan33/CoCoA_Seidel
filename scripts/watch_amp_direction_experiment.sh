#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/hdd10tb/hongyi_waller/miniconda3/envs/hybrid_ring/bin/python3.10}"

RUN_PREFIX="${RUN_PREFIX:?RUN_PREFIX is required}"
EXPECTED="${EXPECTED:-108}"
POLL_SECONDS="${POLL_SECONDS:-120}"
DIM="${DIM:-256}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cocoa_like_2d_mechanism}"
CONTROL_CSV="${CONTROL_CSV:-${OUTPUT_ROOT}/seidel_ratio_target_prior_6d_signed_balanced_3imgs_rms006_020_040_alpha1_lambdas1000_10000_pre400_joint1000_20260606_operator_eval_dim256/seidel_physical_operator_metrics.csv}"
TRAIN_SESSION_GLOB="${TRAIN_SESSION_GLOB:-ampdir_gpu.*_20260606}"

cd "$PROJECT_ROOT"

LOG_DIR="${OUTPUT_ROOT}/${RUN_PREFIX}_logs"
mkdir -p "$LOG_DIR"
STATUS_FILE="${LOG_DIR}/watcher_status.txt"

count_metrics() {
  find "$OUTPUT_ROOT" \
    -path "*${RUN_PREFIX}*/stage1/*/joint/metrics.json" \
    | wc -l \
    | tr -d ' '
}

active_train_sessions() {
  tmux list-sessions 2>/dev/null | grep -Ec "${TRAIN_SESSION_GLOB}" || true
}

echo "[watcher] start $(date)" | tee -a "$STATUS_FILE"
echo "[watcher] run_prefix=${RUN_PREFIX} expected=${EXPECTED} dim=${DIM}" | tee -a "$STATUS_FILE"

while true; do
  count="$(count_metrics)"
  active="$(active_train_sessions)"
  echo "[watcher] $(date) metrics=${count}/${EXPECTED} active_sessions=${active}" | tee -a "$STATUS_FILE"
  if [[ "$count" -ge "$EXPECTED" ]]; then
    break
  fi
  if [[ "$active" -eq 0 ]]; then
    echo "[watcher] ERROR: train sessions ended before expected metrics count" | tee -a "$STATUS_FILE"
    exit 2
  fi
  sleep "$POLL_SECONDS"
done

run_roots=()
while IFS= read -r path; do
  run_roots+=("$(basename "$path")")
done < <(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "${RUN_PREFIX}_amp_direction*_classical6d_alpha1_lambda*" | sort)

if [[ "${#run_roots[@]}" -eq 0 ]]; then
  echo "[watcher] ERROR: no run roots matched ${RUN_PREFIX}" | tee -a "$STATUS_FILE"
  exit 3
fi

input_csv="${OUTPUT_ROOT}/${RUN_PREFIX}_operator_eval_input.csv"
eval_dir="${OUTPUT_ROOT}/${RUN_PREFIX}_operator_eval_dim${DIM}"
eval_csv="${eval_dir}/seidel_physical_operator_metrics.csv"
stats_dir="${OUTPUT_ROOT}/${RUN_PREFIX}_stats_vs_ratio_target_control"
rcp_dir="${OUTPUT_ROOT}/${RUN_PREFIX}_RCP_best_vs_ratio_target_control"

build_args=()
for root in "${run_roots[@]}"; do
  build_args+=(--run-root "$root")
done

echo "[watcher] build evaluator input for ${#run_roots[@]} run roots" | tee -a "$STATUS_FILE"
PYTHONPATH=. "$PYTHON_BIN" scripts/build_rms_floor_operator_eval_input.py \
  "${build_args[@]}" \
  --out "$input_csv" \
  --expected "$EXPECTED" \
  2>&1 | tee -a "$STATUS_FILE"

echo "[watcher] run operator evaluator" | tee -a "$STATUS_FILE"
PYTHONPATH=. "$PYTHON_BIN" scripts/evaluate_seidel_physical_operator_sweep.py \
  "$input_csv" \
  "$eval_dir" \
  --dim "$DIM" \
  2>&1 | tee -a "$STATUS_FILE"

echo "[watcher] plot stats" | tee -a "$STATUS_FILE"
PYTHONPATH=. "$PYTHON_BIN" scripts/plot_amp_direction_comparison.py \
  --control-csv "$CONTROL_CSV" \
  --new-csv "$eval_csv" \
  --out-dir "$stats_dir" \
  2>&1 | tee -a "$STATUS_FILE"

echo "[watcher] build best RCP pairs" | tee -a "$STATUS_FILE"
PYTHONPATH=. "$PYTHON_BIN" scripts/build_amp_direction_best_rcp_pairs.py \
  --control-csv "$CONTROL_CSV" \
  --new-csv "$eval_csv" \
  --output-root "$OUTPUT_ROOT" \
  --out-dir "$rcp_dir" \
  2>&1 | tee -a "$STATUS_FILE"

echo "[watcher] complete $(date)" | tee -a "$STATUS_FILE"
