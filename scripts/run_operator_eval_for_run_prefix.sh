#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_PREFIX=""
DIM="256"
THETA_CONVENTION="classical6d"
POLL_SECONDS="120"
OUTPUT_ROOT="outputs/cocoa_like_2d_mechanism"
DATASET_TWIN_INVARIANCE_PASS="auto"
CANDIDATES=()
PYTHON_BIN="${PYTHON:-python3}"

usage() {
  cat <<'USAGE'
Usage: run_operator_eval_for_run_prefix.sh --run-prefix NAME --candidates NAME... [options]

Waits for run outputs named:
  outputs/cocoa_like_2d_mechanism/${RUN_PREFIX}__${CANDIDATE}/joint/metrics.json

Then writes:
  ${RUN_PREFIX}_operator_input.csv
  ${RUN_PREFIX}_operator_eval_dim${DIM}/seidel_physical_operator_metrics.csv

Options:
  --run-prefix NAME          Required run prefix.
  --candidates NAME...       Required candidate suffixes.
  --dim N                    Evaluator dimension. Default: 256.
  --theta-convention NAME    Default: classical6d.
  --dataset-twin-invariance-pass VALUE
                              Passed to evaluator: auto, true, or false. Default: auto.
  --output-root PATH         Default: outputs/cocoa_like_2d_mechanism.
  --poll-seconds N           Completion polling interval. Default: 120.
  -h, --help                 Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-prefix)
      RUN_PREFIX="$2"
      shift 2
      ;;
    --candidates)
      shift
      CANDIDATES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CANDIDATES+=("$1")
        shift
      done
      ;;
    --dim)
      DIM="$2"
      shift 2
      ;;
    --theta-convention)
      THETA_CONVENTION="$2"
      shift 2
      ;;
    --dataset-twin-invariance-pass)
      DATASET_TWIN_INVARIANCE_PASS="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_PREFIX" ]]; then
  echo "--run-prefix is required" >&2
  exit 2
fi
if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "--candidates is required" >&2
  exit 2
fi

cd "$PROJECT_ROOT"

echo "[operator-eval] run_prefix=${RUN_PREFIX} candidates=${CANDIDATES[*]}"
while true; do
  complete=0
  for candidate in "${CANDIDATES[@]}"; do
    metrics_path="${OUTPUT_ROOT}/${RUN_PREFIX}__${candidate}/joint/metrics.json"
    if [[ -f "$metrics_path" ]]; then
      complete=$((complete + 1))
      echo "[operator-eval] ${candidate}: complete"
    else
      echo "[operator-eval] ${candidate}: waiting"
    fi
  done
  echo "[operator-eval] complete=${complete}/${#CANDIDATES[@]}"
  if [[ "$complete" -eq "${#CANDIDATES[@]}" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

input_csv="${OUTPUT_ROOT}/${RUN_PREFIX}_operator_input.csv"
eval_dir="${OUTPUT_ROOT}/${RUN_PREFIX}_operator_eval_dim${DIM}"

"${PYTHON_BIN}" - "$OUTPUT_ROOT" "$RUN_PREFIX" "$input_csv" "${CANDIDATES[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
run_prefix = sys.argv[2]
input_csv = Path(sys.argv[3])
candidates = sys.argv[4:]

rows = []
for candidate in candidates:
    metrics_path = output_root / f"{run_prefix}__{candidate}" / "joint" / "metrics.json"
    with metrics_path.open() as f:
        metrics = json.load(f)
    config = metrics["config"]
    rows.append(
        {
            "profile": candidate,
            "candidate_id": candidate,
            "pretrain_iter": config.get("pretrain_iter"),
            "pretrain_scalar": config.get("pretrain_scalar"),
            "image": metrics["image"],
            "size": metrics["size"],
            "seidel_convention": metrics["seidel_convention"],
            "fixed_seidel_indices": json.dumps(
                metrics.get("fixed_seidel_indices", config.get("fixed_seidel_indices", []))
            ),
            "seidel_gt": json.dumps(metrics["seidel_gt"]),
            "seidel_final": json.dumps(metrics["seidel_final"]),
            "ssim_recon_gain_vs_gt": metrics["ssim_recon_gain_vs_gt"],
            "nrmse_recon_gain_vs_gt": metrics["nrmse_recon_gain_vs_gt"],
            "nrmse_meas_pred_vs_meas": metrics["nrmse_meas_pred_vs_meas"],
            "recon_raw_hf_ratio": metrics["recon_raw_hf_ratio"],
            "l2_seidel_vs_gt": metrics["l2_seidel_vs_gt"],
            "nerf_depth": config["nerf_depth"],
            "nerf_width": config["nerf_width"],
            "nerf_skips": json.dumps(config["nerf_skips"]),
            "fourier_num_angles": config.get("fourier_num_angles"),
            "fourier_num_octaves": config.get("fourier_num_octaves"),
            "output_mode": config["output_mode"],
            "max_val": config["max_val"],
            "rsd_weight": config["rsd_weight"],
            "nerf_beta": config["nerf_beta"],
        }
    )

input_csv.parent.mkdir(parents=True, exist_ok=True)
with input_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(input_csv)
PY

"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_seidel_physical_operator_sweep.py" \
  "$input_csv" \
  "$eval_dir" \
  --dim "$DIM" \
  --theta-convention "$THETA_CONVENTION" \
  --dataset-twin-invariance-pass "$DATASET_TWIN_INVARIANCE_PASS" \
  --resume

echo "[operator-eval] wrote ${eval_dir}/seidel_physical_operator_metrics.csv"
