#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_PREFIX="pretrain_ablation_tunedprior_noskip6x128_$(date +%Y%m%d)"
IMAGE="fluorescence"
SIZE="256"
MODES=("joint" "frozen")
PRETRAIN_ITERS=("400" "200" "100")
PRETRAIN_SCALARS=("5" "3" "1")
NUM_ITER="1000"
GT_PRESET="ucla"
SEIDEL_CONVENTION="classical6d"
PRIOR_PRESET="tuned-prior"
OUTPUT_MODE="softplus"
MAX_VAL="20"
RSD_WEIGHT="1e-3"
NERF_BETA="5"
TV_WEIGHT="0"
NERF_DEPTH="6"
NERF_WIDTH="128"
NERF_SKIPS="none"
NUM_SHARDS="1"
SHARD_INDEX="0"
PYTHON_BIN="${PYTHON:-python3}"

usage() {
  cat <<'USAGE'
Usage: run_pretrain_ablation_grid.sh [options]

Runs a pretrain ablation grid while keeping object architecture fixed by default:
  MLP depth=6 width=128 skips=none, tuned-prior object settings.

Options:
  --run-prefix NAME          Prefix for output run names.
  --image NAME               Input image name.
  --size N                   Reconstruction size.
  --modes MODE...            Modes passed to run_cocoa_like_2d_mechanism.py.
  --pretrain-iters N...      Pretrain iteration values.
  --pretrain-scalars X...    Pretrain scalar values.
  --num-iter N               Joint/frozen training iterations.
  --gt-preset NAME           Ground-truth Seidel preset.
  --seidel-convention NAME   Seidel convention.
  --prior NAME               Prior preset: tuned-prior or default.
  --nerf-depth N             MLP depth.
  --nerf-width N             MLP width.
  --nerf-skips VALUE         MLP skips, comma-separated or none.
  --num-shards N             Number of case shards.
  --shard-index N            This shard index, 0-based.
  -h, --help                 Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-prefix)
      RUN_PREFIX="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --size)
      SIZE="$2"
      shift 2
      ;;
    --modes)
      shift
      MODES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        MODES+=("$1")
        shift
      done
      ;;
    --pretrain-iters)
      shift
      PRETRAIN_ITERS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        PRETRAIN_ITERS+=("$1")
        shift
      done
      ;;
    --pretrain-scalars)
      shift
      PRETRAIN_SCALARS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        PRETRAIN_SCALARS+=("$1")
        shift
      done
      ;;
    --num-iter)
      NUM_ITER="$2"
      shift 2
      ;;
    --gt-preset)
      GT_PRESET="$2"
      shift 2
      ;;
    --seidel-convention)
      SEIDEL_CONVENTION="$2"
      shift 2
      ;;
    --prior)
      PRIOR_PRESET="$2"
      shift 2
      ;;
    --nerf-depth)
      NERF_DEPTH="$2"
      shift 2
      ;;
    --nerf-width)
      NERF_WIDTH="$2"
      shift 2
      ;;
    --nerf-skips)
      NERF_SKIPS="$2"
      shift 2
      ;;
    --num-shards)
      NUM_SHARDS="$2"
      shift 2
      ;;
    --shard-index)
      SHARD_INDEX="$2"
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

case "$PRIOR_PRESET" in
  tuned-prior|tuned_prior)
    OUTPUT_MODE="softplus"
    MAX_VAL="20"
    RSD_WEIGHT="1e-3"
    NERF_BETA="5"
    TV_WEIGHT="0"
    ;;
  default)
    OUTPUT_MODE="softplus"
    MAX_VAL="40"
    RSD_WEIGHT="5e-4"
    NERF_BETA="1"
    TV_WEIGHT="0"
    ;;
  *)
    echo "Unknown prior preset: ${PRIOR_PRESET}" >&2
    exit 2
    ;;
esac

if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "--num-shards must be >= 1" >&2
  exit 2
fi
if [[ "$SHARD_INDEX" -lt 0 || "$SHARD_INDEX" -ge "$NUM_SHARDS" ]]; then
  echo "--shard-index must be in [0, --num-shards)" >&2
  exit 2
fi

tag_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//+/p}"
  echo "$value"
}

echo "[launcher] host=$(hostname) user=$(whoami) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[launcher] run_prefix=${RUN_PREFIX} image=${IMAGE} size=${SIZE} modes=${MODES[*]}"
echo "[launcher] prior=${PRIOR_PRESET} output=${OUTPUT_MODE} max=${MAX_VAL} rsd=${RSD_WEIGHT} beta=${NERF_BETA} tv=${TV_WEIGHT}"
echo "[launcher] mlp=${NERF_DEPTH}x${NERF_WIDTH} skips=${NERF_SKIPS} num_iter=${NUM_ITER}"
echo "[launcher] shard=${SHARD_INDEX}/${NUM_SHARDS}"

cd "$PROJECT_ROOT"

case_idx=0
for pretrain_iter in "${PRETRAIN_ITERS[@]}"; do
  for pretrain_scalar in "${PRETRAIN_SCALARS[@]}"; do
    if (( case_idx % NUM_SHARDS != SHARD_INDEX )); then
      case_idx=$((case_idx + 1))
      continue
    fi

    scalar_tag="$(tag_value "$pretrain_scalar")"
    run_name="${RUN_PREFIX}__pre${pretrain_iter}__scalar${scalar_tag}"
    out_dir="outputs/cocoa_like_2d_mechanism/${run_name}"
    summary_path="${out_dir}/summary.json"

    if [[ -f "$summary_path" ]]; then
      echo "[skip] pre=${pretrain_iter} scalar=${pretrain_scalar}: ${summary_path} exists"
      case_idx=$((case_idx + 1))
      continue
    fi

    echo "[run] pre=${pretrain_iter} scalar=${pretrain_scalar}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_cocoa_like_2d_mechanism.py" \
      --image "$IMAGE" \
      --size "$SIZE" \
      --modes "${MODES[@]}" \
      --pretrain-iter "$pretrain_iter" \
      --pretrain-scalar "$pretrain_scalar" \
      --num-iter "$NUM_ITER" \
      --gt-preset "$GT_PRESET" \
      --seidel-convention "$SEIDEL_CONVENTION" \
      --output-mode "$OUTPUT_MODE" \
      --max-val "$MAX_VAL" \
      --rsd-weight "$RSD_WEIGHT" \
      --nerf-beta "$NERF_BETA" \
      --tv-weight "$TV_WEIGHT" \
      --nerf-depth "$NERF_DEPTH" \
      --nerf-width "$NERF_WIDTH" \
      --nerf-skips "$NERF_SKIPS" \
      --run-name "$run_name"

    case_idx=$((case_idx + 1))
  done
done

echo "[launcher] complete"
