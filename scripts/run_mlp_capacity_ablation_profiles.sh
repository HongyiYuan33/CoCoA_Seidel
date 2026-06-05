#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_PREFIX="mlp_capacity_ablation_$(date +%Y%m%d)"
IMAGE="fluorescence"
SIZE="256"
MODES=("joint" "frozen")
PRETRAIN_ITER="400"
NUM_ITER="1000"
GT_PRESET="ucla"
SEIDEL_CONVENTION="classical6d"
PRIOR_PRESET="default"
OUTPUT_MODE="softplus"
MAX_VAL="40"
RSD_WEIGHT="5e-4"
NERF_BETA="1"
TV_WEIGHT="0"
PROFILES=()
PYTHON_BIN="${PYTHON:-python3}"

usage() {
  cat <<'USAGE'
Usage: run_mlp_capacity_ablation_profiles.sh [options]

Options:
  --profiles NAME...       Profiles to run. Default: all six profiles.
  --run-prefix NAME        Prefix for output run names.
  --image NAME             Input image name.
  --size N                 Reconstruction size.
  --modes MODE...          Modes passed to run_cocoa_like_2d_mechanism.py.
  --pretrain-iter N        Pretrain iterations.
  --num-iter N             Joint/frozen training iterations.
  --gt-preset NAME         Ground-truth Seidel preset.
  --seidel-convention NAME Seidel convention.
  --prior NAME             Prior preset: default or tuned-prior.
  --output-mode NAME       Override output mode.
  --max-val X              Override max_val.
  --rsd-weight X           Override RSD weight.
  --nerf-beta X            Override Softplus beta.
  --tv-weight X            Override TV weight.
  -h, --help               Show this help.

Profiles:
  baseline     depth=6 width=128 skips=2,4,6
  depth_only   depth=4 width=128 skips=2,4,6
  width_only   depth=6 width=64  skips=2,4,6
  medium       depth=4 width=64  skips=2
  low          depth=3 width=32  skips=none
  skip_only    depth=6 width=128 skips=none
  noskip_6x128 depth=6 width=128 skips=none
  noskip_4x128 depth=4 width=128 skips=none
  noskip_6x64  depth=6 width=64  skips=none
  noskip_4x64  depth=4 width=64  skips=none
  noskip_3x32  depth=3 width=32  skips=none
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profiles)
      shift
      PROFILES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        PROFILES+=("$1")
        shift
      done
      ;;
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
    --pretrain-iter)
      PRETRAIN_ITER="$2"
      shift 2
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
    --output-mode)
      OUTPUT_MODE="$2"
      shift 2
      ;;
    --max-val)
      MAX_VAL="$2"
      shift 2
      ;;
    --rsd-weight)
      RSD_WEIGHT="$2"
      shift 2
      ;;
    --nerf-beta)
      NERF_BETA="$2"
      shift 2
      ;;
    --tv-weight)
      TV_WEIGHT="$2"
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
  default)
    ;;
  tuned-prior|tuned_prior)
    OUTPUT_MODE="softplus"
    MAX_VAL="20"
    RSD_WEIGHT="1e-3"
    NERF_BETA="5"
    TV_WEIGHT="0"
    ;;
  *)
    echo "Unknown prior preset: ${PRIOR_PRESET}" >&2
    exit 2
    ;;
esac

if [[ ${#PROFILES[@]} -eq 0 ]]; then
  PROFILES=(baseline depth_only width_only medium low skip_only)
fi

profile_config() {
  case "$1" in
    baseline)
      echo "6 128 2,4,6"
      ;;
    depth_only)
      echo "4 128 2,4,6"
      ;;
    width_only)
      echo "6 64 2,4,6"
      ;;
    medium)
      echo "4 64 2"
      ;;
    low)
      echo "3 32 none"
      ;;
    skip_only)
      echo "6 128 none"
      ;;
    noskip_6x128)
      echo "6 128 none"
      ;;
    noskip_4x128)
      echo "4 128 none"
      ;;
    noskip_6x64)
      echo "6 64 none"
      ;;
    noskip_4x64)
      echo "4 64 none"
      ;;
    noskip_3x32)
      echo "3 32 none"
      ;;
    *)
      echo "Unknown profile: $1" >&2
      return 2
      ;;
  esac
}

echo "[launcher] host=$(hostname) user=$(whoami) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[launcher] run_prefix=${RUN_PREFIX} image=${IMAGE} size=${SIZE} modes=${MODES[*]}"
echo "[launcher] pretrain=${PRETRAIN_ITER} num_iter=${NUM_ITER}"
echo "[launcher] prior=${PRIOR_PRESET} output=${OUTPUT_MODE} max=${MAX_VAL} rsd=${RSD_WEIGHT} beta=${NERF_BETA} tv=${TV_WEIGHT}"

cd "$PROJECT_ROOT"

for profile in "${PROFILES[@]}"; do
  read -r depth width skips <<<"$(profile_config "$profile")"
  run_name="${RUN_PREFIX}__${profile}"
  out_dir="outputs/cocoa_like_2d_mechanism/${run_name}"
  summary_path="${out_dir}/summary.json"

  if [[ -f "$summary_path" ]]; then
    echo "[skip] ${profile}: ${summary_path} exists"
    continue
  fi

  echo "[run] ${profile}: depth=${depth} width=${width} skips=${skips}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_cocoa_like_2d_mechanism.py" \
    --image "$IMAGE" \
    --size "$SIZE" \
    --modes "${MODES[@]}" \
    --pretrain-iter "$PRETRAIN_ITER" \
    --num-iter "$NUM_ITER" \
    --gt-preset "$GT_PRESET" \
    --seidel-convention "$SEIDEL_CONVENTION" \
    --output-mode "$OUTPUT_MODE" \
    --max-val "$MAX_VAL" \
    --rsd-weight "$RSD_WEIGHT" \
    --nerf-beta "$NERF_BETA" \
    --tv-weight "$TV_WEIGHT" \
    --nerf-depth "$depth" \
    --nerf-width "$width" \
    --nerf-skips "$skips" \
    --run-name "$run_name"
done

echo "[launcher] complete"
