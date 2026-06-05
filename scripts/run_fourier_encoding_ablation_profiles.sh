#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_PREFIX="fourier_encoding_ablation_$(date +%Y%m%d)"
IMAGE="fluorescence"
SIZE="256"
MODES=("joint" "frozen")
PRETRAIN_ITER="400"
PRETRAIN_SCALAR="5"
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
PROFILES=()
PYTHON_BIN="${PYTHON:-python3}"

usage() {
  cat <<'USAGE'
Usage: run_fourier_encoding_ablation_profiles.sh [options]

Runs a one-at-a-time Fourier encoding ablation while keeping the object
architecture fixed by default: MLP depth=6 width=128 skips=none, tuned-prior,
pretrain=400, pretrain-scalar=5.

Options:
  --profiles NAME...          Profiles to run. Default: all profiles below.
  --run-prefix NAME           Prefix for output run names.
  --image NAME                Input image name.
  --size N                    Reconstruction size.
  --modes MODE...             Modes passed to run_cocoa_like_2d_mechanism.py.
  --pretrain-iter N           Pretrain iterations.
  --pretrain-scalar X         Pretrain measurement scalar.
  --num-iter N                Joint/frozen training iterations.
  --gt-preset NAME            Ground-truth Seidel preset.
  --seidel-convention NAME    Seidel convention.
  --prior NAME                Prior preset: tuned-prior or default.
  --nerf-depth N              MLP depth.
  --nerf-width N              MLP width.
  --nerf-skips VALUE          MLP skips, comma-separated or none.
  -h, --help                  Show this help.

Profiles:
  oct7_ang60  num_octaves=7 num_angles=60 baseline
  oct5_ang60  num_octaves=5 num_angles=60
  oct4_ang60  num_octaves=4 num_angles=60
  oct3_ang60  num_octaves=3 num_angles=60
  oct7_ang30  num_octaves=7 num_angles=30
  oct7_ang16  num_octaves=7 num_angles=16
  oct8_ang60  num_octaves=8 num_angles=60
  oct9_ang60  num_octaves=9 num_angles=60
  oct7_ang90  num_octaves=7 num_angles=90
  oct8_ang30  num_octaves=8 num_angles=30
  oct8_ang90  num_octaves=8 num_angles=90
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
    --pretrain-scalar)
      PRETRAIN_SCALAR="$2"
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

if [[ ${#PROFILES[@]} -eq 0 ]]; then
  PROFILES=(oct7_ang60 oct5_ang60 oct4_ang60 oct3_ang60 oct7_ang30 oct7_ang16)
fi

profile_config() {
  case "$1" in
    oct7_ang60)
      echo "7 60"
      ;;
    oct5_ang60)
      echo "5 60"
      ;;
    oct4_ang60)
      echo "4 60"
      ;;
    oct3_ang60)
      echo "3 60"
      ;;
    oct7_ang30)
      echo "7 30"
      ;;
    oct7_ang16)
      echo "7 16"
      ;;
    oct8_ang60)
      echo "8 60"
      ;;
    oct9_ang60)
      echo "9 60"
      ;;
    oct7_ang90)
      echo "7 90"
      ;;
    oct8_ang30)
      echo "8 30"
      ;;
    oct8_ang90)
      echo "8 90"
      ;;
    *)
      echo "Unknown profile: $1" >&2
      return 2
      ;;
  esac
}

echo "[launcher] host=$(hostname) user=$(whoami) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[launcher] run_prefix=${RUN_PREFIX} image=${IMAGE} size=${SIZE} modes=${MODES[*]}"
echo "[launcher] pretrain=${PRETRAIN_ITER} scalar=${PRETRAIN_SCALAR} num_iter=${NUM_ITER}"
echo "[launcher] prior=${PRIOR_PRESET} output=${OUTPUT_MODE} max=${MAX_VAL} rsd=${RSD_WEIGHT} beta=${NERF_BETA} tv=${TV_WEIGHT}"
echo "[launcher] mlp=${NERF_DEPTH}x${NERF_WIDTH} skips=${NERF_SKIPS} profiles=${PROFILES[*]}"

cd "$PROJECT_ROOT"

for profile in "${PROFILES[@]}"; do
  read -r octaves angles <<<"$(profile_config "$profile")"
  run_name="${RUN_PREFIX}__${profile}"
  out_dir="outputs/cocoa_like_2d_mechanism/${run_name}"
  summary_path="${out_dir}/summary.json"

  if [[ -f "$summary_path" ]]; then
    echo "[skip] ${profile}: ${summary_path} exists"
    continue
  fi

  echo "[run] ${profile}: num_octaves=${octaves} num_angles=${angles}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_cocoa_like_2d_mechanism.py" \
    --image "$IMAGE" \
    --size "$SIZE" \
    --modes "${MODES[@]}" \
    --pretrain-iter "$PRETRAIN_ITER" \
    --pretrain-scalar "$PRETRAIN_SCALAR" \
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
    --fourier-num-octaves "$octaves" \
    --fourier-num-angles "$angles" \
    --run-name "$run_name"
done

echo "[launcher] complete"
