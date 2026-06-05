#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAMP="${STAMP:-20260604}"
PYTHON_BIN="${PYTHON:-/hdd10tb/hongyi_waller/miniconda3/envs/hybrid_ring/bin/python3.10}"
SIZE="${SIZE:-512}"
POLL_SECONDS="${POLL_SECONDS:-300}"
GPU_FREE_MIB="${GPU_FREE_MIB:-500}"
LOGDIR="${LOGDIR:-${PROJECT_ROOT}/outputs/cocoa_like_2d_mechanism/size512_best_pretrain_fourier_two_images_${STAMP}_logs}"
IMAGES=(${IMAGES:-dendrites dendrites_dense})

if [[ "${#IMAGES[@]}" -ne 2 ]]; then
  echo "IMAGES must contain exactly two image names; got: ${IMAGES[*]}" >&2
  exit 2
fi

mkdir -p "$LOGDIR"
cd "$PROJECT_ROOT"
exec >> "${LOGDIR}/watcher.log" 2>&1

echo "[start] $(date)"
echo "[identity] host=$(hostname) user=$(whoami) root=${PROJECT_ROOT}"
echo "[images] ${IMAGES[*]} size=${SIZE}"

dense_required=(
  "outputs/cocoa_like_2d_mechanism/pretrain_union_tunedprior_noskip6x128_size256_dendrites_dense_${STAMP}_operator_eval_dim256/seidel_physical_operator_metrics.csv"
  "outputs/cocoa_like_2d_mechanism/fourier_union_tunedprior_noskip6x128_pre400scalar5_size256_dendrites_dense_${STAMP}_operator_eval_dim256/seidel_physical_operator_metrics.csv"
  "outputs/cocoa_like_2d_mechanism/capacity_union_tunedprior_size256_dendrites_dense_${STAMP}_operator_eval_dim256/seidel_physical_operator_metrics.csv"
)

while true; do
  complete=0
  for path in "${dense_required[@]}"; do
    if [[ -f "$path" ]]; then
      complete=$((complete + 1))
    fi
  done
  echo "[wait-dense-eval] $(date) complete=${complete}/${#dense_required[@]}"
  if [[ "$complete" -eq "${#dense_required[@]}" ]]; then
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

select_best_config() {
  local family="$1"
  local image="$2"
  "$PYTHON_BIN" - "$family" "$image" "$STAMP" <<'PY'
import csv
import math
import re
import sys
from pathlib import Path

family, image, stamp = sys.argv[1:4]
root = Path("outputs/cocoa_like_2d_mechanism")

if family == "pretrain":
    patterns = [
        f"pretrain_ablation_tunedprior_noskip6x128_size256_{image}_{stamp}_operator_eval_dim256/seidel_physical_operator_metrics.csv",
        f"pretrain_pairs_ext_tunedprior_noskip6x128_size256_{image}_{stamp}_operator_eval_dim256/seidel_physical_operator_metrics.csv",
        f"pretrain_fine_pairs_tunedprior_noskip6x128_size256_{image}_{stamp}_operator_eval_dim256/seidel_physical_operator_metrics.csv",
        f"pretrain_union_tunedprior_noskip6x128_size256_{image}_{stamp}_operator_eval_dim256/seidel_physical_operator_metrics.csv",
    ]
elif family == "fourier":
    patterns = [
        f"fourier_encoding_ablation_tunedprior_noskip6x128_pre400scalar5_size256_{image}_{stamp}_operator_eval_dim256/seidel_physical_operator_metrics.csv",
        f"fourier_encoding_larger_tunedprior_noskip6x128_pre400scalar5_size256_{image}_{stamp}_operator_eval_dim256/seidel_physical_operator_metrics.csv",
        f"fourier_union_tunedprior_noskip6x128_pre400scalar5_size256_{image}_{stamp}_operator_eval_dim256/seidel_physical_operator_metrics.csv",
    ]
else:
    raise SystemExit(f"unknown family: {family}")

rows = []
for rel in patterns:
    path = root / rel
    if not path.exists():
        continue
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["_source_csv"] = str(path)
            try:
                row["_op"] = float(row.get("operator_error_calibrated", "nan"))
            except Exception:
                row["_op"] = math.nan
            if not math.isnan(row["_op"]):
                rows.append(row)

if not rows:
    raise SystemExit(f"no rows for {family} {image}")

best = min(rows, key=lambda r: r["_op"])
profile = best.get("profile") or best.get("candidate_id") or ""

def clean_number(value):
    x = float(value)
    if abs(x - round(x)) < 1e-8:
        return str(int(round(x)))
    return ("%g" % x)

def tag_number(value):
    return clean_number(value).replace(".", "p").replace("-", "m").replace("+", "p")

if family == "pretrain":
    pre = best.get("pretrain_iter") or ""
    scalar = best.get("pretrain_scalar") or ""
    if not pre or not scalar:
        m = re.search(r"pre(\d+).*scalar([0-9p.]+)", profile)
        if m:
            pre = m.group(1)
            scalar = m.group(2).replace("p", ".")
    pre = clean_number(pre)
    scalar = clean_number(scalar)
    candidate = f"pre{pre}__scalar{tag_number(scalar)}"
    print("\t".join([pre, scalar, candidate, profile, f"{best['_op']:.9f}", best.get("ssim_recon_gain_vs_gt", ""), best["_source_csv"]]))
else:
    m = re.search(r"oct(\d+)_ang(\d+)", profile)
    if not m:
        raise SystemExit(f"cannot parse Fourier profile: {profile}")
    octaves, angles = m.group(1), m.group(2)
    candidate = f"oct{octaves}_ang{angles}"
    print("\t".join([octaves, angles, candidate, profile, f"{best['_op']:.9f}", best.get("ssim_recon_gain_vs_gt", ""), best["_source_csv"]]))
PY
}

run_image() {
  local physical_gpu="$1"
  local image="$2"
  local log_path="$3"
  (
    set -euo pipefail
    cd "$PROJECT_ROOT"
    exec >> "$log_path" 2>&1

    echo "[image-start] $(date) image=${image} physical_gpu=${physical_gpu}"
    export PYTHON="$PYTHON_BIN"
    export PYTHONPATH=.
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$physical_gpu"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    nvidia-smi --query-gpu=index,pci.bus_id,name,memory.used,memory.total,utilization.gpu --format=csv

    IFS=$'\t' read -r pre_iter pre_scalar pre_candidate pre_profile pre_op pre_ssim pre_source <<<"$(select_best_config pretrain "$image")"
    IFS=$'\t' read -r octaves angles fourier_candidate fourier_profile fourier_op fourier_ssim fourier_source <<<"$(select_best_config fourier "$image")"

    echo "[selected-pretrain] image=${image} iter=${pre_iter} scalar=${pre_scalar} candidate=${pre_candidate} source_profile=${pre_profile} op=${pre_op} ssim=${pre_ssim} source=${pre_source}"
    echo "[selected-fourier] image=${image} octaves=${octaves} angles=${angles} candidate=${fourier_candidate} source_profile=${fourier_profile} op=${fourier_op} ssim=${fourier_ssim} source=${fourier_source}"
    {
      echo "family,image,param1,param2,candidate,source_profile,source_op,source_ssim,source_csv"
      echo "pretrain,${image},${pre_iter},${pre_scalar},${pre_candidate},${pre_profile},${pre_op},${pre_ssim},${pre_source}"
      echo "fourier,${image},${octaves},${angles},${fourier_candidate},${fourier_profile},${fourier_op},${fourier_ssim},${fourier_source}"
    } > "${LOGDIR}/selected_best_${image}.csv"

    local pre_prefix="size512_best_pretrain_tunedprior_noskip6x128_${image}_${STAMP}"
    bash scripts/run_pretrain_ablation_pairs.sh \
      --run-prefix "$pre_prefix" \
      --image "$image" \
      --size "$SIZE" \
      --modes joint frozen \
      --pairs "${pre_iter}:${pre_scalar}" \
      --num-iter 1000 \
      --gt-preset ucla \
      --seidel-convention classical6d \
      --prior tuned-prior \
      --nerf-depth 6 \
      --nerf-width 128 \
      --nerf-skips none
    bash scripts/run_operator_eval_for_run_prefix.sh \
      --run-prefix "$pre_prefix" \
      --candidates "$pre_candidate" \
      --dim "$SIZE" \
      --theta-convention classical6d \
      --poll-seconds 180

    local fourier_prefix="size512_best_fourier_tunedprior_noskip6x128_pre400scalar5_${image}_${STAMP}"
    bash scripts/run_fourier_encoding_ablation_profiles.sh \
      --run-prefix "$fourier_prefix" \
      --image "$image" \
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
      --profiles "$fourier_candidate"
    bash scripts/run_operator_eval_for_run_prefix.sh \
      --run-prefix "$fourier_prefix" \
      --candidates "$fourier_candidate" \
      --dim "$SIZE" \
      --theta-convention classical6d \
      --poll-seconds 180

    echo "[image-done] $(date) image=${image}"
  )
}

echo "[launch-size512] $(date)"
run_image 0 "${IMAGES[0]}" "${LOGDIR}/gpu0_${IMAGES[0]}.log" &
pid0=$!
run_image 1 "${IMAGES[1]}" "${LOGDIR}/gpu1_${IMAGES[1]}.log" &
pid1=$!

set +e
wait "$pid0"
st0=$?
wait "$pid1"
st1=$?
set -e

if (( st0 != 0 || st1 != 0 )); then
  echo "[error] size512 image job failed: gpu0=${st0} gpu1=${st1}"
  exit 1
fi

echo "[all-complete] $(date)"
date > "${LOGDIR}/all_complete.marker"
