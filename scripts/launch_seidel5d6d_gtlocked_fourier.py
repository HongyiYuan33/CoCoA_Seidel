"""Launch worker/report phases for the Fourier GPU2 GT-locked 5D/6D sweep."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cocoa_like_2d_mechanism"
PREFIX = "seidel5d6d_gtlocked_tunedadam256_four_images_pre400_joint1000_20260609"
BASELINE_RUN = "capacity4d_dirrms_tunedprior_size256_four_images_20260607__baseline"
SOURCE_CSV = OUTPUT_ROOT / BASELINE_RUN / "stage1_metrics.csv"
PYTHON = Path("/home/hongyi/4tb_nvme/miniconda3/envs/hybrid_ring/bin/python3.10")
IMAGES = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]
CONVENTIONS = ["classical5d", "classical6d"]
EXPECTED_HOST = "waller-fourier"
EXPECTED_USER = "hongyi"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def check_host() -> None:
    host = socket.gethostname()
    user = os.environ.get("USER") or subprocess.check_output(["whoami"], text=True).strip()
    if host != EXPECTED_HOST or user != EXPECTED_USER:
        raise RuntimeError(f"Expected {EXPECTED_USER}@{EXPECTED_HOST}, got {user}@{host}")
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(SOURCE_CSV)


def gpu_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["PYTHONPATH"] = "."
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def gpu2_busy() -> bool:
    query = [
        "nvidia-smi",
        "-i",
        "2",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(query, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return True
    return bool(proc.stdout.strip())


def wait_for_gpu2_idle() -> None:
    while gpu2_busy():
        print("[wait] Fourier GPU2 has active compute processes; sleeping 300s", flush=True)
        time.sleep(300)


def sweep_cmd(convention: str, *, stage: str, full_train: bool) -> list[str]:
    run_name = f"{PREFIX}__{convention}"
    cmd = [
        str(PYTHON),
        "scripts/run_cocoa_like_seidel_accuracy_sweep.py",
        "--run-name",
        run_name,
        "--stage",
        stage,
        "--images",
        *IMAGES,
        "--candidate-mode",
        "gt_locked_front4",
        "--gt-locked-source-csv",
        str(SOURCE_CSV),
        "--directions",
        "cocoa_signed",
        "signed_balanced",
        "--strengths",
        "0.06",
        "0.20",
        "0.40",
        "--seidel-convention",
        convention,
        "--stage1-size",
        "256",
        "--stage1-pretrain-iter",
        "400" if full_train else "2",
        "--stage1-num-iter",
        "1000" if full_train else "2",
        "--pretrain-scalar",
        "5",
        "--lr-obj",
        "0.005",
        "--lr-seidel",
        "0.01",
        "--seidel-optimizer",
        "adam",
        "--rsd-weight",
        "1e-3",
        "--tv-weight",
        "0",
        "--max-val",
        "20",
        "--nerf-beta",
        "5",
        "--output-mode",
        "softplus",
        "--scheduler",
        "cosine",
        "--eta-min-ratio",
        "0.04",
        "--nerf-depth",
        "6",
        "--nerf-width",
        "128",
        "--nerf-skips",
        "2,4,6",
        "--fourier-num-angles",
        "60",
        "--fourier-num-octaves",
        "7",
    ]
    if stage == "stage1":
        cmd.append("--case-subprocess")
    return cmd


def count_completed(run_name: str) -> int:
    stage_root = OUTPUT_ROOT / run_name / "stage1"
    total = 0
    for path in stage_root.glob("*/joint/metrics.json"):
        try:
            metrics = json.loads(path.read_text())
        except Exception:
            continue
        if metrics.get("sweep_case_complete") is True:
            total += 1
    return total


def worker() -> None:
    check_host()
    wait_for_gpu2_idle()
    env = gpu_env()
    for convention in CONVENTIONS:
        print(f"[worker] starting {convention}", flush=True)
        run(sweep_cmd(convention, stage="stage1", full_train=True), env=env)
        print(f"[worker] finished {convention}", flush=True)


def wait_for_stage1() -> None:
    while True:
        counts = {conv: count_completed(f"{PREFIX}__{conv}") for conv in CONVENTIONS}
        print(f"[report-wait] completed={counts}", flush=True)
        if all(counts[conv] >= 24 for conv in CONVENTIONS):
            return
        time.sleep(300)


def evaluator_cmd(convention: str) -> list[str]:
    run_name = f"{PREFIX}__{convention}"
    return [
        str(PYTHON),
        "scripts/evaluate_seidel_physical_operator_sweep.py",
        str(OUTPUT_ROOT / run_name / "stage1_metrics.csv"),
        str(OUTPUT_ROOT / run_name / "stage1_operator_eval_dim256"),
        "--dim",
        "256",
        "--theta-convention",
        convention,
        "--dataset-twin-invariance-pass",
        "true",
        "--resume",
    ]


def report() -> None:
    check_host()
    env = gpu_env()
    wait_for_stage1()
    for convention in CONVENTIONS:
        run(sweep_cmd(convention, stage="report", full_train=True), env=env)
        run(evaluator_cmd(convention), env=env)
    run(
        [
            str(PYTHON),
            "scripts/build_seidel5d6d_gtlocked_rcp_stats.py",
            "--output-root",
            str(OUTPUT_ROOT),
            "--prefix",
            PREFIX,
            "--baseline-run",
            BASELINE_RUN,
        ],
        env=env,
    )
    print("[report] complete", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["worker", "report"])
    args = parser.parse_args()
    if args.phase == "worker":
        worker()
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
