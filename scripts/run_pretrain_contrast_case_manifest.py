"""Run pretrain-contrast stage1 cases from a case manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREFIX = "pretrain_contrast_top10plusbase4d_size256_three_images_rms040_pre400_joint1000_20260609"
DEFAULT_LOG_DIR = (
    PROJECT_ROOT
    / "outputs/cocoa_like_2d_mechanism"
    / f"{DEFAULT_PREFIX}_logs"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def method_id(row: dict[str, Any]) -> str:
    return str(row.get("method") or row.get("pretrain_method") or "")


def load_settings(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text())
    return {method_id(row): dict(row) for row in rows}


def setting_float(setting: dict[str, Any], key: str, default: float) -> str:
    return str(float(setting.get(key, default)))


def case_items(rows: list[dict[str, str]], *, shard_index: int, num_shards: int) -> list[dict[str, str]]:
    out = []
    for offset, row in enumerate(rows):
        case_index = int(row.get("case_index") or offset)
        if case_index % num_shards == shard_index:
            out.append(row)
    return out


def build_cmd(
    *,
    python: str,
    prefix: str,
    row: dict[str, str],
    setting: dict[str, Any],
    seidel_convention: str,
    candidate_mode: str,
    gt_locked_source_csv: Path | None,
    gt_locked_w311_scale: float,
    gt_locked_wd_scale: float,
    gt_locked_atol: float,
    stage1_size: int,
    pretrain_iter: int,
    joint_iter: int,
    measurement_direct: bool,
) -> list[str]:
    method = row["pretrain_method"]
    effective_candidate_mode = "measurement_direct" if measurement_direct else candidate_mode
    cmd = [
        python,
        "scripts/run_cocoa_like_seidel_accuracy_sweep.py",
        "--stage",
        "stage1",
        "--run-name",
        f"{prefix}__{method}",
        "--images",
        row["image"],
        "--seidel-convention",
        seidel_convention,
        "--candidate-mode",
        effective_candidate_mode,
        "--stage1-size",
        str(stage1_size),
        "--stage1-pretrain-iter",
        str(pretrain_iter),
        "--stage1-num-iter",
        str(joint_iter),
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
        "--pretrain-scalar",
        setting_float(setting, "pretrain_scalar", 1.0),
        "--pretrain-target-transform",
        str(setting.get("target_transform", "none")),
        "--pretrain-contrast-alpha",
        setting_float(setting, "contrast_alpha", 1.0),
        "--pretrain-percentile-lo",
        setting_float(setting, "percentile_lo", 1.0),
        "--pretrain-percentile-hi",
        setting_float(setting, "percentile_hi", 99.0),
        "--pretrain-gamma",
        setting_float(setting, "gamma", 1.0),
        "--pretrain-rsd-weight",
        setting_float(setting, "pretrain_rsd_weight", 0.0),
        "--pretrain-edge-weight",
        setting_float(setting, "pretrain_edge_weight", 0.0),
        "--pretrain-edge-mode",
        str(setting.get("pretrain_edge_mode", "sobel")),
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
        "--skip-report",
        "--skip-config-write",
    ]
    if not measurement_direct:
        cmd.extend(
            [
                "--directions",
                row["direction"],
                "--strengths",
                f"{float(row['target_rms']):.12g}",
            ]
        )
    if gt_locked_source_csv is not None:
        cmd.extend(["--gt-locked-source-csv", str(gt_locked_source_csv)])
        cmd.extend(
            [
                "--gt-locked-w311-scale",
                str(gt_locked_w311_scale),
                "--gt-locked-wd-scale",
                str(gt_locked_wd_scale),
                "--gt-locked-atol",
                str(gt_locked_atol),
            ]
        )
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, default=DEFAULT_LOG_DIR / "case_manifest.csv")
    parser.add_argument("--settings-manifest", type=Path, default=DEFAULT_LOG_DIR / "settings_manifest.json")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seidel-convention", default="classical4d")
    parser.add_argument("--candidate-mode", default="direction")
    parser.add_argument(
        "--measurement-direct",
        action="store_true",
        help="Run one measurement_direct candidate per image and do not pass RMS/direction candidates.",
    )
    parser.add_argument("--gt-locked-source-csv", type=Path, default=None)
    parser.add_argument("--gt-locked-w311-scale", type=float, default=-0.5)
    parser.add_argument("--gt-locked-wd-scale", type=float, default=0.5)
    parser.add_argument("--gt-locked-atol", type=float, default=1e-7)
    parser.add_argument("--stage1-size", type=int, default=256)
    parser.add_argument("--stage1-pretrain-iter", type=int, default=400)
    parser.add_argument("--stage1-num-iter", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    settings = load_settings(args.settings_manifest)
    rows = case_items(read_csv(args.case_manifest), shard_index=args.shard_index, num_shards=args.num_shards)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"[worker] prefix={args.prefix} shard={args.shard_index}/{args.num_shards} cases={len(rows)}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    for row in rows:
        method = row["pretrain_method"]
        if method not in settings:
            raise KeyError(f"Missing setting for method {method!r}")
        cmd = build_cmd(
            python=args.python,
            prefix=args.prefix,
            row=row,
            setting=settings[method],
            seidel_convention=args.seidel_convention,
            candidate_mode=args.candidate_mode,
            gt_locked_source_csv=args.gt_locked_source_csv,
            gt_locked_w311_scale=args.gt_locked_w311_scale,
            gt_locked_wd_scale=args.gt_locked_wd_scale,
            gt_locked_atol=args.gt_locked_atol,
            stage1_size=args.stage1_size,
            pretrain_iter=args.stage1_pretrain_iter,
            joint_iter=args.stage1_num_iter,
            measurement_direct=args.measurement_direct,
        )
        print(
            f"[case] index={row.get('case_index')} image={row['image']} method={method} "
            f"rms={row['target_rms']}",
            flush=True,
        )
        if args.dry_run:
            print(" ".join(cmd), flush=True)
            continue
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    print("[worker-done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
