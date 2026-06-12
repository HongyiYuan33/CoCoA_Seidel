"""Summarize, evaluate, and render RCPs for a pretrain-contrast case manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs/cocoa_like_2d_mechanism"
DEFAULT_PREFIX = "pretrain_contrast_top10plusbase4d_size256_three_images_rms040_pre400_joint1000_20260609"
DEFAULT_LOG_DIR = OUTPUT_ROOT / f"{DEFAULT_PREFIX}_logs"
DEFAULT_RCP_DIR = OUTPUT_ROOT / f"{DEFAULT_PREFIX}_rcp_stats"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, separators=(",", ":"))
                        if isinstance(value, (list, dict, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def method_id(row: dict[str, Any]) -> str:
    return str(row.get("method") or row.get("pretrain_method") or "")


def load_settings(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    out = []
    for row in rows:
        item = dict(row)
        item["method"] = method_id(item)
        out.append(item)
    return out


def case_metrics_path(output_root: Path, prefix: str, row: dict[str, str]) -> Path:
    return (
        output_root
        / f"{prefix}__{row['pretrain_method']}"
        / "stage1"
        / f"{row['image']}__{row['candidate_id']}"
        / "joint"
        / "metrics.json"
    )


def completed_rows(output_root: Path, prefix: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    done = []
    for row in rows:
        path = case_metrics_path(output_root, prefix, row)
        if not path.is_file():
            continue
        try:
            metrics = json.loads(path.read_text())
        except Exception:
            continue
        if metrics.get("sweep_case_complete") is True:
            done.append(row)
    return done


def wait_for_completion(
    *,
    output_root: Path,
    prefix: str,
    rows: list[dict[str, str]],
    poll_seconds: float,
) -> None:
    while True:
        done = completed_rows(output_root, prefix, rows)
        print(f"[wait] completed={len(done)}/{len(rows)}", flush=True)
        if len(done) == len(rows):
            return
        time.sleep(poll_seconds)


def write_stage1_metrics(output_root: Path, prefix: str, rows: list[dict[str, str]]) -> dict[str, Path]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        metrics_path = case_metrics_path(output_root, prefix, row)
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("sweep_case_complete") is not True:
            raise RuntimeError(f"Incomplete case: {metrics_path}")
        by_method.setdefault(row["pretrain_method"], []).append(metrics)
    paths = {}
    for method, metrics_rows in by_method.items():
        run_dir = output_root / f"{prefix}__{method}"
        out_csv = run_dir / "stage1_metrics.csv"
        metrics_rows.sort(key=lambda item: (str(item.get("image", "")), str(item.get("candidate_id", ""))))
        write_csv(metrics_rows, out_csv)
        paths[method] = out_csv
        print(f"[stage1-csv] {method} rows={len(metrics_rows)} path={out_csv}", flush=True)
    return paths


def write_selected_settings(
    settings: list[dict[str, Any]],
    methods: set[str],
    path: Path,
) -> Path:
    selected = [row for row in settings if method_id(row) in methods]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selected, indent=2) + "\n")
    return path


def run_evaluators(
    python: str,
    stage1_paths: dict[str, Path],
    *,
    theta_convention: str,
) -> None:
    for method, stage1_csv in sorted(stage1_paths.items()):
        out_dir = stage1_csv.parent / "stage1_operator_eval_dim256"
        cmd = [
            python,
            "scripts/evaluate_seidel_physical_operator_sweep.py",
            str(stage1_csv),
            str(out_dir),
            "--dim",
            "256",
            "--theta-convention",
            theta_convention,
            "--dataset-twin-invariance-pass",
            "true",
            "--resume",
        ]
        print(f"[eval] {method}", flush=True)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def run_rcp_builder(
    *,
    python: str,
    prefix: str,
    settings_manifest: Path,
    output_dir: Path,
) -> None:
    cmd = [
        python,
        "scripts/build_pretrain_contrast_rcp_stats.py",
        "--prefix",
        prefix,
        "--settings-manifest",
        str(settings_manifest),
        "--output-dir",
        str(output_dir),
    ]
    print(f"[rcp] {output_dir}", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, default=DEFAULT_LOG_DIR / "case_manifest.csv")
    parser.add_argument("--settings-manifest", type=Path, default=DEFAULT_LOG_DIR / "settings_manifest.json")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RCP_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--theta-convention", default="classical4d")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_csv(args.case_manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.wait:
        wait_for_completion(
            output_root=args.output_root,
            prefix=args.prefix,
            rows=rows,
            poll_seconds=args.poll_seconds,
        )
    done = completed_rows(args.output_root, args.prefix, rows)
    if len(done) != len(rows):
        raise RuntimeError(f"Only {len(done)}/{len(rows)} cases complete")

    stage1_paths = write_stage1_metrics(args.output_root, args.prefix, rows)
    run_evaluators(args.python, stage1_paths, theta_convention=args.theta_convention)
    selected_settings = write_selected_settings(
        load_settings(args.settings_manifest),
        set(stage1_paths),
        args.output_dir / "settings_manifest.selected_for_report.json",
    )
    run_rcp_builder(
        python=args.python,
        prefix=args.prefix,
        settings_manifest=selected_settings,
        output_dir=args.output_dir,
    )
    print(f"[report-done] cases={len(rows)} methods={len(stage1_paths)} out={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
