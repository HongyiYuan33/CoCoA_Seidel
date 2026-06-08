#!/usr/bin/env python3
"""Run grouped gauge-aware operator evaluation and sign reports."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CLASSICAL_CONVENTIONS = {"classical4d", "classical5d", "classical6d", "backend6"}


def convention_for_row(row: dict[str, Any]) -> str:
    value = (
        row.get("seidel_convention")
        or row.get("theta_convention")
        or row.get("model_name")
        or "classical6d"
    )
    value = str(value)
    if value == "backend6":
        return "backend6"
    if value not in CLASSICAL_CONVENTIONS:
        raise ValueError(f"Unsupported theta convention for gauge-aware pipeline: {value!r}")
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, tuple, dict)):
                    value = json.dumps(value, separators=(",", ":"))
                out[key] = value
            writer.writerow(out)


def run_command(cmd: list[str], *, cwd: Path) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dataset-twin-invariance-pass", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--combined-name", default="gauge_aware_operator_metrics.csv")
    parser.add_argument("--skip-sign-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = args.output_dir / "split_inputs"
    split_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(convention_for_row(row), []).append(row)

    evaluator_csvs: list[Path] = []
    for convention, group in sorted(grouped.items()):
        input_path = split_dir / f"{convention}_operator_input.csv"
        eval_dir = args.output_dir / f"evaluator_{convention}"
        write_csv(group, input_path)
        cmd = [
            str(args.python_bin),
            "scripts/evaluate_seidel_physical_operator_sweep.py",
            str(input_path),
            str(eval_dir),
            "--dim",
            str(int(args.dim)),
            "--theta-convention",
            convention,
            "--dataset-twin-invariance-pass",
            str(args.dataset_twin_invariance_pass),
        ]
        if args.resume:
            cmd.append("--resume")
        run_command(cmd, cwd=PROJECT_ROOT)
        evaluator_csvs.append(eval_dir / "seidel_physical_operator_metrics.csv")

    combined: list[dict[str, Any]] = []
    for path in evaluator_csvs:
        combined.extend(read_rows(path))
    combined_path = args.output_dir / args.combined_name
    write_csv(combined, combined_path)

    if not args.skip_sign_report:
        run_command(
            [
                str(args.python_bin),
                "scripts/build_gauge_aware_sign_report.py",
                str(combined_path),
                str(args.output_dir / "gauge_aware_sign_report"),
            ],
            cwd=PROJECT_ROOT,
        )

    summary = {
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "combined_csv": str(combined_path),
        "num_input_rows": len(rows),
        "num_output_rows": len(combined),
        "groups": {key: len(value) for key, value in sorted(grouped.items())},
        "dim": int(args.dim),
        "dataset_twin_invariance_pass": str(args.dataset_twin_invariance_pass),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {combined_path} ({len(combined)} rows)", flush=True)


if __name__ == "__main__":
    main()
