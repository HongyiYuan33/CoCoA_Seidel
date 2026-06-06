#!/usr/bin/env python3
"""Merge two operator-evaluator CSVs into one keyed by synthetic lambda values."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def keep_row(row: dict[str, Any], images: set[str], rms_values: set[float], direction: str) -> bool:
    if str(row.get("image")) not in images:
        return False
    if str(row.get("direction")) != direction:
        return False
    try:
        rms = round(float(row["target_wavefront_rms"]), 6)
    except (KeyError, TypeError, ValueError):
        return False
    if rms not in rms_values:
        return False
    try:
        lam = float(row.get("lambda", 0.0))
    except (TypeError, ValueError):
        return False
    return lam == 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-csv", type=Path, required=True)
    parser.add_argument("--bottom-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-lambda", type=float, default=0.0)
    parser.add_argument("--bottom-lambda", type=float, default=1.0)
    parser.add_argument("--direction", default="signed_balanced")
    parser.add_argument("--images", nargs="+", default=["Iksung_beads", "dendrites", "dendrites_dense"])
    parser.add_argument("--rms-values", nargs="+", type=float, default=[0.06, 0.20, 0.40])
    parser.add_argument("--expected", type=int, default=18)
    args = parser.parse_args()

    images = set(args.images)
    rms_values = {round(float(value), 6) for value in args.rms_values}
    rows: list[dict[str, Any]] = []
    for path, lambda_value, method_name in [
        (args.top_csv, args.top_lambda, "top"),
        (args.bottom_csv, args.bottom_lambda, "bottom"),
    ]:
        for row in read_rows(path):
            if not keep_row(row, images, rms_values, args.direction):
                continue
            out = dict(row)
            out["lambda"] = lambda_value
            out["pair_method"] = method_name
            rows.append(out)

    rows.sort(key=lambda row: (float(row["target_wavefront_rms"]), str(row["image"]), float(row["lambda"])))
    if args.expected is not None and len(rows) != int(args.expected):
        raise RuntimeError(f"Expected {args.expected} rows, found {len(rows)}")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[merge] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
