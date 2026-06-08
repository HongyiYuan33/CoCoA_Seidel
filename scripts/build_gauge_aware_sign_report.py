#!/usr/bin/env python3
"""Build gauge-aware sign and canonical metric reports from evaluator CSVs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COEFF_LABELS = ["W040", "W131", "W222", "W220", "W311", "Wd"]
SIGN_SOURCES = ["raw", "physical", "gauge"]


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, int, float, bool)) or value is None:
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def parse_match_vector(value: Any) -> list[bool | None]:
    parsed = parse_jsonish(value)
    if parsed is None:
        return []
    out: list[bool | None] = []
    for item in parsed:
        if item is None:
            out.append(None)
        elif isinstance(item, bool):
            out.append(item)
        elif str(item).strip().lower() in {"1", "true", "yes"}:
            out.append(True)
        elif str(item).strip().lower() in {"0", "false", "no"}:
            out.append(False)
        else:
            out.append(None)
    return out


def parse_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        writer.writerows(rows)


def group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("seidel_convention") or row.get("theta_convention") or "unknown"),
        str(row.get("oracle_mode") or row.get("method") or row.get("mode") or "all"),
        str(row.get("target_wavefront_rms") or row.get("strength") or "all"),
        str(row.get("direction") or "all"),
    )


def summarize_signs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    summary: list[dict[str, Any]] = []
    for (dimension, mode, target_rms, direction), group in sorted(groups.items()):
        for coeff_idx, coeff_name in enumerate(COEFF_LABELS):
            out: dict[str, Any] = {
                "seidel_convention": dimension,
                "mode": mode,
                "target_wavefront_rms": target_rms,
                "direction": direction,
                "coefficient": coeff_name,
            }
            for source in SIGN_SOURCES:
                total = 0
                matches = 0
                for row in group:
                    vector = parse_match_vector(row.get(f"canonical_sign_match_{source}"))
                    if coeff_idx >= len(vector):
                        continue
                    value = vector[coeff_idx]
                    if value is None:
                        continue
                    total += 1
                    matches += int(bool(value))
                out[f"{source}_valid_n"] = total
                out[f"{source}_match_n"] = matches
                out[f"{source}_match_rate"] = float(matches / total) if total else math.nan
            summary.append(out)
    return summary


def summarize_metric_by_rms(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        target = str(row.get("target_wavefront_rms") or row.get("strength") or "")
        if not target:
            continue
        value = parse_float(row, metric)
        if not math.isfinite(value):
            continue
        groups[
            (
                str(row.get("seidel_convention") or row.get("theta_convention") or "unknown"),
                str(row.get("oracle_mode") or row.get("method") or row.get("mode") or "all"),
                target,
            )
        ].append(value)
    out = []
    for (dimension, mode, target), values in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], float(item[0][2]))):
        out.append(
            {
                "seidel_convention": dimension,
                "mode": mode,
                "target_wavefront_rms": target,
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }
        )
    return out


def plot_sign_summary(summary: list[dict[str, Any]], out_path: Path) -> None:
    if not summary:
        return
    dims = sorted({row["seidel_convention"] for row in summary})
    modes = sorted({row["mode"] for row in summary})
    fig, axes = plt.subplots(
        len(dims),
        len(modes),
        figsize=(max(8, 4.8 * len(modes)), max(4, 3.2 * len(dims))),
        squeeze=False,
    )
    colors = {"raw": "#777777", "physical": "#3b83bd", "gauge": "#31a354"}
    width = 0.24
    x = np.arange(len(COEFF_LABELS), dtype=np.float64)
    for r_idx, dim in enumerate(dims):
        for c_idx, mode in enumerate(modes):
            ax = axes[r_idx, c_idx]
            rows = [
                row
                for row in summary
                if row["seidel_convention"] == dim
                and row["mode"] == mode
                and row["target_wavefront_rms"] == "all"
                and row["direction"] == "all"
            ]
            if not rows:
                rows = [
                    row
                    for row in summary
                    if row["seidel_convention"] == dim and row["mode"] == mode
                ]
            by_coeff = {row["coefficient"]: row for row in rows}
            for s_idx, source in enumerate(SIGN_SOURCES):
                y = [
                    float(by_coeff.get(coeff, {}).get(f"{source}_match_rate", math.nan))
                    for coeff in COEFF_LABELS
                ]
                ax.bar(x + (s_idx - 1) * width, y, width=width, color=colors[source], label=source)
            ax.set_ylim(0.0, 1.05)
            ax.set_xticks(x)
            ax.set_xticklabels(COEFF_LABELS, rotation=25, ha="right")
            ax.set_title(f"{dim} | {mode}", fontsize=10, fontweight="bold")
            ax.set_ylabel("sign match rate")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_by_rms(rows: list[dict[str, Any]], metric: str, out_path: Path, ylabel: str) -> None:
    summary = summarize_metric_by_rms(rows, metric)
    if not summary:
        return
    dims = sorted({row["seidel_convention"] for row in summary})
    modes = sorted({row["mode"] for row in summary})
    fig, axes = plt.subplots(1, len(dims), figsize=(max(7, 6 * len(dims)), 4.2), squeeze=False)
    for c_idx, dim in enumerate(dims):
        ax = axes[0, c_idx]
        for mode in modes:
            sub = sorted(
                [row for row in summary if row["seidel_convention"] == dim and row["mode"] == mode],
                key=lambda row: float(row["target_wavefront_rms"]),
            )
            if not sub:
                continue
            x = [float(row["target_wavefront_rms"]) for row in sub]
            y = [float(row["mean"]) for row in sub]
            ax.plot(x, y, marker="o", linewidth=2, label=mode)
        ax.set_title(dim, fontsize=10, fontweight="bold")
        ax.set_xlabel("GT Seidel wavefront RMS")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def add_all_rollups(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(summary)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        groups[(row["seidel_convention"], row["mode"], row["coefficient"])].append(row)
    for (dimension, mode, coeff), group in groups.items():
        out = {
            "seidel_convention": dimension,
            "mode": mode,
            "target_wavefront_rms": "all",
            "direction": "all",
            "coefficient": coeff,
        }
        for source in SIGN_SOURCES:
            total = sum(int(row[f"{source}_valid_n"]) for row in group)
            matches = sum(int(row[f"{source}_match_n"]) for row in group)
            out[f"{source}_valid_n"] = total
            out[f"{source}_match_n"] = matches
            out[f"{source}_match_rate"] = float(matches / total) if total else math.nan
        rows.append(out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.input_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sign_summary = add_all_rollups(summarize_signs(rows))
    write_csv(args.out_dir / "sign_agreement_raw_vs_physical_vs_gauge_by_coeff.csv", sign_summary)
    plot_sign_summary(
        sign_summary,
        args.out_dir / "sign_agreement_raw_vs_physical_vs_gauge_by_coeff.png",
    )
    write_csv(
        args.out_dir / "canonical_operator_error_by_rms.csv",
        summarize_metric_by_rms(rows, "canonical_operator_error_gauge"),
    )
    write_csv(
        args.out_dir / "canonical_recovered_over_gt_rms_by_rms.csv",
        summarize_metric_by_rms(rows, "canonical_recovered_over_gt_wavefront_rms_gauge"),
    )
    plot_metric_by_rms(
        rows,
        "canonical_operator_error_gauge",
        args.out_dir / "canonical_operator_error_by_rms.png",
        "gauge-canonical operator error",
    )
    plot_metric_by_rms(
        rows,
        "canonical_recovered_over_gt_wavefront_rms_gauge",
        args.out_dir / "canonical_recovered_over_gt_rms_by_rms.png",
        "gauge-canonical recovered / GT wavefront RMS",
    )
    print(f"[done] wrote gauge-aware sign report to {args.out_dir}")


if __name__ == "__main__":
    main()
