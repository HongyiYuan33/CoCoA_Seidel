#!/usr/bin/env python3
"""Compare W040-fast coefficient-LR runs against baseline variants."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import field_weighted_wavefront_rms


IMAGES = ("Iksung_beads", "dendrites", "dendrites_dense")
RMS_VALUES = (0.06, 0.2, 0.4)


def parse_vec(value: Any) -> np.ndarray:
    text = str(value).strip()
    try:
        return np.asarray(json.loads(text), dtype=float)
    except Exception:
        return np.asarray(ast.literal_eval(text), dtype=float)


def parse_method_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Method spec must be LABEL=CSV, got {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty method label in {spec!r}")
    return label, Path(path)


def load_rows(path: Path, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("direction")) != "signed_balanced":
                continue
            if str(row.get("image")) not in set(IMAGES):
                continue
            rms = round(float(row["target_wavefront_rms"]), 6)
            if rms not in set(RMS_VALUES):
                continue
            if float(row.get("lambda", 1000.0)) != 1000.0:
                continue
            gt = parse_vec(row["seidel_gt"])
            raw = parse_vec(row["seidel_final"])
            aligned = parse_vec(row["aligned_seidel_physical"])
            gt_rms = field_weighted_wavefront_rms(gt)
            raw_rms = field_weighted_wavefront_rms(raw)
            aligned_rms = field_weighted_wavefront_rms(aligned)
            rows.append(
                {
                    "method": method,
                    "image": row["image"],
                    "rms": rms,
                    "operator_error": float(row["operator_error_calibrated"]),
                    "ssim": float(row["ssim_recon_gain_vs_gt"]),
                    "nrmse": float(row["nrmse_recon_gain_vs_gt"]),
                    "raw_over_gt": raw_rms / max(gt_rms, 1e-12),
                    "aligned_over_gt": aligned_rms / max(gt_rms, 1e-12),
                    "prior_loss": float(row.get("final_seidel_rms_floor_loss") or 0.0),
                    "w040_gt": float(gt[0]),
                    "w040_raw": float(raw[0]),
                    "w040_aligned": float(aligned[0]),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for method in methods:
        for rms in RMS_VALUES:
            subset = [row for row in rows if row["method"] == method and abs(row["rms"] - rms) < 1e-9]
            if not subset:
                continue
            summary.append(
                {
                    "method": method,
                    "rms": rms,
                    "n": len(subset),
                    "operator_error": float(np.mean([row["operator_error"] for row in subset])),
                    "aligned_over_gt": float(np.mean([row["aligned_over_gt"] for row in subset])),
                    "raw_over_gt": float(np.mean([row["raw_over_gt"] for row in subset])),
                    "ssim": float(np.mean([row["ssim"] for row in subset])),
                    "nrmse": float(np.mean([row["nrmse"] for row in subset])),
                    "prior_loss": float(np.mean([row["prior_loss"] for row in subset])),
                }
            )
    return summary


def case_compare(rows: list[dict[str, Any]], methods: list[str], baseline: str) -> list[dict[str, Any]]:
    by_key = {(row["method"], row["image"], row["rms"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for image in IMAGES:
        for rms in RMS_VALUES:
            base = by_key.get((baseline, image, rms))
            if not base:
                continue
            for method in methods:
                row = by_key.get((method, image, rms))
                if not row or method == baseline:
                    continue
                output.append(
                    {
                        "method": method,
                        "image": image,
                        "rms": rms,
                        "delta_operator_error": row["operator_error"] - base["operator_error"],
                        "delta_aligned_over_gt": row["aligned_over_gt"] - base["aligned_over_gt"],
                        "delta_ssim": row["ssim"] - base["ssim"],
                        "baseline_operator_error": base["operator_error"],
                        "method_operator_error": row["operator_error"],
                        "baseline_aligned_over_gt": base["aligned_over_gt"],
                        "method_aligned_over_gt": row["aligned_over_gt"],
                        "baseline_ssim": base["ssim"],
                        "method_ssim": row["ssim"],
                    }
                )
    return output


def plot_summary(summary: list[dict[str, Any]], methods: list[str], out_dir: Path) -> None:
    colors = {
        "baseline_ratio1": "#4c78a8",
        "w040_fixed_gt": "#f58518",
        "coeflr_1e3_1e4": "#e45756",
        "coeflr_2e3_2e4": "#b279a2",
    }
    labels = {
        "baseline_ratio1": "baseline ratio=1",
        "w040_fixed_gt": "W040 fixed GT",
        "coeflr_1e3_1e4": "W040 1e-3 / other 1e-4",
        "coeflr_2e3_2e4": "W040 2e-3 / other 2e-4",
    }
    metric_specs = [
        ("operator_error", "operator error calibrated"),
        ("aligned_over_gt", "aligned recovered/GT RMS"),
        ("ssim", "SSIM gain"),
        ("nrmse", "NRMSE gain"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=180)
    for ax, (metric, ylabel) in zip(axes.ravel(), metric_specs):
        for method in methods:
            values = []
            for rms in RMS_VALUES:
                found = [row for row in summary if row["method"] == method and abs(row["rms"] - rms) < 1e-9]
                values.append(found[0][metric] if found else np.nan)
            ax.plot(
                RMS_VALUES,
                values,
                marker="o",
                linewidth=2,
                color=colors.get(method),
                label=labels.get(method, method),
            )
            for x_val, y_val in zip(RMS_VALUES, values):
                if np.isfinite(y_val):
                    ax.text(x_val, y_val, f"{y_val:.3f}", fontsize=7, ha="center", va="bottom")
        ax.set_xlabel("GT target wavefront RMS")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Coefficient-specific LR comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "01_coeflr_methods_by_rms.png")
    plt.close(fig)


def write_report(
    summary: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    methods: list[str],
    out_dir: Path,
) -> None:
    lines = [
        "# Coefficient-specific LR comparison",
        "",
        "All runs use ratio target alpha=1, lambda=1000, signed_balanced, 6D, pre400/joint1000.",
        "Coefficient-LR runs keep all 6 coefficients trainable and use `[10,1,1,1,1,1]` gradient multipliers.",
        "",
        "## Mean by RMS",
        "",
        "| method | RMS | op err | aligned/GT RMS | raw/GT RMS | SSIM | NRMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['rms']:.2f} | {row['operator_error']:.4f} | "
            f"{row['aligned_over_gt']:.3f} | {row['raw_over_gt']:.3f} | "
            f"{row['ssim']:.3f} | {row['nrmse']:.3f} |"
        )
    lines += [
        "",
        "## Main Read",
        "",
        "- The coefficient-specific LR settings are worse than the baseline at RMS 0.20 and 0.40.",
        "- `W040=1e-3, other=1e-4` is slightly good only at RMS 0.06 operator error, but fails for high RMS.",
        "- `W040=2e-3, other=2e-4` is not a rescue; it is also worse than baseline at all high-RMS cases.",
        "- Hard W040=GT remains much stronger than W040-fast LR, so the problem is not simply that W040 needs faster updates.",
        "",
        "## Case Deltas vs Baseline",
        "",
        "| method | image | RMS | delta op | delta aligned/GT | delta SSIM | method op | method SSIM |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['method']} | {row['image']} | {row['rms']:.2f} | "
            f"{row['delta_operator_error']:+.4f} | {row['delta_aligned_over_gt']:+.3f} | "
            f"{row['delta_ssim']:+.3f} | {row['method_operator_error']:.4f} | {row['method_ssim']:.3f} |"
        )
    lines.append("")
    (out_dir / "trend_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True, help="LABEL=CSV")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="baseline_ratio1")
    args = parser.parse_args()

    methods: list[str] = []
    rows: list[dict[str, Any]] = []
    for spec in args.method:
        method, path = parse_method_spec(spec)
        methods.append(method)
        rows.extend(load_rows(path, method))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, methods)
    comparison = case_compare(rows, methods, args.baseline)
    write_csv(args.out_dir / "all_rows.csv", rows)
    write_csv(args.out_dir / "summary_by_method_rms.csv", summary)
    write_csv(args.out_dir / "case_delta_vs_baseline.csv", comparison)
    plot_summary(summary, methods, args.out_dir)
    write_report(summary, comparison, methods, args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
