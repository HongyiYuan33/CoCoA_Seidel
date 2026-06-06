#!/usr/bin/env python3
"""Compare W040-fixed Seidel recovery against the ratio-target baseline."""

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


IMAGES = {"Iksung_beads", "dendrites", "dendrites_dense"}
RMS_VALUES = {0.06, 0.2, 0.4}


def parse_vec(value: Any) -> np.ndarray:
    text = str(value).strip()
    try:
        return np.asarray(json.loads(text), dtype=float)
    except Exception:
        return np.asarray(ast.literal_eval(text), dtype=float)


def load_rows(path: Path, method_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if float(row.get("lambda", 1000)) != 1000:
                continue
            if str(row.get("direction")) != "signed_balanced":
                continue
            if str(row.get("image")) not in IMAGES:
                continue
            rms = round(float(row["target_wavefront_rms"]), 6)
            if rms not in RMS_VALUES:
                continue
            gt = parse_vec(row["seidel_gt"])
            raw = parse_vec(row["seidel_final"])
            aligned = parse_vec(row["aligned_seidel_physical"])
            gt_rms = field_weighted_wavefront_rms(gt)
            raw_rms = field_weighted_wavefront_rms(raw)
            aligned_rms = field_weighted_wavefront_rms(aligned)
            rows.append(
                {
                    "method_name": method_name,
                    "image": row["image"],
                    "rms": rms,
                    "operator_error": float(row["operator_error_calibrated"]),
                    "ssim": float(row["ssim_recon_gain_vs_gt"]),
                    "nrmse": float(row["nrmse_recon_gain_vs_gt"]),
                    "raw_over_gt": raw_rms / max(gt_rms, 1e-12),
                    "aligned_over_gt": aligned_rms / max(gt_rms, 1e-12),
                    "prior_loss": float(row.get("final_seidel_rms_floor_loss") or 0.0),
                    "gt_fixed_seidel_indices": row.get("gt_fixed_seidel_indices", ""),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--fixed-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.baseline_csv, "baseline_ratio1") + load_rows(args.fixed_csv, "w040_fixed_gt")
    by_key = {(row["method_name"], row["image"], row["rms"]): row for row in rows}
    keys = sorted({(row["image"], row["rms"]) for row in rows}, key=lambda item: (item[1], item[0]))

    comparison: list[dict[str, Any]] = []
    for image, rms in keys:
        baseline = by_key.get(("baseline_ratio1", image, rms))
        fixed = by_key.get(("w040_fixed_gt", image, rms))
        if not baseline or not fixed:
            continue
        out = {"image": image, "rms": rms}
        for metric in [
            "operator_error",
            "ssim",
            "nrmse",
            "raw_over_gt",
            "aligned_over_gt",
            "prior_loss",
            "w040_gt",
            "w040_raw",
            "w040_aligned",
        ]:
            out[f"baseline_{metric}"] = baseline[metric]
            out[f"w040fixed_{metric}"] = fixed[metric]
            out[f"delta_{metric}"] = fixed[metric] - baseline[metric]
        comparison.append(out)

    summary: list[dict[str, Any]] = []
    for method_name in ["baseline_ratio1", "w040_fixed_gt"]:
        for rms in [0.06, 0.2, 0.4]:
            subset = [row for row in rows if row["method_name"] == method_name and abs(row["rms"] - rms) < 1e-9]
            if not subset:
                continue
            summary.append(
                {
                    "method_name": method_name,
                    "rms": rms,
                    "n": len(subset),
                    "operator_error": float(np.mean([row["operator_error"] for row in subset])),
                    "ssim": float(np.mean([row["ssim"] for row in subset])),
                    "nrmse": float(np.mean([row["nrmse"] for row in subset])),
                    "raw_over_gt": float(np.mean([row["raw_over_gt"] for row in subset])),
                    "aligned_over_gt": float(np.mean([row["aligned_over_gt"] for row in subset])),
                    "prior_loss": float(np.mean([row["prior_loss"] for row in subset])),
                }
            )

    write_csv(args.out_dir / "all_rows.csv", rows)
    write_csv(args.out_dir / "case_compare.csv", comparison)
    write_csv(args.out_dir / "summary_by_method_rms.csv", summary)

    rms_vals = [0.06, 0.2, 0.4]
    colors = {"baseline_ratio1": "#4c78a8", "w040_fixed_gt": "#f58518"}
    method_labels = {"baseline_ratio1": "ratio target=1x", "w040_fixed_gt": "W040 fixed to GT"}
    metrics = [
        ("operator_error", "operator error calibrated"),
        ("aligned_over_gt", "aligned recovered/GT RMS"),
        ("ssim", "SSIM gain"),
        ("nrmse", "NRMSE gain"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180)
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for method_name in ["baseline_ratio1", "w040_fixed_gt"]:
            values = []
            for rms in rms_vals:
                found = [
                    row
                    for row in summary
                    if row["method_name"] == method_name and abs(row["rms"] - rms) < 1e-9
                ]
                values.append(found[0][metric] if found else np.nan)
            ax.plot(rms_vals, values, marker="o", label=method_labels[method_name], color=colors[method_name])
            for x_val, y_val in zip(rms_vals, values):
                if np.isfinite(y_val):
                    ax.text(x_val, y_val, f"{y_val:.3f}", fontsize=8, ha="center", va="bottom")
        ax.set_xlabel("GT target wavefront RMS")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("W040 fixed-to-GT vs ratio-target baseline", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out_dir / "01_w040_fixed_vs_baseline_by_rms.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=180, sharex=True)
    case_labels = [f"{row['image']}\nRMS {row['rms']:.2f}" for row in comparison]
    x = np.arange(len(comparison))
    delta_specs = [
        ("operator_error", "Delta operator error (fixed - baseline)"),
        ("aligned_over_gt", "Delta aligned recovered/GT RMS"),
        ("ssim", "Delta SSIM"),
    ]
    for ax, (metric, ylabel) in zip(axes, delta_specs):
        values = [row[f"delta_{metric}"] for row in comparison]
        ax.axhline(0, color="0.35", linewidth=0.9)
        bar_colors = []
        for value in values:
            better = value < 0 if metric == "operator_error" else value > 0
            bar_colors.append("#54a24b" if better else "#e45756")
        ax.bar(x, values, color=bar_colors)
        for idx, value in enumerate(values):
            ax.text(
                idx,
                value,
                f"{value:+.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(case_labels, fontsize=8)
    fig.suptitle("Per-case change after fixing W040 to GT", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out_dir / "02_per_case_delta_w040_fixed.png")
    plt.close(fig)

    lines = [
        "# W040 fixed-to-GT comparison",
        "",
        "Baseline is ratio target alpha=1, lambda=1000, full 6D trainable. "
        "New run locks W040 to its GT value and recovers the other five coefficients.",
        "",
        "## Mean by RMS",
        "",
        "| method | RMS | op err | aligned/GT RMS | raw/GT RMS | SSIM | NRMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method_name']} | {row['rms']:.2f} | {row['operator_error']:.4f} | "
            f"{row['aligned_over_gt']:.3f} | {row['raw_over_gt']:.3f} | "
            f"{row['ssim']:.3f} | {row['nrmse']:.3f} |"
        )
    lines += [
        "",
        "## Case Deltas",
        "",
        "| image | RMS | delta op | delta aligned/GT | delta SSIM | baseline op | fixed op | baseline SSIM | fixed SSIM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['image']} | {row['rms']:.2f} | {row['delta_operator_error']:+.4f} | "
            f"{row['delta_aligned_over_gt']:+.3f} | {row['delta_ssim']:+.3f} | "
            f"{row['baseline_operator_error']:.4f} | {row['w040fixed_operator_error']:.4f} | "
            f"{row['baseline_ssim']:.3f} | {row['w040fixed_ssim']:.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Fixing W040 to GT strongly reduces operator error at RMS 0.20 and 0.40.",
        "- The recovered Seidel RMS becomes lower than the baseline because one dominant term is no longer trainable and the remaining five terms do not need to inflate as much.",
        "- Object SSIM is mixed: high-RMS dendrites improve, but Iksung RMS0.20 remains the same collapsed-object outlier.",
        "",
    ]
    (args.out_dir / "trend_report.md").write_text("\n".join(lines))
    print(args.out_dir)


if __name__ == "__main__":
    main()
