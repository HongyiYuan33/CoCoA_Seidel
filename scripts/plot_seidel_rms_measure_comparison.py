#!/usr/bin/env python3
"""Plot wavefront-RMS-prior vs coefficient-RMS-prior Seidel recovery trends."""

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

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import field_weighted_wavefront_rms  # noqa: E402


COEFF_LABELS = ("W040", "W131", "W222", "W220", "W311", "Wd")
MEASURE_LABELS = {
    "wavefront": "wavefront_RMS_prior",
    "coefficient": "coefficient_RMS_prior",
}
MEASURE_COLORS = {
    "wavefront": "#4c78a8",
    "coefficient": "#f58518",
}
MEASURE_OFFSETS = {
    "wavefront": -0.007,
    "coefficient": 0.007,
}


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def parse_vec(value: Any) -> np.ndarray:
    parsed = parse_jsonish(value)
    if parsed is None:
        raise ValueError("Missing Seidel vector")
    arr = np.asarray(parsed, dtype=np.float64).reshape(-1)
    if arr.size < 6:
        arr = np.pad(arr, (0, 6 - arr.size))
    return arr[:6]


def parse_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value in (None, ""):
        return float(default)
    return float(value)


def coeff_rms(coeffs: np.ndarray) -> float:
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)[:6]
    return float(np.sqrt(np.mean(coeffs * coeffs)))


def abs_coeff_cv(coeffs: np.ndarray) -> float:
    values = np.abs(np.asarray(coeffs, dtype=np.float64).reshape(-1)[:6])
    mean = float(np.mean(values))
    if mean <= 1e-12:
        return math.nan
    return float(np.std(values, ddof=0) / mean)


def w222_abs_share(coeffs: np.ndarray) -> float:
    values = np.abs(np.asarray(coeffs, dtype=np.float64).reshape(-1)[:6])
    denom = float(np.sum(values))
    if denom <= 1e-12:
        return math.nan
    return float(values[2] / denom)


def load_case_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            measure = str(row.get("seidel_rms_prior_measure") or "wavefront")
            if measure not in MEASURE_LABELS:
                continue
            gt = parse_vec(row["seidel_gt"])
            raw = parse_vec(row["seidel_final"])
            aligned = parse_vec(row["aligned_seidel_physical"])
            gt_wf = field_weighted_wavefront_rms(gt)
            raw_wf = field_weighted_wavefront_rms(raw)
            aligned_wf = field_weighted_wavefront_rms(aligned)
            gt_coeff = coeff_rms(gt)
            raw_coeff = coeff_rms(raw)
            aligned_coeff = coeff_rms(aligned)
            rows.append(
                {
                    "measure": measure,
                    "measure_label": MEASURE_LABELS[measure],
                    "seed": int(float(row.get("seed", 0))),
                    "image": row.get("image", ""),
                    "direction": row.get("direction", ""),
                    "target_wavefront_rms": parse_float(row, "target_wavefront_rms"),
                    "target_coeff_rms": parse_float(row, "target_coeff_rms"),
                    "lambda": parse_float(row, "lambda", parse_float(row, "seidel_rms_floor_weight")),
                    "alpha": parse_float(row, "alpha", parse_float(row, "seidel_rms_floor_alpha")),
                    "operator_error_calibrated": parse_float(row, "operator_error_calibrated"),
                    "operator_error_phys_equiv": parse_float(row, "operator_error_phys_equiv"),
                    "ssim_recon_gain_vs_gt": parse_float(row, "ssim_recon_gain_vs_gt"),
                    "nrmse_recon_gain_vs_gt": parse_float(row, "nrmse_recon_gain_vs_gt"),
                    "raw_wavefront_over_gt": raw_wf / max(gt_wf, 1e-12),
                    "aligned_wavefront_over_gt": aligned_wf / max(gt_wf, 1e-12),
                    "raw_coeff_over_gt": raw_coeff / max(gt_coeff, 1e-12),
                    "aligned_coeff_over_gt": aligned_coeff / max(gt_coeff, 1e-12),
                    "aligned_w222_abs_share": w222_abs_share(aligned),
                    "aligned_abs_coeff_cv": abs_coeff_cv(aligned),
                    "raw_abs_coeff_cv": abs_coeff_cv(raw),
                    "gt_wavefront_rms": gt_wf,
                    "raw_wavefront_rms": raw_wf,
                    "aligned_wavefront_rms": aligned_wf,
                    "gt_coeff_rms": gt_coeff,
                    "raw_coeff_rms": raw_coeff,
                    "aligned_coeff_rms": aligned_coeff,
                    "best_physical_transform": row.get("best_physical_transform", ""),
                    "metrics_path": row.get("metrics_path", ""),
                    "run_root": row.get("run_root", ""),
                }
            )
    rows.sort(key=lambda row: (float(row["target_wavefront_rms"]), int(row["seed"]), row["measure"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["measure"], float(row["target_wavefront_rms"]))].append(row)
    metrics = [
        "operator_error_calibrated",
        "aligned_wavefront_over_gt",
        "aligned_coeff_over_gt",
        "aligned_w222_abs_share",
        "aligned_abs_coeff_cv",
        "ssim_recon_gain_vs_gt",
        "nrmse_recon_gain_vs_gt",
    ]
    out: list[dict[str, Any]] = []
    for (measure, rms), values in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        row: dict[str, Any] = {
            "measure": measure,
            "measure_label": MEASURE_LABELS[measure],
            "target_wavefront_rms": rms,
            "n": len(values),
            "seeds": ",".join(str(v["seed"]) for v in sorted(values, key=lambda v: int(v["seed"]))),
        }
        for metric in metrics:
            arr = np.asarray([float(v[metric]) for v in values if math.isfinite(float(v[metric]))], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else math.nan
            row[f"{metric}_std"] = float(np.std(arr, ddof=0)) if arr.size else math.nan
        out.append(row)
    return out


def plot_metric(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
    hline: float | None = None,
) -> None:
    rms_values = sorted({float(row["target_wavefront_rms"]) for row in rows})
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=180)
    for measure in ("wavefront", "coefficient"):
        part = [row for row in rows if row["measure"] == measure]
        if not part:
            continue
        means = []
        stds = []
        for rms in rms_values:
            values = [float(row[metric]) for row in part if math.isclose(float(row["target_wavefront_rms"]), rms, abs_tol=1e-8)]
            arr = np.asarray(values, dtype=np.float64)
            means.append(float(np.mean(arr)) if arr.size else math.nan)
            stds.append(float(np.std(arr, ddof=0)) if arr.size else math.nan)
        xs = np.asarray(rms_values, dtype=np.float64) + MEASURE_OFFSETS[measure]
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            marker="o",
            linewidth=2,
            capsize=4,
            color=MEASURE_COLORS[measure],
            label=MEASURE_LABELS[measure],
        )
        for row in part:
            x = float(row["target_wavefront_rms"]) + MEASURE_OFFSETS[measure]
            ax.scatter(x, float(row[metric]), s=20, alpha=0.42, color=MEASURE_COLORS[measure])
    if hline is not None:
        ax.axhline(float(hline), color="0.35", linewidth=1.0, linestyle="--")
    ax.set_xticks(rms_values)
    ax.set_xticklabels([f"{rms:.2f}" for rms in rms_values])
    ax.set_xlabel("GT target wavefront RMS level")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_two_metrics(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    specs: list[tuple[str, str, str, float | None]],
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(specs), figsize=(7.2 * len(specs), 5.0), dpi=180)
    if len(specs) == 1:
        axes = [axes]
    rms_values = sorted({float(row["target_wavefront_rms"]) for row in rows})
    for ax, (metric, ylabel, subtitle, hline) in zip(axes, specs):
        for measure in ("wavefront", "coefficient"):
            part = [row for row in rows if row["measure"] == measure]
            means = []
            stds = []
            for rms in rms_values:
                values = [float(row[metric]) for row in part if math.isclose(float(row["target_wavefront_rms"]), rms, abs_tol=1e-8)]
                arr = np.asarray(values, dtype=np.float64)
                means.append(float(np.mean(arr)) if arr.size else math.nan)
                stds.append(float(np.std(arr, ddof=0)) if arr.size else math.nan)
            xs = np.asarray(rms_values, dtype=np.float64) + MEASURE_OFFSETS[measure]
            ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2, capsize=4, color=MEASURE_COLORS[measure], label=MEASURE_LABELS[measure])
            for row in part:
                x = float(row["target_wavefront_rms"]) + MEASURE_OFFSETS[measure]
                ax.scatter(x, float(row[metric]), s=20, alpha=0.42, color=MEASURE_COLORS[measure])
        if hline is not None:
            ax.axhline(float(hline), color="0.35", linewidth=1.0, linestyle="--")
        ax.set_xticks(rms_values)
        ax.set_xticklabels([f"{rms:.2f}" for rms in rms_values])
        ax.set_xlabel("GT target wavefront RMS level")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)


def write_readme(summary: list[dict[str, Any]], out_dir: Path, csv_path: Path) -> None:
    lines = [
        "# Seidel RMS Prior Measure Comparison",
        "",
        f"Input evaluator CSV: `{csv_path}`",
        "",
        "Comparison: `wavefront_RMS_prior` vs `coefficient_RMS_prior`.",
        "Rows are dendrites / signed_balanced / amp_direction / lambda=1000 / alpha=1 / seeds 0-3.",
        "",
        "| prior measure | GT RMS | n | op err | aligned WF/GT | aligned coeff/GT | W222 share | coeff CV | SSIM | NRMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['measure_label']} | {float(row['target_wavefront_rms']):.2f} | {int(row['n'])} | "
            f"{float(row['operator_error_calibrated_mean']):.4f} +- {float(row['operator_error_calibrated_std']):.4f} | "
            f"{float(row['aligned_wavefront_over_gt_mean']):.3f} +- {float(row['aligned_wavefront_over_gt_std']):.3f} | "
            f"{float(row['aligned_coeff_over_gt_mean']):.3f} +- {float(row['aligned_coeff_over_gt_std']):.3f} | "
            f"{float(row['aligned_w222_abs_share_mean']):.3f} | "
            f"{float(row['aligned_abs_coeff_cv_mean']):.3f} | "
            f"{float(row['ssim_recon_gain_vs_gt_mean']):.4f} | "
            f"{float(row['nrmse_recon_gain_vs_gt_mean']):.4f} |"
        )
    lines += [
        "",
        "Generated plots:",
        "- `01_operator_error_calibrated_vs_rms_level.png`",
        "- `02_recovered_over_gt_wavefront_rms_vs_rms_level.png`",
        "- `03_recovered_over_gt_coefficient_rms_vs_rms_level.png`",
        "- `04_w222_share_and_coeff_cv_vs_rms_level.png`",
        "- `05_ssim_nrmse_vs_rms_level.png`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_case_rows(args.csv)
    if not rows:
        raise RuntimeError(f"No wavefront/coefficient RMS prior rows found in {args.csv}")
    summary = summarize(rows)
    write_csv(args.out_dir / "case_metrics_with_aligned_rms.csv", rows)
    write_csv(args.out_dir / "summary_by_prior_measure_and_rms.csv", summary)
    plot_metric(
        rows,
        args.out_dir / "01_operator_error_calibrated_vs_rms_level.png",
        metric="operator_error_calibrated",
        ylabel="operator_error_calibrated",
        title="Operator error vs RMS level",
    )
    plot_metric(
        rows,
        args.out_dir / "02_recovered_over_gt_wavefront_rms_vs_rms_level.png",
        metric="aligned_wavefront_over_gt",
        ylabel="aligned recovered / GT wavefront RMS",
        title="Recovered wavefront strength vs RMS level",
        hline=1.0,
    )
    plot_metric(
        rows,
        args.out_dir / "03_recovered_over_gt_coefficient_rms_vs_rms_level.png",
        metric="aligned_coeff_over_gt",
        ylabel="aligned recovered / GT coefficient RMS",
        title="Recovered coefficient strength vs RMS level",
        hline=1.0,
    )
    plot_two_metrics(
        rows,
        args.out_dir / "04_w222_share_and_coeff_cv_vs_rms_level.png",
        specs=[
            ("aligned_w222_abs_share", "W222 abs share", "W222 share of |aligned coefficients|", None),
            ("aligned_abs_coeff_cv", "CV of |aligned coefficients|", "Coefficient magnitude CV", None),
        ],
        title="Coefficient concentration vs RMS level",
    )
    plot_two_metrics(
        rows,
        args.out_dir / "05_ssim_nrmse_vs_rms_level.png",
        specs=[
            ("ssim_recon_gain_vs_gt", "SSIM gain", "Reconstruction SSIM", None),
            ("nrmse_recon_gain_vs_gt", "NRMSE gain", "Reconstruction NRMSE", None),
        ],
        title="Reconstruction quality vs RMS level",
    )
    write_readme(summary, args.out_dir, args.csv)
    print(f"[done] wrote RMS-measure comparison plots to {args.out_dir}")


if __name__ == "__main__":
    main()
