#!/usr/bin/env python3
"""Analyze whether Seidel RMS losses equalize recovered coefficients."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import (
    field_weighted_wavefront_rms,
)


COEFF_LABELS = ["W040", "W131", "W222", "W220", "W311", "Wd"]


def parse_vector(value: str) -> np.ndarray:
    if value is None:
        raise ValueError("missing vector")
    parsed = json.loads(value)
    arr = np.asarray(parsed, dtype=np.float64).reshape(-1)
    if arr.size < 6:
        arr = np.pad(arr, (0, 6 - arr.size))
    return arr[:6]


def parse_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    return float(value)


def coeff_equalization_metrics(coeffs: np.ndarray, active_mask: np.ndarray) -> dict[str, float]:
    abs_all = np.abs(coeffs)
    abs_active = abs_all[active_mask]

    def safe_cv(values: np.ndarray) -> float:
        mean = float(np.mean(values))
        if mean <= 1e-12:
            return 0.0
        return float(np.std(values) / mean)

    def safe_dominance(values: np.ndarray) -> float:
        total = float(np.sum(values))
        if total <= 1e-12:
            return 0.0
        return float(np.max(values) / total)

    def safe_max_over_mean(values: np.ndarray) -> float:
        mean = float(np.mean(values))
        if mean <= 1e-12:
            return 0.0
        return float(np.max(values) / mean)

    return {
        "abs_cv_all6": safe_cv(abs_all),
        "abs_cv_gt_active": safe_cv(abs_active),
        "dominance_all6": safe_dominance(abs_all),
        "dominance_gt_active": safe_dominance(abs_active),
        "max_over_mean_all6": safe_max_over_mean(abs_all),
        "mean_abs_all6": float(np.mean(abs_all)),
        "mean_abs_gt_active": float(np.mean(abs_active)),
    }


def load_method_rows(method: str, csv_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gt = parse_vector(row["seidel_gt"])
            raw = parse_vector(row["seidel_final"])
            aligned = parse_vector(row["aligned_seidel_physical"])
            active_mask = np.abs(gt) > 1e-10
            if not np.any(active_mask):
                active_mask = np.ones(6, dtype=bool)

            target_rms = parse_float(row, "target_wavefront_rms")
            lam = parse_float(row, "lambda", parse_float(row, "seidel_rms_floor_weight"))
            image = row.get("image") or row.get("object_label") or row.get("sample") or "unknown"
            direction = row.get("direction") or row.get("seidel_name") or ""

            for coeff_kind, coeffs in (
                ("GT", gt),
                ("raw_recovered", raw),
                ("aligned_recovered", aligned),
            ):
                metrics = coeff_equalization_metrics(coeffs, active_mask)
                rms = field_weighted_wavefront_rms(coeffs)
                rows.append(
                    {
                        "method": method,
                        "coeff_kind": coeff_kind,
                        "image": image,
                        "direction": direction,
                        "target_wavefront_rms": target_rms,
                        "lambda": lam,
                        "alpha": parse_float(row, "alpha"),
                        "operator_error_calibrated": parse_float(row, "operator_error_calibrated"),
                        "ssim_recon_gain_vs_gt": parse_float(row, "ssim_recon_gain_vs_gt"),
                        "coeffs": coeffs,
                        "gt_coeffs": gt,
                        "wavefront_rms": rms,
                        "wavefront_over_gt_rms": rms / max(field_weighted_wavefront_rms(gt), 1e-12),
                        **metrics,
                    }
                )
    return rows


def write_case_rows(rows: list[dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "method",
        "coeff_kind",
        "image",
        "direction",
        "target_wavefront_rms",
        "lambda",
        "alpha",
        "operator_error_calibrated",
        "ssim_recon_gain_vs_gt",
        "wavefront_rms",
        "wavefront_over_gt_rms",
        "abs_cv_all6",
        "abs_cv_gt_active",
        "dominance_all6",
        "dominance_gt_active",
        "max_over_mean_all6",
        "mean_abs_all6",
        "mean_abs_gt_active",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["coeff_kind"], row["target_wavefront_rms"])].append(row)

    metric_keys = [
        "abs_cv_all6",
        "abs_cv_gt_active",
        "dominance_all6",
        "dominance_gt_active",
        "max_over_mean_all6",
        "mean_abs_all6",
        "mean_abs_gt_active",
        "wavefront_rms",
        "wavefront_over_gt_rms",
        "operator_error_calibrated",
        "ssim_recon_gain_vs_gt",
    ]
    summary: list[dict[str, object]] = []
    for key, values in sorted(grouped.items(), key=lambda item: (str(item[0][0]), str(item[0][1]), float(item[0][2]))):
        out: dict[str, object] = {
            "method": key[0],
            "coeff_kind": key[1],
            "target_wavefront_rms": key[2],
            "n": len(values),
        }
        for metric in metric_keys:
            arr = np.asarray([float(v[metric]) for v in values if not math.isnan(float(v[metric]))], dtype=np.float64)
            out[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else math.nan
            out[f"{metric}_std"] = float(np.std(arr)) if arr.size else math.nan
        summary.append(out)
    return summary


def write_summary(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_cv_summary(summary: list[dict[str, object]], out_path: Path) -> None:
    methods = [
        "GT_reference",
        "ratio1_lam1000",
        "ratio1_lam10000",
        "w040_fixed_gt_lam1000",
        "coeflr_w0401e3_other1e4_lam1000",
        "coeflr_w0402e3_other2e4_lam1000",
    ]
    colors = {
        "GT_reference": "#222222",
        "ratio1_lam1000": "#1f77b4",
        "ratio1_lam10000": "#17becf",
        "w040_fixed_gt_lam1000": "#2ca02c",
        "coeflr_w0401e3_other1e4_lam1000": "#d62728",
        "coeflr_w0402e3_other2e4_lam1000": "#ff7f0e",
    }
    markers = {
        "GT_reference": "s",
        "ratio1_lam1000": "o",
        "ratio1_lam10000": "o",
        "w040_fixed_gt_lam1000": "^",
        "coeflr_w0401e3_other1e4_lam1000": "D",
        "coeflr_w0402e3_other2e4_lam1000": "v",
    }

    fig, ax = plt.subplots(figsize=(11, 6.2))
    for method in methods:
        selected = [
            row
            for row in summary
            if row["method"] == method
            and (
                (method == "GT_reference" and row["coeff_kind"] == "GT")
                or (method != "GT_reference" and row["coeff_kind"] == "aligned_recovered")
            )
        ]
        if not selected:
            continue
        xs = [float(row["target_wavefront_rms"]) for row in selected]
        ys = [float(row["abs_cv_all6_mean"]) for row in selected]
        ax.plot(xs, ys, marker=markers[method], linewidth=2, markersize=7, color=colors[method], label=method)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.015, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Coefficient magnitude equalization: lower CV means more equal abs(coeffs)")
    ax.set_xlabel("GT target wavefront RMS")
    ax.set_ylabel("CV of |aligned recovered Seidel coefficients|")
    ax.set_xticks([0.06, 0.20, 0.40])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_rms_vs_cv(rows: list[dict[str, object]], out_path: Path) -> None:
    selected = [
        row
        for row in rows
        if row["coeff_kind"] == "aligned_recovered" and row["method"] != "GT_reference"
    ]
    colors = {
        0.06: "#1f77b4",
        0.20: "#ff7f0e",
        0.40: "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for rms in [0.06, 0.20, 0.40]:
        part = [row for row in selected if abs(float(row["target_wavefront_rms"]) - rms) < 1e-8]
        ax.scatter(
            [float(row["wavefront_over_gt_rms"]) for row in part],
            [float(row["abs_cv_all6"]) for row in part],
            color=colors[rms],
            label=f"RMS {rms:g}",
            alpha=0.75,
            s=55,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_title("Equalization vs recovered/GT wavefront RMS")
    ax.set_xlabel("aligned recovered / GT wavefront RMS")
    ax.set_ylabel("CV of |aligned recovered coefficients|")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_example_coefficients(rows: list[dict[str, object]], out_path: Path) -> None:
    example_images = ["Iksung_beads", "dendrites", "dendrites_dense"]
    example_rms = [0.20, 0.40]
    methods = [
        "GT_reference",
        "ratio1_lam1000",
        "w040_fixed_gt_lam1000",
        "coeflr_w0401e3_other1e4_lam1000",
        "coeflr_w0402e3_other2e4_lam1000",
    ]
    labels = {
        "GT_reference": "GT",
        "ratio1_lam1000": "ratio target",
        "w040_fixed_gt_lam1000": "W040 fixed",
        "coeflr_w0401e3_other1e4_lam1000": "coefLR 1e-3/1e-4",
        "coeflr_w0402e3_other2e4_lam1000": "coefLR 2e-3/2e-4",
    }

    fig, axes = plt.subplots(len(example_rms), len(example_images), figsize=(16, 7.4), sharey=False)
    width = 0.15
    x = np.arange(len(COEFF_LABELS), dtype=np.float64)

    for row_i, rms in enumerate(example_rms):
        for col_i, image in enumerate(example_images):
            ax = axes[row_i, col_i]
            for method_i, method in enumerate(methods):
                coeff_kind = "GT" if method == "GT_reference" else "aligned_recovered"
                matches = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["coeff_kind"] == coeff_kind
                    and row["image"] == image
                    and abs(float(row["target_wavefront_rms"]) - rms) < 1e-8
                ]
                if not matches:
                    continue
                coeffs = np.abs(np.asarray(matches[0]["coeffs"], dtype=np.float64))
                ax.bar(x + (method_i - 2) * width, coeffs, width=width, label=labels[method])
            ax.set_title(f"{image} | RMS {rms:g}", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(COEFF_LABELS, rotation=35, ha="right")
            ax.grid(axis="y", alpha=0.2)
            if col_i == 0:
                ax.set_ylabel("|coefficient|")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("GT vs recovered coefficient magnitudes", y=0.985, fontsize=14)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_markdown_report(summary: list[dict[str, object]], out_path: Path) -> None:
    lookup = {
        (row["method"], row["coeff_kind"], float(row["target_wavefront_rms"])): row
        for row in summary
    }
    methods = [
        "GT_reference",
        "ratio1_lam1000",
        "ratio1_lam10000",
        "w040_fixed_gt_lam1000",
        "coeflr_w0401e3_other1e4_lam1000",
        "coeflr_w0402e3_other2e4_lam1000",
    ]
    lines = [
        "# Seidel Coefficient Equalization Analysis",
        "",
        "RMS is computed from the centered Seidel wavefront over pupil samples, then averaged over field samples with field-height weights. It is not the RMS of the six coefficient values.",
        "",
        "`abs_cv_all6` is the coefficient-magnitude coefficient of variation. Lower values mean the six absolute coefficient magnitudes are more equal.",
        "",
        "| method | RMS | CV(|coeff|) | recovered/GT wavefront RMS | operator_error_calibrated | SSIM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        coeff_kind = "GT" if method == "GT_reference" else "aligned_recovered"
        for rms in [0.06, 0.20, 0.40]:
            row = lookup.get((method, coeff_kind, rms))
            if row is None:
                continue
            op = float(row["operator_error_calibrated_mean"])
            ssim = float(row["ssim_recon_gain_vs_gt_mean"])
            if method == "GT_reference":
                op_text = ""
                ssim_text = ""
            else:
                op_text = f"{op:.4f}"
                ssim_text = f"{ssim:.4f}"
            lines.append(
                f"| {method} | {rms:.2f} | {float(row['abs_cv_all6_mean']):.3f} | "
                f"{float(row['wavefront_over_gt_rms_mean']):.3f} | {op_text} | {ssim_text} |"
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- GT signed_balanced coefficients are not equal-magnitude; their CV is substantially higher than most ratio-target recovered coefficients.",
            "- Ratio-target recovered coefficients have much lower CV, especially at RMS 0.06 and 0.20, which verifies the equal-magnitude tendency.",
            "- For high RMS, some methods become both low-CV and low recovered/GT RMS, indicating an equal-small coefficient collapse rather than a correct Seidel recovery.",
            "- W040 fixed to GT largely removes this ambiguity because one coefficient scale is anchored.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--method-csv",
        action="append",
        default=[],
        help="METHOD=PATH entries. Methods may include all rows in the CSV.",
    )
    args = parser.parse_args()

    if not args.method_csv:
        raise SystemExit("At least one --method-csv METHOD=PATH is required.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    gt_added: set[tuple[str, float]] = set()
    for entry in args.method_csv:
        if "=" not in entry:
            raise SystemExit(f"Bad --method-csv entry: {entry!r}")
        method, path_text = entry.split("=", 1)
        method_rows = load_method_rows(method, Path(path_text))

        filtered: list[dict[str, object]] = []
        for row in method_rows:
            lam = float(row["lambda"])
            if method == "ratio1_lam1000" and abs(lam - 1000.0) > 1e-8:
                continue
            if method == "ratio1_lam10000" and abs(lam - 10000.0) > 1e-8:
                continue
            if method not in {"ratio1_lam1000", "ratio1_lam10000"} and abs(lam - 1000.0) > 1e-8:
                continue
            if float(row["target_wavefront_rms"]) not in {0.06, 0.2, 0.4}:
                continue
            filtered.append(row)

        for row in filtered:
            if row["coeff_kind"] == "GT":
                key = (str(row["image"]), float(row["target_wavefront_rms"]))
                if key in gt_added:
                    continue
                gt_added.add(key)
                gt_row = dict(row)
                gt_row["method"] = "GT_reference"
                rows.append(gt_row)
            elif row["coeff_kind"] == "aligned_recovered":
                rows.append(row)

    write_case_rows(rows, args.out_dir / "coefficient_equalization_case_metrics.csv")
    summary = summarize(rows)
    write_summary(summary, args.out_dir / "coefficient_equalization_summary_by_method_rms.csv")
    plot_cv_summary(summary, args.out_dir / "01_abs_coeff_cv_by_method_rms.png")
    plot_rms_vs_cv(rows, args.out_dir / "02_recovered_gt_rms_vs_abs_coeff_cv.png")
    plot_example_coefficients(rows, args.out_dir / "03_example_abs_coeff_bars.png")
    write_markdown_report(summary, args.out_dir / "README_coeff_equalization.md")
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
