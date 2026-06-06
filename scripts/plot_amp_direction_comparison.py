#!/usr/bin/env python3
"""Plot amplitude-direction Seidel recovery trends against a ratio-target control."""

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

from hybrid_ring_cocoa.evaluation.seidel_operator_evaluator import field_weighted_wavefront_rms


IMAGES = ("Iksung_beads", "dendrites", "dendrites_dense")
RMS_VALUES = (0.06, 0.20, 0.40)
VARIANTS = ("amp_direction", "amp_direction_detach_norm")
COEFF_LABELS = ("W040", "W131", "W222", "W220", "W311", "Wd")


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
        raise ValueError("Missing vector")
    arr = np.asarray(parsed, dtype=np.float64).reshape(-1)
    if arr.size < 6:
        arr = np.pad(arr, (0, 6 - arr.size))
    return arr[:6]


def parse_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value in (None, ""):
        return float(default)
    return float(value)


def abs_coeff_cv(coeffs: np.ndarray) -> float:
    values = np.abs(np.asarray(coeffs, dtype=np.float64).reshape(-1)[:6])
    mean = float(np.mean(values))
    if mean <= 1e-12:
        return 0.0
    return float(np.std(values) / mean)


def load_rows(path: Path, *, method: str, keep_lambda: float | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("image") not in IMAGES:
                continue
            if row.get("direction") != "signed_balanced":
                continue
            rms = round(float(row["target_wavefront_rms"]), 6)
            if rms not in {round(v, 6) for v in RMS_VALUES}:
                continue
            lam = parse_float(row, "lambda", parse_float(row, "seidel_rms_floor_weight"))
            if keep_lambda is not None and not math.isclose(lam, float(keep_lambda), abs_tol=1e-6):
                continue
            gt = parse_vec(row["seidel_gt"])
            raw = parse_vec(row["seidel_final"])
            aligned = parse_vec(row["aligned_seidel_physical"])
            gt_rms = field_weighted_wavefront_rms(gt)
            raw_rms = field_weighted_wavefront_rms(raw)
            aligned_rms = field_weighted_wavefront_rms(aligned)
            parameterization = row.get("seidel_parameterization") or ("direct" if method == "control_ratio1_lam1000" else method)
            rows.append(
                {
                    "method": method,
                    "seidel_parameterization": parameterization,
                    "image": row["image"],
                    "rms": rms,
                    "lambda": lam,
                    "operator_error_calibrated": parse_float(row, "operator_error_calibrated"),
                    "ssim": parse_float(row, "ssim_recon_gain_vs_gt"),
                    "nrmse": parse_float(row, "nrmse_recon_gain_vs_gt"),
                    "raw_over_gt": raw_rms / max(gt_rms, 1e-12),
                    "aligned_over_gt": aligned_rms / max(gt_rms, 1e-12),
                    "abs_coeff_cv": abs_coeff_cv(aligned),
                    "seidel_amplitude_final": parse_float(row, "seidel_amplitude_final"),
                    "seidel_direction_rms_final": parse_float(row, "seidel_direction_rms_final"),
                    "gt_coeffs": gt.tolist(),
                    "aligned_coeffs": aligned.tolist(),
                    "metrics_path": row.get("metrics_path", ""),
                    "run_root": row.get("run_root", ""),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, separators=(",", ":"))
                out[key] = value
            writer.writerow(out)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], float(row["lambda"]), float(row["rms"]))].append(row)

    metrics = [
        "operator_error_calibrated",
        "aligned_over_gt",
        "raw_over_gt",
        "abs_coeff_cv",
        "ssim",
        "nrmse",
        "seidel_amplitude_final",
        "seidel_direction_rms_final",
    ]
    out: list[dict[str, Any]] = []
    for (method, lam, rms), values in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        row: dict[str, Any] = {"method": method, "lambda": lam, "rms": rms, "n": len(values)}
        for metric in metrics:
            arr = np.asarray([float(v[metric]) for v in values if math.isfinite(float(v[metric]))], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else math.nan
            row[f"{metric}_std"] = float(np.std(arr)) if arr.size else math.nan
        out.append(row)
    return out


def select_best(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["method"] not in VARIANTS:
            continue
        groups[(row["method"], row["image"], float(row["rms"]))].append(row)

    selected: list[dict[str, Any]] = []
    for key, values in sorted(groups.items(), key=lambda item: (item[0][2], item[0][1], item[0][0])):
        in_band = [row for row in values if 0.8 <= float(row["aligned_over_gt"]) <= 1.2]
        if in_band:
            best = min(in_band, key=lambda row: float(row["operator_error_calibrated"]))
            reason = "lowest_op_with_aligned_rms_0p8_to_1p2"
        else:
            best = min(
                values,
                key=lambda row: (
                    abs(float(row["aligned_over_gt"]) - 1.0),
                    float(row["operator_error_calibrated"]),
                ),
            )
            reason = "closest_aligned_rms_to_1"
        output = dict(best)
        output["selection_reason"] = reason
        selected.append(output)
    return selected


def plot_lambda_sweep(summary: list[dict[str, Any]], out_dir: Path) -> None:
    colors = {0.06: "#1f77b4", 0.20: "#ff7f0e", 0.40: "#2ca02c"}
    linestyles = {"amp_direction": "-", "amp_direction_detach_norm": "--"}
    labels = {
        "operator_error_calibrated_mean": "operator_error_calibrated",
        "aligned_over_gt_mean": "aligned recovered / GT RMS",
        "abs_coeff_cv_mean": "CV of |aligned coefficients|",
        "ssim_mean": "SSIM gain",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), dpi=180)
    for ax, (metric, ylabel) in zip(axes.ravel(), labels.items()):
        for method in VARIANTS:
            for rms in RMS_VALUES:
                part = [
                    row
                    for row in summary
                    if row["method"] == method and math.isclose(float(row["rms"]), rms, abs_tol=1e-8)
                ]
                if not part:
                    continue
                xs = [float(row["lambda"]) for row in part]
                ys = [float(row[metric]) for row in part]
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=2,
                    linestyle=linestyles[method],
                    color=colors[rms],
                    label=f"{method} RMS {rms:g}",
                )
        ax.set_xscale("log")
        ax.set_xlabel("lambda_a")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Amplitude-direction lambda sweep", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "01_amp_direction_lambda_sweep.png")
    plt.close(fig)


def plot_best_vs_control(control: list[dict[str, Any]], best: list[dict[str, Any]], out_dir: Path) -> None:
    control_by_key = {(row["image"], float(row["rms"])): row for row in control}
    rows: list[dict[str, Any]] = []
    for row in best:
        key = (row["image"], float(row["rms"]))
        base = control_by_key.get(key)
        if base is None:
            continue
        rows.append(
            {
                "method": row["method"],
                "image": row["image"],
                "rms": float(row["rms"]),
                "control_op": float(base["operator_error_calibrated"]),
                "new_op": float(row["operator_error_calibrated"]),
                "control_aligned": float(base["aligned_over_gt"]),
                "new_aligned": float(row["aligned_over_gt"]),
                "control_cv": float(base["abs_coeff_cv"]),
                "new_cv": float(row["abs_coeff_cv"]),
            }
        )
    write_csv(out_dir / "best_delta_vs_control.csv", rows)

    metrics = [
        ("operator_error", "operator error calibrated", "control_op", "new_op"),
        ("aligned_over_gt", "aligned recovered / GT RMS", "control_aligned", "new_aligned"),
        ("abs_coeff_cv", "CV of |aligned coeffs|", "control_cv", "new_cv"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    x = np.arange(len(RMS_VALUES), dtype=np.float64)
    width = 0.22
    for ax, (_, ylabel, base_key, new_key) in zip(axes, metrics):
        base_values = []
        for rms in RMS_VALUES:
            subset = [row for row in rows if math.isclose(row["rms"], rms, abs_tol=1e-8)]
            base_values.append(float(np.mean([row[base_key] for row in subset])) if subset else math.nan)
        ax.bar(x - width, base_values, width=width, label="control ratio1", color="#4c78a8")
        for offset, method, color in [
            (0.0, "amp_direction", "#f58518"),
            (width, "amp_direction_detach_norm", "#54a24b"),
        ]:
            values = []
            for rms in RMS_VALUES:
                subset = [
                    row for row in rows if row["method"] == method and math.isclose(row["rms"], rms, abs_tol=1e-8)
                ]
                values.append(float(np.mean([row[new_key] for row in subset])) if subset else math.nan)
            ax.bar(x + offset, values, width=width, label=method, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{rms:g}" for rms in RMS_VALUES])
        ax.set_xlabel("GT target wavefront RMS")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Best amplitude-direction setting vs existing ratio-target control", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "02_best_vs_control_by_rms.png")
    plt.close(fig)


def write_report(summary: list[dict[str, Any]], best: list[dict[str, Any]], out_dir: Path) -> None:
    lines = [
        "# Amplitude-Direction Seidel Recovery Comparison",
        "",
        "Control is the existing direct ratio-target alpha=1, lambda=1000 run.",
        "New methods sweep lambda_a over 10, 100, 500, 1000, 5000, 10000.",
        "",
        "## Best New Setting Per Image/RMS/Variant",
        "",
        "| variant | image | RMS | lambda_a | op err | aligned/GT RMS | coeff CV | reason |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in best:
        lines.append(
            f"| {row['method']} | {row['image']} | {float(row['rms']):.2f} | "
            f"{float(row['lambda']):.0f} | {float(row['operator_error_calibrated']):.4f} | "
            f"{float(row['aligned_over_gt']):.3f} | {float(row['abs_coeff_cv']):.3f} | "
            f"{row['selection_reason']} |"
        )
    lines += [
        "",
        "## Mean Sweep Table",
        "",
        "| method | lambda_a | RMS | n | op err | aligned/GT RMS | coeff CV | SSIM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {float(row['lambda']):.0f} | {float(row['rms']):.2f} | "
            f"{int(row['n'])} | {float(row['operator_error_calibrated_mean']):.4f} | "
            f"{float(row['aligned_over_gt_mean']):.3f} | {float(row['abs_coeff_cv_mean']):.3f} | "
            f"{float(row['ssim_mean']):.3f} |"
        )
    lines.append("")
    (out_dir / "trend_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-csv", type=Path, required=True)
    parser.add_argument("--new-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    control = load_rows(args.control_csv, method="control_ratio1_lam1000", keep_lambda=1000.0)
    new_rows = []
    for row in load_rows(args.new_csv, method="amp_direction"):
        if row["seidel_parameterization"] in VARIANTS:
            row["method"] = row["seidel_parameterization"]
            new_rows.append(row)
    all_rows = control + new_rows
    summary = summarize(all_rows)
    best = select_best(new_rows)

    write_csv(args.out_dir / "all_rows.csv", all_rows)
    write_csv(args.out_dir / "summary_by_method_lambda_rms.csv", summary)
    write_csv(args.out_dir / "best_new_by_case_variant.csv", best)
    plot_lambda_sweep(summary, args.out_dir)
    plot_best_vs_control(control, best, args.out_dir)
    write_report(summary, best, args.out_dir)
    print(f"[done] wrote amplitude-direction comparison to {args.out_dir}")


if __name__ == "__main__":
    main()
