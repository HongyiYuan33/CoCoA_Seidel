#!/usr/bin/env python3
"""Build RCP panels comparing direct no-RMS Seidel dimensionality controls."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_lambda0_vs_10000_rcp_pairs import (  # noqa: E402
    build_case,
    coeff_ylim,
    collect_display_ranges,
    display_path,
    draw_coeff_card,
    draw_image_panel,
    field_weighted_wavefront_rms,
    parse_float,
    parse_vector,
    rms_label,
    safe_name,
)


ROW_SPECS = [
    ("6d", "6D no RMS | direct coefficient recovery"),
    ("5d_no_defocus", "5D no RMS | no defocus"),
    ("4d_no_w311_no_defocus", "4D no RMS | no W311 + no defocus"),
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def seed_value(row: dict[str, Any]) -> int:
    return int(float(row.get("seed", 0)))


def coeff_rms(coeffs: np.ndarray) -> float:
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)[:6]
    return float(np.sqrt(np.mean(coeffs * coeffs)))


def coeff_abs_cv(coeffs: np.ndarray) -> float:
    values = np.abs(np.asarray(coeffs, dtype=np.float64).reshape(-1)[:6])
    mean = float(np.mean(values))
    if mean <= 1e-12:
        return math.nan
    return float(np.std(values, ddof=0) / mean)


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, float, int], dict[str, Any]]:
    indexed: dict[tuple[str, str, float, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["image"]),
            str(row["direction"]),
            round(float(row["target_wavefront_rms"]), 6),
            seed_value(row),
        )
        indexed[key] = row
    return indexed


def label_for_case(row: dict[str, Any], case: dict[str, Any], base_label: str) -> str:
    aligned = parse_vector(row["aligned_seidel_physical"])
    gt = parse_vector(row["seidel_gt"])
    wf_ratio = case["aligned_rms"] / max(case["gt_rms"], 1e-12)
    coeff_ratio = coeff_rms(aligned) / max(coeff_rms(gt), 1e-12)
    op = parse_float(row, "operator_error_calibrated")
    ssim = parse_float(row, "ssim_recon_gain_vs_gt")
    nrmse = parse_float(row, "nrmse_recon_gain_vs_gt")
    return (
        f"{base_label} | seed={seed_value(row)} | op={op:.4f} | "
        f"wf={wf_ratio:.3f}x | coeff={coeff_ratio:.3f}x | "
        f"CV={coeff_abs_cv(aligned):.3f} | SSIM={ssim:.4f} | NRMSE={nrmse:.4f}"
    )


def case_metrics(row: dict[str, Any], case: dict[str, Any], prefix: str) -> dict[str, Any]:
    aligned = parse_vector(row["aligned_seidel_physical"])
    raw = parse_vector(row["seidel_final"])
    gt = parse_vector(row["seidel_gt"])
    return {
        f"{prefix}_operator_error_calibrated": parse_float(row, "operator_error_calibrated"),
        f"{prefix}_operator_error_phys_equiv": parse_float(row, "operator_error_phys_equiv"),
        f"{prefix}_aligned_wavefront_over_gt": case["aligned_rms"] / max(case["gt_rms"], 1e-12),
        f"{prefix}_raw_wavefront_over_gt": field_weighted_wavefront_rms(raw) / max(case["gt_rms"], 1e-12),
        f"{prefix}_aligned_coeff_over_gt": coeff_rms(aligned) / max(coeff_rms(gt), 1e-12),
        f"{prefix}_abs_coeff_cv": coeff_abs_cv(aligned),
        f"{prefix}_ssim": parse_float(row, "ssim_recon_gain_vs_gt"),
        f"{prefix}_nrmse": parse_float(row, "nrmse_recon_gain_vs_gt"),
        f"{prefix}_w311_aligned": float(aligned[4]),
        f"{prefix}_wd_aligned": float(aligned[5]),
        f"{prefix}_candidate_id": row.get("candidate_id", ""),
        f"{prefix}_run_root": row.get("run_root", ""),
    }


def make_three_row_panel(
    *,
    rows_by_name: dict[str, dict[str, Any]],
    output_root: Path,
    out_path: Path,
    title_prefix: str,
) -> dict[str, Any]:
    cases = {name: build_case(row, output_root) for name, row in rows_by_name.items()}
    case_list = [cases[name] for name, _label in ROW_SPECS]
    ranges = collect_display_ranges(case_list)
    ylimits = coeff_ylim(case_list)

    ref = rows_by_name["6d"]
    image = str(ref["image"])
    direction = str(ref["direction"])
    target = parse_float(ref, "target_wavefront_rms")
    seed = seed_value(ref)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20.0, 13.8), dpi=150)
    outer = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.16, 1.0],
        height_ratios=[1, 1, 1],
        left=0.027,
        right=0.985,
        top=0.94,
        bottom=0.045,
        wspace=0.045,
        hspace=0.24,
    )
    title = f"{title_prefix} | seed {seed} | {image} | {direction} | GT wavefront RMS {target:.2f}"
    fig.suptitle(title, fontsize=15.5, fontweight="bold", y=0.982)

    side_y = [0.78, 0.50, 0.22]
    for row_idx, (key, label) in enumerate(ROW_SPECS):
        draw_image_panel(fig, outer[row_idx, 0], cases[key], ranges)
        draw_coeff_card(fig, outer[row_idx, 1], cases[key], ylimits, lambda_label=label_for_case(rows_by_name[key], cases[key], label))
        fig.text(0.013, side_y[row_idx], label, rotation=90, ha="center", va="center", fontsize=10.3, fontweight="bold")

    fig.savefig(out_path)
    plt.close(fig)

    manifest = {
        "image": image,
        "direction": direction,
        "target_wavefront_rms": target,
        "seed": seed,
        "path": display_path(out_path),
    }
    for key, _label in ROW_SPECS:
        manifest.update(case_metrics(rows_by_name[key], cases[key], key))
    return manifest


def make_contact_sheets(manifest_rows: list[dict[str, Any]], out_dir: Path) -> None:
    by_rms: dict[str, list[dict[str, Any]]] = {}
    for row in manifest_rows:
        by_rms.setdefault(rms_label(float(row["target_wavefront_rms"])), []).append(row)

    overview_dir = out_dir / "00_contact_sheets_by_rms"
    overview_dir.mkdir(parents=True, exist_ok=True)
    resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    for label, rows in sorted(by_rms.items()):
        thumbs = []
        for row in sorted(rows, key=lambda item: int(item["seed"])):
            path = PROJECT_ROOT / str(row["path"])
            im = Image.open(path).convert("RGB")
            target_width = 900
            target_height = int(round(im.height * target_width / im.width))
            thumbs.append(im.resize((target_width, target_height), resample_filter))
        if not thumbs:
            continue
        cols = 2
        gap = 24
        rows_n = int(math.ceil(len(thumbs) / cols))
        cell_w = max(im.width for im in thumbs)
        cell_h = max(im.height for im in thumbs)
        canvas = Image.new(
            "RGB",
            (cols * cell_w + (cols - 1) * gap, rows_n * cell_h + (rows_n - 1) * gap),
            "white",
        )
        for idx, im in enumerate(thumbs):
            x = (idx % cols) * (cell_w + gap)
            y = (idx // cols) * (cell_h + gap)
            canvas.paste(im, (x, y))
        canvas.save(overview_dir / f"{label}_contact_sheet.png")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    metrics = [
        ("operator_error_calibrated", "operator_error_calibrated"),
        ("aligned_wavefront_over_gt", "aligned_wavefront_over_gt"),
        ("aligned_coeff_over_gt", "aligned_coeff_over_gt"),
        ("abs_coeff_cv", "abs_coeff_cv"),
        ("ssim", "ssim"),
        ("nrmse", "nrmse"),
    ]
    for target in sorted({float(row["target_wavefront_rms"]) for row in manifest_rows}):
        target_rows = [row for row in manifest_rows if math.isclose(float(row["target_wavefront_rms"]), target)]
        for key, label in ROW_SPECS:
            out: dict[str, Any] = {
                "dimension_control": key,
                "label": label,
                "target_wavefront_rms": target,
                "num_seeds": len(target_rows),
            }
            for suffix, metric_name in metrics:
                values = np.asarray([float(row[f"{key}_{suffix}"]) for row in target_rows], dtype=np.float64)
                out[f"{metric_name}_mean"] = float(np.nanmean(values))
                out[f"{metric_name}_std"] = float(np.nanstd(values, ddof=0))
            summary.append(out)
    return summary


def plot_summary(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not summary_rows:
        return
    labels = {key: label for key, label in ROW_SPECS}
    colors = {
        "6d": "#1f77b4",
        "5d_no_defocus": "#2ca02c",
        "4d_no_w311_no_defocus": "#d62728",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), dpi=160)
    for key, _label in ROW_SPECS:
        rows = [row for row in summary_rows if row["dimension_control"] == key]
        rows = sorted(rows, key=lambda row: float(row["target_wavefront_rms"]))
        x = np.asarray([float(row["target_wavefront_rms"]) for row in rows])
        op = np.asarray([float(row["operator_error_calibrated_mean"]) for row in rows])
        op_std = np.asarray([float(row["operator_error_calibrated_std"]) for row in rows])
        wf = np.asarray([float(row["aligned_wavefront_over_gt_mean"]) for row in rows])
        wf_std = np.asarray([float(row["aligned_wavefront_over_gt_std"]) for row in rows])
        axes[0].errorbar(x, op, yerr=op_std, marker="o", linewidth=2.0, capsize=3, label=labels[key], color=colors[key])
        axes[1].errorbar(x, wf, yerr=wf_std, marker="o", linewidth=2.0, capsize=3, label=labels[key], color=colors[key])

    axes[0].set_title("Operator error by Seidel dimension")
    axes[0].set_xlabel("GT wavefront RMS level")
    axes[0].set_ylabel("operator_error_calibrated")
    axes[0].grid(alpha=0.24)
    axes[1].set_title("Recovered Seidel strength")
    axes[1].set_xlabel("GT wavefront RMS level")
    axes[1].set_ylabel("aligned recovered / GT wavefront RMS")
    axes[1].axhline(1.0, color="0.25", linewidth=1.0, linestyle="--")
    axes[1].grid(alpha=0.24)
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Direct no-RMS coefficient recovery: 6D vs no-defocus controls", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.93))
    fig.savefig(out_dir / "no_RMS_dimension_control_summary.png")
    plt.close(fig)


def write_readme(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Direct No-RMS Dimension-Control RCP",
        "",
        f"6D evaluator CSV: `{args.csv_6d}`",
        f"5D no-defocus evaluator CSV: `{args.csv_5d}`",
        f"4D no-W311/no-defocus evaluator CSV: `{args.csv_4d}`",
        "",
        "Each PNG controls image, direction, GT RMS level, seed, training iterations, object/model params, and no-RMS direct coefficient recovery.",
        "",
        "Rows:",
        "- 6D no RMS | direct coefficient recovery",
        "- 5D no RMS | no defocus",
        "- 4D no RMS | no W311 + no defocus",
        "",
        "Summary files:",
        "- `summary_by_dimension_and_rms.csv`",
        "- `00_stats/no_RMS_dimension_control_summary.png`",
        "- `00_contact_sheets_by_rms/rms*_contact_sheet.png`",
        "",
        f"Generated RCP files: {len(rows)}",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['path']}`")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-6d", type=Path, required=True)
    parser.add_argument("--csv-5d", type=Path, required=True)
    parser.add_argument("--csv-4d", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cocoa_like_2d_mechanism"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title-prefix", default="Controlled RCP: direct no-RMS dimensionality")
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()

    indexed = {
        "6d": index_rows(load_rows(args.csv_6d)),
        "5d_no_defocus": index_rows(load_rows(args.csv_5d)),
        "4d_no_w311_no_defocus": index_rows(load_rows(args.csv_4d)),
    }
    all_keys = sorted(
        set(indexed["6d"]) | set(indexed["5d_no_defocus"]) | set(indexed["4d_no_w311_no_defocus"]),
        key=lambda item: (item[2], item[0], item[1], item[3]),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for image, direction, target, seed in all_keys:
        rows_by_name = {key: table.get((image, direction, target, seed)) for key, table in indexed.items()}
        if any(row is None for row in rows_by_name.values()):
            missing.append(f"{image} {direction} rms={target:.2f} seed={seed}")
            continue
        label = rms_label(target)
        out_path = (
            args.out_dir
            / label
            / safe_name(image)
            / f"seed{seed}"
            / (
                "direct_no_RMS_6D_vs_5D_no_defocus_vs_4D_no_W311_no_defocus__"
                f"{safe_name(image)}__{safe_name(direction)}__{label}__seed{seed}__RCP_vertical.png"
            )
        )
        manifest_rows.append(
            make_three_row_panel(
                rows_by_name=rows_by_name,  # type: ignore[arg-type]
                output_root=args.output_root,
                out_path=out_path,
                title_prefix=args.title_prefix,
            )
        )

    write_csv(manifest_rows, args.out_dir / "manifest.csv")
    summary_rows = build_summary(manifest_rows)
    write_csv(summary_rows, args.out_dir / "summary_by_dimension_and_rms.csv")
    plot_summary(summary_rows, args.out_dir / "00_stats")
    write_readme(manifest_rows, args.out_dir, args)
    if not args.no_contact_sheets:
        make_contact_sheets(manifest_rows, args.out_dir)
    if missing:
        (args.out_dir / "missing_cases.txt").write_text("\n".join(missing) + "\n")
    print(f"[done] wrote {len(manifest_rows)} dimension-control RCP panels to {args.out_dir}")
    if missing:
        print(f"[warn] missing {len(missing)} complete groups; see {args.out_dir / 'missing_cases.txt'}")


if __name__ == "__main__":
    main()
