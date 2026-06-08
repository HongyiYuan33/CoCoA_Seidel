#!/usr/bin/env python3
"""Build three-row RCP panels for 4D/6D oracle-control sweeps."""

from __future__ import annotations

import argparse
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
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_lambda0_vs_10000_rcp_pairs import (  # noqa: E402
    COEFF_LABELS,
    build_case,
    coeff_ylim,
    collect_display_ranges,
    display_path,
    draw_coeff_card,
    draw_image_panel,
    parse_float,
    parse_vector,
    rms_label,
    safe_name,
    short_float,
)


IMAGE_ORDER = ["Test_figure_1", "Iksung_beads", "dendrites", "dendrites_dense"]
DIRECTION_ORDER = ["cocoa_signed", "signed_balanced"]
DIMENSION_ORDER = ["classical4d", "classical6d"]
MODE_ORDER = ["joint_no_RMS", "seidel_gt_fixed", "object_gt_fixed"]
MODE_LABELS = {
    "joint_no_RMS": "joint no-RMS | direct",
    "seidel_gt_fixed": "Seidel fixed GT",
    "object_gt_fixed": "object fixed GT",
}


def load_eval_rows(csv_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    return rows


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
        writer.writerows(rows)


def rank_lookup(value: str, order: list[str]) -> tuple[int, str]:
    return (order.index(value), value) if value in order else (len(order), value)


def grouped_keys(rows: list[dict[str, Any]]) -> list[tuple[str, int, str, str, float]]:
    keys = {
        (
            str(row["seidel_convention"]),
            int(float(row.get("seed", 0) or 0)),
            str(row["image"]),
            str(row["direction"]),
            round(float(row["target_wavefront_rms"]), 6),
        )
        for row in rows
    }
    return sorted(
        keys,
        key=lambda item: (
            rank_lookup(item[0], DIMENSION_ORDER),
            item[1],
            rank_lookup(item[2], IMAGE_ORDER),
            rank_lookup(item[3], DIRECTION_ORDER),
            item[4],
        ),
    )


def make_oracle_panel(
    *,
    cases: list[dict[str, Any]],
    out_path: Path,
    title_prefix: str,
) -> dict[str, Any]:
    ranges = collect_display_ranges(cases)
    ylimits = coeff_ylim(cases)
    row = cases[0]["row"]
    target = parse_float(row, "target_wavefront_rms")
    dimension = str(row.get("seidel_convention", row.get("dimension", "?")))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20.0, 13.2), dpi=150)
    outer = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.16, 1.0],
        height_ratios=[1, 1, 1],
        left=0.025,
        right=0.985,
        top=0.93,
        bottom=0.045,
        wspace=0.045,
        hspace=0.23,
    )
    title = (
        f"{title_prefix} | {dimension} | {row['image']} | {row['direction']} | "
        f"GT RMS {target:.2f}"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.982)

    for idx, case in enumerate(cases):
        mode = str(case["row"].get("oracle_mode", case["row"].get("mode", "?")))
        label = MODE_LABELS.get(mode, mode)
        draw_image_panel(fig, outer[idx, 0], case, ranges)
        draw_coeff_card(fig, outer[idx, 1], case, ylimits, lambda_label=label)
        y = 0.805 - idx * 0.295
        fig.text(0.012, y, label, rotation=90, ha="center", va="center", fontsize=11, fontweight="bold")

    fig.savefig(out_path)
    plt.close(fig)

    manifest: dict[str, Any] = {
        "seidel_convention": dimension,
        "dimension": row.get("dimension", "4D" if dimension == "classical4d" else "6D"),
        "seed": int(float(row.get("seed", 0) or 0)),
        "image": row["image"],
        "direction": row["direction"],
        "target_wavefront_rms": target,
        "path": display_path(out_path),
    }
    for case in cases:
        mode = str(case["row"].get("oracle_mode", "?"))
        prefix = mode.replace("-", "_")
        manifest[f"{prefix}_operator_error_calibrated"] = parse_float(case["row"], "operator_error_calibrated")
        manifest[f"{prefix}_recovered_over_gt_rms"] = parse_float(case["row"], "wavefront_recovered_over_gt_rms")
        manifest[f"{prefix}_ssim"] = parse_float(case["row"], "ssim_recon_gain_vs_gt")
        manifest[f"{prefix}_nrmse"] = parse_float(case["row"], "nrmse_recon_gain_vs_gt")
    return manifest


def coefficient_shape_stats(row: dict[str, Any]) -> dict[str, float]:
    key = "aligned_seidel_physical" if row.get("aligned_seidel_physical") not in (None, "") else "seidel_final"
    try:
        coeff = parse_vector(row[key])
    except Exception:
        coeff = parse_vector(row["seidel_final"])
    abs_coeff = np.abs(coeff)
    mean_abs = float(np.mean(abs_coeff))
    cv = float(np.std(abs_coeff) / max(mean_abs, 1e-12))
    share_w222 = float(abs_coeff[2] / max(float(np.sum(abs_coeff)), 1e-12))
    return {"coeff_abs_cv": cv, "w222_abs_share": share_w222}


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["seidel_convention"]),
            str(row["oracle_mode"]),
            str(row["direction"]),
            round(float(row["target_wavefront_rms"]), 6),
        )
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for (dimension, oracle_mode, direction, target), group in grouped.items():
        stats = [coefficient_shape_stats(row) for row in group]
        out.append(
            {
                "seidel_convention": dimension,
                "dimension": "4D" if dimension == "classical4d" else "6D",
                "oracle_mode": oracle_mode,
                "direction": direction,
                "target_wavefront_rms": target,
                "num_rows": len(group),
                "mean_operator_error_calibrated": float(
                    np.mean([parse_float(row, "operator_error_calibrated") for row in group])
                ),
                "median_operator_error_calibrated": float(
                    np.median([parse_float(row, "operator_error_calibrated") for row in group])
                ),
                "mean_recovered_over_gt_rms": float(
                    np.mean([parse_float(row, "wavefront_recovered_over_gt_rms") for row in group])
                ),
                "mean_coeff_recovered_over_gt_rms": float(
                    np.mean([parse_float(row, "coeff_recovered_over_gt_rms") for row in group])
                ),
                "mean_ssim": float(np.mean([parse_float(row, "ssim_recon_gain_vs_gt") for row in group])),
                "mean_nrmse": float(np.mean([parse_float(row, "nrmse_recon_gain_vs_gt") for row in group])),
                "mean_coeff_abs_cv": float(np.mean([stat["coeff_abs_cv"] for stat in stats])),
                "mean_w222_abs_share": float(np.mean([stat["w222_abs_share"] for stat in stats])),
            }
        )
    return sorted(
        out,
        key=lambda row: (
            rank_lookup(str(row["seidel_convention"]), DIMENSION_ORDER),
            rank_lookup(str(row["direction"]), DIRECTION_ORDER),
            float(row["target_wavefront_rms"]),
            rank_lookup(str(row["oracle_mode"]), MODE_ORDER),
        ),
    )


def plot_summary(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not summary_rows:
        return
    plot_specs = [
        ("mean_operator_error_calibrated", "operator_error_calibrated", "operator_error_calibrated_by_rms.png"),
        ("mean_recovered_over_gt_rms", "recovered / GT wavefront RMS", "recovered_over_gt_wavefront_rms_by_rms.png"),
        ("mean_coeff_abs_cv", "CV of |aligned coefficients|", "coeff_abs_cv_by_rms.png"),
        ("mean_w222_abs_share", "|W222| share", "w222_abs_share_by_rms.png"),
        ("mean_ssim", "SSIM gain vs GT", "ssim_by_rms.png"),
        ("mean_nrmse", "NRMSE gain vs GT", "nrmse_by_rms.png"),
    ]
    colors = {
        "joint_no_RMS": "#1f77b4",
        "seidel_gt_fixed": "#2ca02c",
        "object_gt_fixed": "#d62728",
    }
    for key, ylabel, filename in plot_specs:
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), squeeze=False)
        for r_idx, dimension in enumerate(DIMENSION_ORDER):
            for c_idx, direction in enumerate(DIRECTION_ORDER):
                ax = axes[r_idx, c_idx]
                subset = [
                    row
                    for row in summary_rows
                    if row["seidel_convention"] == dimension and row["direction"] == direction
                ]
                for mode in MODE_ORDER:
                    mode_rows = sorted(
                        [row for row in subset if row["oracle_mode"] == mode],
                        key=lambda row: float(row["target_wavefront_rms"]),
                    )
                    if not mode_rows:
                        continue
                    x = [float(row["target_wavefront_rms"]) for row in mode_rows]
                    y = [float(row[key]) for row in mode_rows]
                    ax.plot(
                        x,
                        y,
                        marker="o",
                        linewidth=2.0,
                        color=colors.get(mode),
                        label=MODE_LABELS.get(mode, mode),
                    )
                    for xx, yy in zip(x, y):
                        ax.text(xx, yy, short_float(yy, 3), fontsize=7, ha="center", va="bottom")
                ax.set_title(f"{dimension} | {direction}", fontsize=10, fontweight="bold")
                ax.set_xlabel("GT Seidel wavefront RMS")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.25)
                if key == "mean_recovered_over_gt_rms":
                    ax.axhline(1.0, color="0.35", linewidth=1.0, linestyle="--")
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)


def make_contact_sheets(manifest_rows: list[dict[str, Any]], out_dir: Path) -> None:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        by_group[(str(row["seidel_convention"]), rms_label(float(row["target_wavefront_rms"])))].append(row)

    contact_dir = out_dir / "00_contact_sheets_by_dimension_rms"
    contact_dir.mkdir(parents=True, exist_ok=True)
    resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    for (dimension, rms), rows in sorted(by_group.items()):
        images: list[Image.Image] = []
        for row in sorted(
            rows,
            key=lambda item: (
                rank_lookup(str(item["image"]), IMAGE_ORDER),
                rank_lookup(str(item["direction"]), DIRECTION_ORDER),
            ),
        ):
            path = PROJECT_ROOT / str(row["path"])
            im = Image.open(path).convert("RGB")
            target_width = 860
            target_height = int(round(im.height * target_width / im.width))
            images.append(im.resize((target_width, target_height), resample_filter))
        if not images:
            continue
        cols = 2
        gap = 24
        rows_n = int(math.ceil(len(images) / cols))
        cell_w = max(im.width for im in images)
        cell_h = max(im.height for im in images)
        canvas = Image.new(
            "RGB",
            (cols * cell_w + (cols - 1) * gap, rows_n * cell_h + (rows_n - 1) * gap),
            "white",
        )
        for idx, im in enumerate(images):
            x = (idx % cols) * (cell_w + gap)
            y = (idx // cols) * (cell_h + gap)
            canvas.paste(im, (x, y))
        canvas.save(contact_dir / f"{dimension}__{rms}__contact_sheet.png")


def write_readme(out_dir: Path, args: argparse.Namespace, manifest_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 4D + 6D Oracle Controls RCP",
        "",
        "Rows in every RCP:",
        "- top: joint no-RMS direct coefficient recovery",
        "- middle: Seidel fixed exactly to GT, object recovered",
        "- bottom: object fixed to sharp GT, Seidel recovered",
        "",
        "Folder layout:",
        "- `<dimension>/<image>/<direction>/<rms>/...RCP_vertical.png`",
        "- `00_contact_sheets_by_dimension_rms/`",
        "- `summary_by_dimension_mode_direction_rms.csv`",
        "",
        "Input evaluator CSVs:",
    ]
    for csv_path in args.csv:
        lines.append(f"- `{csv_path}`")
    lines.extend(["", f"Generated RCP files: {len(manifest_rows)}", ""])
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cocoa_like_2d_mechanism"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--combined-csv", type=Path, default=None)
    parser.add_argument("--title-prefix", default="4D + 6D oracle controls")
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()

    eval_rows = load_eval_rows(args.csv)
    if args.combined_csv is not None:
        write_csv(eval_rows, args.combined_csv)

    row_by_key: dict[tuple[str, int, str, str, float, str], dict[str, Any]] = {}
    for row in eval_rows:
        key = (
            str(row["seidel_convention"]),
            int(float(row.get("seed", 0) or 0)),
            str(row["image"]),
            str(row["direction"]),
            round(float(row["target_wavefront_rms"]), 6),
            str(row["oracle_mode"]),
        )
        row_by_key[key] = row

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for dimension, seed, image, direction, target in grouped_keys(eval_rows):
        cases = []
        for mode in MODE_ORDER:
            row = row_by_key.get((dimension, seed, image, direction, target, mode))
            if row is None:
                missing.append(f"{dimension} seed{seed} {image} {direction} rms{target:.2f} {mode}")
                continue
            cases.append(build_case(row, args.output_root))
        if len(cases) != len(MODE_ORDER):
            continue
        label = rms_label(target)
        out_path = (
            args.out_dir
            / safe_name(dimension)
            / safe_name(image)
            / safe_name(direction)
            / label
            / (
                f"oracle_controls__{safe_name(dimension)}__{safe_name(image)}__"
                f"{safe_name(direction)}__seed{seed}__{label}__RCP_vertical.png"
            )
        )
        manifest_rows.append(make_oracle_panel(cases=cases, out_path=out_path, title_prefix=args.title_prefix))

    write_csv(manifest_rows, args.out_dir / "manifest.csv")
    summary_rows = summarize_rows(eval_rows)
    write_csv(summary_rows, args.out_dir / "summary_by_dimension_mode_direction_rms.csv")
    plot_summary(summary_rows, args.out_dir)
    if not args.no_contact_sheets:
        make_contact_sheets(manifest_rows, args.out_dir)
    write_readme(args.out_dir, args, manifest_rows)
    if missing:
        (args.out_dir / "missing_cases.txt").write_text("\n".join(missing) + "\n")
    print(f"[done] wrote {len(manifest_rows)} oracle RCP files to {args.out_dir}")
    if missing:
        print(f"[warn] missing {len(missing)} rows; see {args.out_dir / 'missing_cases.txt'}")


if __name__ == "__main__":
    main()
