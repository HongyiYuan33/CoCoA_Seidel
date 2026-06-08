#!/usr/bin/env python3
"""Build RCP panels comparing wavefront-RMS and coefficient-RMS priors."""

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
    parse_float,
    parse_vector,
    rms_label,
    safe_name,
)


MEASURE_LABELS = {
    "wavefront": "wavefront RMS prior",
    "coefficient": "coefficient RMS prior",
}


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


def w222_abs_share(coeffs: np.ndarray) -> float:
    values = np.abs(np.asarray(coeffs, dtype=np.float64).reshape(-1)[:6])
    denom = float(np.sum(values))
    if denom <= 1e-12:
        return math.nan
    return float(values[2] / denom)


def indexed_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, float, int, str], dict[str, Any]]:
    out: dict[tuple[str, str, float, int, str], dict[str, Any]] = {}
    for row in rows:
        measure = str(row.get("seidel_rms_prior_measure") or "wavefront")
        if measure not in MEASURE_LABELS:
            continue
        key = (
            str(row["image"]),
            str(row["direction"]),
            round(float(row["target_wavefront_rms"]), 6),
            seed_value(row),
            measure,
        )
        out[key] = row
    return out


def label_for_case(row: dict[str, Any], case: dict[str, Any]) -> str:
    measure = str(row.get("seidel_rms_prior_measure") or "wavefront")
    aligned = parse_vector(row["aligned_seidel_physical"])
    gt = parse_vector(row["seidel_gt"])
    wf_ratio = case["aligned_rms"] / max(case["gt_rms"], 1e-12)
    coeff_ratio = coeff_rms(aligned) / max(coeff_rms(gt), 1e-12)
    return (
        f"{MEASURE_LABELS[measure]} | seed={seed_value(row)} | "
        f"lambda={parse_float(row, 'lambda'):.0f} | "
        f"wf={wf_ratio:.3f}x | coeff={coeff_ratio:.3f}x | "
        f"W222={w222_abs_share(aligned):.3f} | CV={coeff_abs_cv(aligned):.3f}"
    )


def make_pair_panel(
    *,
    wavefront_row: dict[str, Any],
    coefficient_row: dict[str, Any],
    output_root: Path,
    out_path: Path,
    title_prefix: str,
) -> dict[str, Any]:
    wavefront_case = build_case(wavefront_row, output_root)
    coefficient_case = build_case(coefficient_row, output_root)
    cases = [wavefront_case, coefficient_case]
    ranges = collect_display_ranges(cases)
    ylimits = coeff_ylim(cases)

    image = str(wavefront_row["image"])
    direction = str(wavefront_row["direction"])
    target = parse_float(wavefront_row, "target_wavefront_rms")
    seed = seed_value(wavefront_row)
    top_label = label_for_case(wavefront_row, wavefront_case)
    bottom_label = label_for_case(coefficient_row, coefficient_case)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20.0, 9.4), dpi=150)
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.16, 1.0],
        height_ratios=[1, 1],
        left=0.027,
        right=0.985,
        top=0.92,
        bottom=0.055,
        wspace=0.045,
        hspace=0.22,
    )
    title = f"{title_prefix} | seed {seed} | {image} | {direction} | 6D | GT wavefront RMS {target:.2f}"
    fig.suptitle(title, fontsize=15.5, fontweight="bold", y=0.978)

    draw_image_panel(fig, outer[0, 0], wavefront_case, ranges)
    draw_coeff_card(fig, outer[0, 1], wavefront_case, ylimits, lambda_label=top_label)
    draw_image_panel(fig, outer[1, 0], coefficient_case, ranges)
    draw_coeff_card(fig, outer[1, 1], coefficient_case, ylimits, lambda_label=bottom_label)

    fig.text(0.013, 0.705, "wavefront RMS prior", rotation=90, ha="center", va="center", fontsize=11, fontweight="bold")
    fig.text(0.013, 0.285, "coefficient RMS prior", rotation=90, ha="center", va="center", fontsize=11, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)

    wf_aligned = parse_vector(wavefront_row["aligned_seidel_physical"])
    coeff_aligned = parse_vector(coefficient_row["aligned_seidel_physical"])
    gt = parse_vector(wavefront_row["seidel_gt"])
    return {
        "image": image,
        "direction": direction,
        "target_wavefront_rms": target,
        "seed": seed,
        "wavefront_operator_error_calibrated": parse_float(wavefront_row, "operator_error_calibrated"),
        "coefficient_operator_error_calibrated": parse_float(coefficient_row, "operator_error_calibrated"),
        "wavefront_aligned_wavefront_over_gt": wavefront_case["aligned_rms"] / max(wavefront_case["gt_rms"], 1e-12),
        "coefficient_aligned_wavefront_over_gt": coefficient_case["aligned_rms"] / max(coefficient_case["gt_rms"], 1e-12),
        "wavefront_aligned_coeff_over_gt": coeff_rms(wf_aligned) / max(coeff_rms(gt), 1e-12),
        "coefficient_aligned_coeff_over_gt": coeff_rms(coeff_aligned) / max(coeff_rms(gt), 1e-12),
        "wavefront_w222_abs_share": w222_abs_share(wf_aligned),
        "coefficient_w222_abs_share": w222_abs_share(coeff_aligned),
        "wavefront_abs_coeff_cv": coeff_abs_cv(wf_aligned),
        "coefficient_abs_coeff_cv": coeff_abs_cv(coeff_aligned),
        "path": display_path(out_path),
    }


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
        canvas = Image.new("RGB", (cols * cell_w + (cols - 1) * gap, rows_n * cell_h + (rows_n - 1) * gap), "white")
        for idx, im in enumerate(thumbs):
            x = (idx % cols) * (cell_w + gap)
            y = (idx // cols) * (cell_h + gap)
            canvas.paste(im, (x, y))
        canvas.save(overview_dir / f"{label}_contact_sheet.png")


def write_manifest(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (out_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(rows: list[dict[str, Any]], out_dir: Path, csv_path: Path) -> None:
    lines = [
        "# Wavefront RMS Prior vs Coefficient RMS Prior RCP",
        "",
        f"Input evaluator CSV: `{csv_path}`",
        "",
        "Each PNG controls image, direction, GT RMS level, lambda, alpha, parameterization, and seed.",
        "Top row is `wavefront RMS prior`; bottom row is `coefficient RMS prior`.",
        "",
        "Folder layout:",
        "- `rms*/dendrites/seed*/wavefront_RMS_prior_vs_coefficient_RMS_prior__...png`",
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
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cocoa_like_2d_mechanism"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title-prefix", default="Controlled RCP: wavefront RMS prior vs coefficient RMS prior")
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    indexed = indexed_rows(rows)
    pair_keys = sorted(
        {
            (image, direction, target, seed)
            for image, direction, target, seed, _measure in indexed
        },
        key=lambda item: (item[2], item[0], item[1], item[3]),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for image, direction, target, seed in pair_keys:
        wf = indexed.get((image, direction, target, seed, "wavefront"))
        coeff = indexed.get((image, direction, target, seed, "coefficient"))
        if wf is None or coeff is None:
            missing.append(f"{image} {direction} rms={target:.2f} seed={seed}")
            continue
        label = rms_label(target)
        out_path = (
            args.out_dir
            / label
            / safe_name(image)
            / f"seed{seed}"
            / (
                "wavefront_RMS_prior_vs_coefficient_RMS_prior__"
                f"{safe_name(image)}__{safe_name(direction)}__{label}__seed{seed}__RCP_vertical.png"
            )
        )
        manifest_rows.append(
            make_pair_panel(
                wavefront_row=wf,
                coefficient_row=coeff,
                output_root=args.output_root,
                out_path=out_path,
                title_prefix=args.title_prefix,
            )
        )

    write_manifest(manifest_rows, args.out_dir)
    write_readme(manifest_rows, args.out_dir, args.csv)
    if not args.no_contact_sheets:
        make_contact_sheets(manifest_rows, args.out_dir)
    if missing:
        (args.out_dir / "missing_cases.txt").write_text("\n".join(missing) + "\n")
    print(f"[done] wrote {len(manifest_rows)} RMS-measure RCP panels to {args.out_dir}")
    if missing:
        print(f"[warn] missing {len(missing)} pairs; see {args.out_dir / 'missing_cases.txt'}")


if __name__ == "__main__":
    main()
