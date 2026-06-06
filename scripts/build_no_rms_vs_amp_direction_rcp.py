#!/usr/bin/env python3
"""Build controlled RCP panels comparing no-RMS direct recovery to amp-direction variants."""

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


IMAGE_ORDER = ["Iksung_beads", "dendrites", "dendrites_dense"]
VARIANTS = ("amp_direction", "amp_direction_detach_norm")


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def rounded_rms(row: dict[str, Any]) -> float:
    return round(float(row["target_wavefront_rms"]), 6)


def aligned_over_gt_rms(row: dict[str, Any]) -> float:
    gt = parse_vector(row["seidel_gt"])
    aligned = parse_vector(row["aligned_seidel_physical"])
    gt_rms = field_weighted_wavefront_rms(gt)
    return field_weighted_wavefront_rms(aligned) / max(gt_rms, 1e-12)


def coefficient_abs_cv(row: dict[str, Any]) -> float:
    coeff = abs(parse_vector(row["aligned_seidel_physical"]))
    mean = float(coeff.mean())
    if mean <= 1e-12:
        return math.nan
    return float(coeff.std(ddof=0) / mean)


def filter_controlled_rows(
    rows: list[dict[str, Any]],
    *,
    images: set[str],
    direction: str,
    rms_values: set[float],
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if str(row.get("image")) not in images:
            continue
        if str(row.get("direction")) != direction:
            continue
        if rounded_rms(row) not in rms_values:
            continue
        selected.append(row)
    return selected


def index_no_rms_rows(
    rows: list[dict[str, Any]],
    *,
    images: set[str],
    direction: str,
    rms_values: set[float],
    lambda_value: float,
) -> dict[tuple[str, float], dict[str, Any]]:
    indexed: dict[tuple[str, float], dict[str, Any]] = {}
    for row in filter_controlled_rows(rows, images=images, direction=direction, rms_values=rms_values):
        if not math.isclose(parse_float(row, "lambda"), lambda_value, rel_tol=0.0, abs_tol=1e-6):
            continue
        indexed[(str(row["image"]), rounded_rms(row))] = row
    return indexed


def select_best_amp_rows(
    rows: list[dict[str, Any]],
    *,
    images: set[str],
    direction: str,
    rms_values: set[float],
    variants: tuple[str, ...],
    rms_band: tuple[float, float],
) -> dict[tuple[str, str, float], dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in filter_controlled_rows(rows, images=images, direction=direction, rms_values=rms_values):
        variant = str(row.get("seidel_parameterization") or row.get("method"))
        if variant not in variants:
            continue
        grouped.setdefault((variant, str(row["image"]), rounded_rms(row)), []).append(row)

    selected: dict[tuple[str, str, float], dict[str, Any]] = {}
    lo, hi = rms_band
    for key, values in grouped.items():
        in_band = [row for row in values if lo <= aligned_over_gt_rms(row) <= hi]
        if in_band:
            best = min(in_band, key=lambda row: parse_float(row, "operator_error_calibrated"))
            reason = f"lowest_op_with_aligned_rms_{lo:g}_to_{hi:g}"
        else:
            best = min(
                values,
                key=lambda row: (
                    abs(aligned_over_gt_rms(row) - 1.0),
                    parse_float(row, "operator_error_calibrated"),
                ),
            )
            reason = "closest_aligned_rms_to_1"
        out = dict(best)
        out["selection_reason"] = reason
        selected[key] = out
    return selected


def image_rank(image: str) -> tuple[int, str]:
    if image in IMAGE_ORDER:
        return IMAGE_ORDER.index(image), image
    return len(IMAGE_ORDER), image


def sorted_case_keys(images: list[str], rms_values: list[float]) -> list[tuple[str, float]]:
    return sorted(
        [(image, round(float(rms), 6)) for rms in rms_values for image in images],
        key=lambda item: (item[1], image_rank(item[0])),
    )


def make_three_row_panel(
    *,
    cases: list[dict[str, Any]],
    labels: list[str],
    out_path: Path,
    title_prefix: str,
) -> dict[str, Any]:
    ranges = collect_display_ranges(cases)
    ylimits = coeff_ylim(cases)
    row = cases[0]["row"]
    target = parse_float(row, "target_wavefront_rms")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20.0, 13.4), dpi=150)
    outer = fig.add_gridspec(
        len(cases),
        2,
        width_ratios=[1.16, 1.0],
        height_ratios=[1] * len(cases),
        left=0.027,
        right=0.985,
        top=0.925,
        bottom=0.045,
        wspace=0.045,
        hspace=0.215,
    )
    title = f"{title_prefix} | {row['image']} | {row['direction']} | 6D | GT RMS {target:.2f}"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.982)

    label_y = [0.802, 0.505, 0.208]
    for idx, (case, label) in enumerate(zip(cases, labels)):
        draw_image_panel(fig, outer[idx, 0], case, ranges)
        draw_coeff_card(fig, outer[idx, 1], case, ylimits, lambda_label=label)
        fig.text(
            0.013,
            label_y[idx] if idx < len(label_y) else 0.5,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.3,
            fontweight="bold",
        )

    fig.savefig(out_path)
    plt.close(fig)

    output = {
        "image": row["image"],
        "direction": row["direction"],
        "target_wavefront_rms": target,
        "path": display_path(out_path),
    }
    prefixes = ["no_rms", "amp_direction", "amp_direction_detach_norm"]
    for prefix, case in zip(prefixes, cases):
        case_row = case["row"]
        output[f"{prefix}_lambda"] = parse_float(case_row, "lambda")
        output[f"{prefix}_operator_error_calibrated"] = parse_float(case_row, "operator_error_calibrated")
        output[f"{prefix}_aligned_over_gt_rms"] = case["aligned_rms"] / max(case["gt_rms"], 1e-12)
        output[f"{prefix}_raw_over_gt_rms"] = case["raw_rms"] / max(case["gt_rms"], 1e-12)
        output[f"{prefix}_ssim"] = parse_float(case_row, "ssim_recon_gain_vs_gt")
        output[f"{prefix}_nrmse"] = parse_float(case_row, "nrmse_recon_gain_vs_gt")
        output[f"{prefix}_aligned_abs_coeff_cv"] = coefficient_abs_cv(case_row)
        output[f"{prefix}_candidate_id"] = case_row.get("candidate_id", "")
        output[f"{prefix}_run_root"] = case_row.get("run_root", "")
        output[f"{prefix}_selection_reason"] = case_row.get("selection_reason", "")
    return output


def make_contact_sheets(manifest_rows: list[dict[str, Any]], out_dir: Path) -> None:
    by_rms: dict[str, list[dict[str, Any]]] = {}
    for row in manifest_rows:
        by_rms.setdefault(rms_label(float(row["target_wavefront_rms"])), []).append(row)

    overview_dir = out_dir / "00_contact_sheets_by_rms"
    overview_dir.mkdir(parents=True, exist_ok=True)
    resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    for label, rows in sorted(by_rms.items()):
        images = []
        for row in sorted(rows, key=lambda item: image_rank(str(item["image"]))):
            path = PROJECT_ROOT / str(row["path"])
            im = Image.open(path).convert("RGB")
            target_width = 880
            target_height = int(round(im.height * target_width / im.width))
            images.append(im.resize((target_width, target_height), resample_filter))
        if not images:
            continue
        gap = 22
        cell_w = max(im.width for im in images)
        cell_h = max(im.height for im in images)
        canvas = Image.new("RGB", (cell_w, len(images) * cell_h + (len(images) - 1) * gap), "white")
        for idx, im in enumerate(images):
            canvas.paste(im, (0, idx * (cell_h + gap)))
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


def write_readme(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# No-RMS vs Amplitude-Direction RCP Comparisons",
        "",
        "Controlled variables:",
        f"- images: {', '.join(args.images)}",
        f"- direction: {args.direction}",
        f"- RMS: {', '.join(f'{x:.2f}' for x in args.rms_values)}",
        "- dimension: 6D",
        "- pretrain/joint iterations and tuned object/model settings follow the source runs.",
        "",
        "Rows in each PNG:",
        f"- top: no RMS prior, direct Seidel, lambda={args.no_rms_lambda:g}",
        "- middle: best amp_direction lambda_a for the same image/RMS",
        "- bottom: best amp_direction_detach_norm lambda_a for the same image/RMS",
        "",
        "Best amp-direction selection rule:",
        f"- among cases with aligned recovered/GT RMS in [{args.rms_band[0]:g}, {args.rms_band[1]:g}], choose lowest operator_error_calibrated",
        "- if no case is in that band, choose closest aligned recovered/GT RMS to 1.0, then lower operator_error_calibrated",
        "",
        f"No-RMS evaluator CSV: `{args.no_rms_csv}`",
        f"Amp-direction evaluator CSV: `{args.amp_csv}`",
        "",
        "Folder layout:",
        "- `rms*/<image>/no_rms_vs_amp_direction__<image>__signed_balanced__<rms>__RCP_vertical.png`",
        "- `00_contact_sheets_by_rms/<rms>_contact_sheet.png`",
        "",
        f"Generated comparison RCP files: {len(rows)}",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['path']}` | amp_lambda={row['amp_direction_lambda']:.0f} | "
            f"detach_lambda={row['amp_direction_detach_norm_lambda']:.0f}"
        )
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rms-csv", type=Path, required=True)
    parser.add_argument("--amp-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cocoa_like_2d_mechanism"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--direction", default="signed_balanced")
    parser.add_argument("--images", nargs="+", default=["Iksung_beads", "dendrites", "dendrites_dense"])
    parser.add_argument("--rms-values", nargs="+", type=float, default=[0.06, 0.20, 0.40])
    parser.add_argument("--no-rms-lambda", type=float, default=0.0)
    parser.add_argument("--rms-band", nargs=2, type=float, default=[0.8, 1.2])
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()

    images = set(args.images)
    rms_values = {round(float(v), 6) for v in args.rms_values}
    no_rms = index_no_rms_rows(
        load_rows(args.no_rms_csv),
        images=images,
        direction=args.direction,
        rms_values=rms_values,
        lambda_value=args.no_rms_lambda,
    )
    amp = select_best_amp_rows(
        load_rows(args.amp_csv),
        images=images,
        direction=args.direction,
        rms_values=rms_values,
        variants=VARIANTS,
        rms_band=(float(args.rms_band[0]), float(args.rms_band[1])),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for image, rms in sorted_case_keys(args.images, args.rms_values):
        no_rms_row = no_rms.get((image, rms))
        amp_row = amp.get(("amp_direction", image, rms))
        detach_row = amp.get(("amp_direction_detach_norm", image, rms))
        if no_rms_row is None or amp_row is None or detach_row is None:
            missing.append(
                f"{image} {args.direction} rms={rms:.2f} "
                f"no_rms={no_rms_row is not None} amp={amp_row is not None} detach={detach_row is not None}"
            )
            continue

        cases = [
            build_case(no_rms_row, args.output_root),
            build_case(amp_row, args.output_root),
            build_case(detach_row, args.output_root),
        ]
        label = rms_label(rms)
        out_path = (
            args.out_dir
            / label
            / safe_name(image)
            / f"no_rms_vs_amp_direction__{safe_name(image)}__{safe_name(args.direction)}__{label}__RCP_vertical.png"
        )
        labels = [
            f"no RMS | lambda={args.no_rms_lambda:g}",
            f"amp_direction | lambda_a={parse_float(amp_row, 'lambda'):g}",
            f"amp_direction_detach_norm | lambda_a={parse_float(detach_row, 'lambda'):g}",
        ]
        manifest_rows.append(
            make_three_row_panel(
                cases=cases,
                labels=labels,
                out_path=out_path,
                title_prefix="Controlled RCP: no RMS vs amplitude-direction",
            )
        )

    write_manifest(manifest_rows, args.out_dir)
    write_readme(manifest_rows, args.out_dir, args)
    if not args.no_contact_sheets:
        make_contact_sheets(manifest_rows, args.out_dir)
    if missing:
        (args.out_dir / "missing_cases.txt").write_text("\n".join(missing) + "\n")
    print(f"[done] wrote {len(manifest_rows)} controlled RCP files to {args.out_dir}")
    if missing:
        print(f"[warn] missing {len(missing)} cases; see {args.out_dir / 'missing_cases.txt'}")


if __name__ == "__main__":
    main()
